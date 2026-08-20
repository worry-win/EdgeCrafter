"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import math
import sys
import warnings
from typing import Iterable

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA
from .metrics_format import format_yolo_per_class_metrics_table


def _max_macro_pr_curve_f1(precision, recalls):
    valid = precision > -1
    if not valid.any():
        return -1.0, -1.0
    p_sum = np.where(valid, precision, 0.0).sum(axis=1)
    p_count = valid.sum(axis=1)
    p_mean = np.divide(p_sum, p_count, out=np.zeros_like(p_sum), where=p_count > 0)
    f1 = 2 * p_mean * recalls / np.maximum(p_mean + recalls, 1e-12)
    index = int(np.argmax(f1))
    return float(f1[index]), float(recalls[index])


def summarize_pr_curve_f1(coco_eval):
    """Summarize maximum macro-F1 across the standard COCO IoU sweep."""
    precision = coco_eval.eval['precision'][:, :, :, 0, -1]
    recalls = coco_eval.params.recThrs
    iou_thresholds = coco_eval.params.iouThrs
    target_thresholds = np.arange(0.50, 0.951, 0.05)

    f1_by_iou = []
    recall_by_iou = []
    for threshold in target_thresholds:
        matches = np.flatnonzero(np.isclose(iou_thresholds, threshold))
        if len(matches) != 1:
            raise ValueError(f"COCO evaluator is missing IoU threshold {threshold:.2f}")
        f1, recall = _max_macro_pr_curve_f1(precision[matches[0]], recalls)
        f1_by_iou.append(f1)
        recall_by_iou.append(recall)

    return {
        'f1_iou50': f1_by_iou[0],
        'recall_iou50': recall_by_iou[0],
        'f1_iou95': f1_by_iou[-1],
        'recall_iou95': recall_by_iou[-1],
        'f1_iou50_95_mean': float(np.mean(f1_by_iou)),
    }


def summarize_yolo_pr_curve_metrics(coco_eval, coco_gt):
    """Select an independent best PR operating point for every category."""
    precisions = coco_eval.eval.get('precision')
    scores = coco_eval.eval.get('scores')
    if precisions is None or scores is None:
        warnings.warn(
            'Skipping WZW-aligned F1 because COCO precision/scores are unavailable.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    rec_thrs = coco_eval.params.recThrs
    iou_thrs = np.asarray(coco_eval.params.iouThrs)
    cat_ids = list(coco_eval.params.catIds)
    iou50_idx = int(np.argmin(np.abs(iou_thrs - 0.50)))
    iou95_idx = int(np.argmin(np.abs(iou_thrs - 0.95)))
    if not (
        np.isclose(iou_thrs[iou50_idx], 0.50)
        and np.isclose(iou_thrs[iou95_idx], 0.95)
    ):
        warnings.warn(
            'Skipping WZW-aligned F1 because COCO IoU thresholds 0.50 and 0.95 '
            'are required.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def best_at_iou(iou_idx, class_idx):
        p_curve = precisions[iou_idx, :, class_idx, 0, -1]
        s_curve = scores[iou_idx, :, class_idx, 0, -1]
        valid = p_curve > -1
        if not np.any(valid):
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'confidence': 0.0,
                'ap': 0.0,
            }

        p_valid = p_curve[valid]
        r_valid = rec_thrs[valid]
        s_valid = s_curve[valid]
        f1_curve = 2 * p_valid * r_valid / (p_valid + r_valid + 1e-16)
        best_idx = int(np.argmax(f1_curve))
        return {
            'precision': float(p_valid[best_idx]),
            'recall': float(r_valid[best_idx]),
            'f1': float(f1_curve[best_idx]),
            'confidence': float(s_valid[best_idx]),
            'ap': float(np.mean(p_valid)),
        }

    def aggregate(class_metrics):
        if not class_metrics:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'map50': 0.0,
            }

        mean_p = float(np.mean([metric['precision'] for metric in class_metrics]))
        mean_r = float(np.mean([metric['recall'] for metric in class_metrics]))
        return {
            'precision': mean_p,
            'recall': mean_r,
            'f1': float(2 * mean_p * mean_r / (mean_p + mean_r + 1e-16)),
            'map50': float(np.mean([metric['ap'] for metric in class_metrics])),
        }

    metrics_by_iou = [
        [
            best_at_iou(iou_idx, class_idx)
            for class_idx in range(len(cat_ids))
        ]
        for iou_idx in range(len(iou_thrs))
    ]
    overall50 = aggregate(metrics_by_iou[iou50_idx])
    overall95 = aggregate(metrics_by_iou[iou95_idx])
    overall5095 = {
        key: float(np.mean([
            aggregate(class_metrics)[key]
            for class_metrics in metrics_by_iou
        ]))
        for key in ('precision', 'recall', 'f1', 'map50')
    }

    cats = coco_gt.loadCats(cat_ids)
    cat_name_by_id = {
        int(cat['id']): cat.get('name', str(cat['id']))
        for cat in cats
    }
    yolo_per_class = {}
    for class_idx, cat_id in enumerate(cat_ids):
        metric50 = metrics_by_iou[iou50_idx][class_idx]
        metric95 = metrics_by_iou[iou95_idx][class_idx]
        f1_5095 = float(np.mean([
            metrics[class_idx]['f1']
            for metrics in metrics_by_iou
        ]))
        cat_id = int(cat_id)
        yolo_per_class[cat_id] = {
            'name': cat_name_by_id.get(cat_id, str(cat_id)),
            'precision': metric50['precision'],
            'recall': metric50['recall'],
            'f1': metric50['f1'],
            'confidence': metric50['confidence'],
            'map50': metric50['ap'],
            'f1_iou95': metric95['f1'],
            'f1_iou50_95': f1_5095,
        }

    return {
        'yolo_per_class': yolo_per_class,
        'yolo_overall': overall50,
        'yolo_f1_iou50': overall50,
        'yolo_f1_iou95': overall95,
        'yolo_f1_iou50_95': overall5095,
        'yolo_overall_prf_map50': [
            overall50['precision'],
            overall50['recall'],
            overall50['f1'],
            overall50['map50'],
        ],
    }


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator.labels is not None:
        
        try:
            from tabulate import tabulate
        except ImportError:
            def tabulate(rows, headers, tablefmt=None):
                values = [headers, *rows]
                widths = [max(len(str(row[i])) for row in values) for i in range(len(headers))]

                def format_row(row):
                    return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))

                separator = "-+-".join("-" * width for width in widths)
                return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])
        
        res_per_type = {}
        headers = ['class']

        for iou_type in coco_evaluator.iou_types:
            if iou_type not in coco_evaluator.coco_eval:
                continue
            
            precisions = coco_evaluator.coco_eval[iou_type].eval['precision']
            ap = np.mean(precisions[..., 0, -1], axis=(0, 1)) * 100
            ap_50 = np.mean(precisions[0, :, :, 0, -1], axis=0) * 100
            
            prefix = 'bbox' if iou_type == 'bbox' else 'segm'
            headers.extend([f'{prefix}-AP', f'{prefix}-AP50'])
            res_per_type[iou_type] = (ap, ap_50)

        # Construct rows by merging metrics for each class
        table_data = []
        for k, name in enumerate(coco_evaluator.labels):
            row = [name]
            for iou_type in coco_evaluator.iou_types:
                if iou_type in res_per_type:
                    ap, ap_50 = res_per_type[iou_type]
                    row.extend([f'{ap[k]:.2f}', f'{ap_50[k]:.2f}'])
            table_data.append(row)

        print(f"\n### Class-wise Evaluation Metrics ###")
        print(tabulate(table_data, headers=headers, tablefmt='pretty'))
        
    
    if coco_evaluator is not None:
        if 'segm' in iou_types:
            stats['coco_eval_mask'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        elif 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            f1 = summarize_pr_curve_f1(coco_evaluator.coco_eval['bbox'])
            stats['coco_eval_bbox_f1'] = f1['f1_iou50']
            stats['coco_eval_bbox_f1_recall'] = f1['recall_iou50']
            stats['coco_eval_bbox_f1_iou95'] = f1['f1_iou95']
            stats['coco_eval_bbox_f1_iou95_recall'] = f1['recall_iou95']
            stats['coco_eval_bbox_f1_iou50_95_mean'] = f1['f1_iou50_95_mean']
            print(
                f"bbox-macro-F1@IoU50(PR-curve): {f1['f1_iou50']:.6f} "
                f"(recall-grid={f1['recall_iou50']:.3f})"
            )
            print(
                f"bbox-macro-F1@IoU95(PR-curve): {f1['f1_iou95']:.6f} "
                f"(recall-grid={f1['recall_iou95']:.3f})"
            )
            print(
                "bbox-macro-F1@IoU50:95(PR-curve mean): "
                f"{f1['f1_iou50_95_mean']:.6f}"
            )

            yolo_stats = summarize_yolo_pr_curve_metrics(
                coco_evaluator.coco_eval['bbox'],
                coco_evaluator.coco_gt,
            )
            if yolo_stats is not None:
                stats.update(yolo_stats)
                overall50 = yolo_stats['yolo_f1_iou50']
                overall95 = yolo_stats['yolo_f1_iou95']
                overall5095 = yolo_stats['yolo_f1_iou50_95']
                print(
                    f"[YOLO Overall] "
                    f"P={overall50['precision']:.4f}, "
                    f"R={overall50['recall']:.4f}, "
                    f"F1@0.50={overall50['f1']:.4f}, "
                    f"F1@0.95={overall95['f1']:.4f}, "
                    f"F1@50:95={overall5095['f1']:.4f}, "
                    f"mAP@50={overall50['map50']:.4f}"
                )
                table = format_yolo_per_class_metrics_table(
                    yolo_stats['yolo_per_class']
                )
                if table is not None:
                    print(table)

    return stats, coco_evaluator
