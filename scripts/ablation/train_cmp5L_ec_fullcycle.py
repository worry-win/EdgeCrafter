"""Opt-in EC-X0..X5 full-cycle train/validation entrypoint (seed 42 only)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ecdetseg'), str(ROOT)]

from engine.core import YAMLConfig, yaml_utils  # noqa: E402
from engine.misc import MetricLogger, SmoothedValue, dist_utils  # noqa: E402
from engine.solver.ec_engine import _BatchSlice, _optimizer_step_due  # noqa: E402
from engine.solver.ec_solver import ECSolver  # noqa: E402
import engine.solver.ec_solver as ec_solver_module  # noqa: E402
from scripts.ablation.cmp5L_ec_fullcycle_core import (  # noqa: E402
    augmentation_mask, calibration_coefficient_from_rank_ratios,
    pack_numpy_rng_state, partition_detection_losses,
    resolve_calibration_batch_path, stage_ramp, unpack_numpy_rng_state,
)
from scripts.ablation.cmp5L_shared_query_kd import (  # noqa: E402
    candidate_ranking_kd, final_layer_hungarian_matches, one_way_protected_kd,
)
from scripts.ablation.train_cmp5L_shared_query_kd import (  # noqa: E402
    _load_frozen_teacher, _module, _move_targets, _student_forward,
    _targets_absolute_xyxy, _teacher_forward,
)
from scripts.ablation.cmp5L_query_behavior_kd import (  # noqa: E402
    DecoderReplayInputs, _run_decoder_with_capture, run_student_with_capture,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def install_atomic_saver():
    def atomic_save(value, path):
        if not dist_utils.is_main_process():
            return
        destination = Path(path)
        temporary = destination.with_name(destination.name + f'.tmp.{os.getpid()}')
        try:
            torch.save(value, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    dist_utils.save_on_master = atomic_save


def rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def snapshot_rng():
    return {'python': random.getstate(), 'numpy': pack_numpy_rng_state(np.random.get_state()),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(unpack_numpy_rng_state(state['numpy']))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def grad_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    terms = [value.detach().float().square().sum() for value in gradients if value is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def finite_gradients(module):
    return all(torch.isfinite(parameter.grad).all().item() for parameter in module.parameters()
               if parameter.grad is not None)


def ema_state_snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_state_unchanged(before, module):
    current = module.state_dict()
    if before.keys() != current.keys():
        raise RuntimeError('teacher state keys changed')
    for name, old in before.items():
        if not torch.equal(old, current[name]):
            raise RuntimeError(f'teacher forward modified state: {name}')


def teacher_runtime(teacher, arm):
    return SimpleNamespace(teacher=teacher, spec=SimpleNamespace(
        num_queries=300,
        teacher_memory_mode='normal' if arm == 'ECX2' else 'privileged_np',
    ))


def objective(arm, outputs, teacher_logits, targets, matcher, *, global_images, ddp_world):
    matches = final_layer_hungarian_matches(matcher, outputs, targets, normal_query_count=300)
    function = one_way_protected_kd if arm == 'ECX4' else candidate_ranking_kd
    return function(
        outputs['pred_logits'].float(), teacher_logits[-1].float(),
        outputs['pred_boxes'], targets, matches,
        normal_query_count=300, negative_topk=20, negative_max_iou=.3,
        global_image_count=global_images, ddp_world_size=ddp_world,
    )


class Runtime:
    def __init__(self, solver, args):
        self.solver = solver
        self.arm = args.arm
        self.args = args
        self.output = Path(solver.cfg.output_dir)
        self.manifest = json.loads(Path(args.manifest).read_text())
        self.coefficient = solver.calibration.get('coefficient')
        self.teacher = None
        if self.arm == 'ECX1':
            self.teacher = _load_frozen_teacher(_module(solver.model), args.fixed_teacher, solver.device)
        elif self.arm in ('ECX2', 'ECX3', 'ECX4'):
            self.teacher = solver.ema.module
        if self.teacher is not None:
            self.teacher.eval().requires_grad_(False)
        self.fixed_teacher_sha = sha256(args.fixed_teacher) if self.arm == 'ECX1' else None
        self.fixed_teacher_state = (
            {key: value.detach().cpu().clone() for key, value in self.teacher.state_dict().items()}
            if self.arm == 'ECX1' else None
        )
        self.updates = 0
        self.skips = 0
        self.diagnostic_rows = 0
        self.active_branch = None
        self.intervention_microbatches = 0
        self.eligible_microbatches = 0
        self.original_forward = None
        self.x5_seen = 0


ACTIVE: Runtime | None = None


@contextmanager
def frozen_bn_statistics(module):
    batchnorm = [part for part in module.modules()
                 if isinstance(part, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                      torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]
    states = [part.training for part in batchnorm]
    try:
        for part in batchnorm:
            part.eval()
        yield
    finally:
        for part, state in zip(batchnorm, states):
            part.train(state)


def replay_detection_output(transformer, normal_output, replay, raw):
    """Recreate EC's decoder/DN/pre fields; reuse encoder proposals exactly once."""
    out_bboxes, out_logits, out_corners, out_refs, out_masks, pre_bboxes, pre_logits, pre_segs = raw
    meta = replay.denoising_metadata
    if meta is not None:
        sizes = meta['dn_num_split']
        dn_pre_logits, pre_logits = transformer._split(pre_logits, 1, sizes)
        dn_pre_bboxes, pre_bboxes = transformer._split(pre_bboxes, 1, sizes)
        dn_pre_segs, pre_segs = transformer._split(pre_segs, 1, sizes)
        dn_logits, out_logits = transformer._split(out_logits, 2, sizes)
        dn_bboxes, out_bboxes = transformer._split(out_bboxes, 2, sizes)
        dn_masks, out_masks = transformer._split(out_masks, 2, sizes)
        dn_corners, out_corners = transformer._split(out_corners, 2, sizes)
        dn_refs, out_refs = transformer._split(out_refs, 2, sizes)
    output = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1],
              'pred_masks': out_masks[-1] if out_masks is not None else None,
              'enc_aux_outputs': normal_output['enc_aux_outputs'],
              'enc_meta': normal_output['enc_meta']}
    if out_corners is not None:
        output.update({'pred_corners': out_corners[-1], 'ref_points': out_refs[-1],
                       'up': transformer.up, 'reg_scale': transformer.reg_scale})
        output['aux_outputs'] = transformer._set_aux_loss2(
            out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
            out_masks[:-1] if out_masks is not None else None,
            out_corners[-1], out_logits[-1])
    else:
        output['aux_outputs'] = transformer._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
    if transformer.use_pre_outputs:
        output['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes,
                                 'pred_masks': pre_segs}
    if meta is not None:
        if dn_corners is not None:
            output['dn_outputs'] = transformer._set_aux_loss2(
                dn_logits, dn_bboxes, dn_corners, dn_refs, dn_masks,
                dn_corners[-1], dn_logits[-1])
        else:
            output['dn_outputs'] = transformer._set_aux_loss(dn_logits, dn_bboxes)
        if transformer.use_pre_outputs:
            output['dn_pre_outputs'] = {'pred_logits': dn_pre_logits,
                                        'pred_boxes': dn_pre_bboxes,
                                        'pred_masks': dn_pre_segs}
        output['dn_meta'] = meta
    return output


def install_dual_forward(runtime):
    module = _module(runtime.solver.model)
    original = module.forward
    runtime.original_forward = original

    def dual_forward(self, samples, targets=None):
        active = runtime.active_branch
        if active is None:
            return original(samples, targets=targets)
        features = self.forward_features(samples)
        normal, _trace, replay = run_student_with_capture(self.decoder, features, targets)
        if replay.denoising_metadata is not None:
            dn = int(replay.denoising_metadata['dn_num_split'][0])
            attention_mask = replay.attention_mask
            if attention_mask is None or not bool(attention_mask[dn:, :dn].all()):
                raise RuntimeError('normal queries may attend to GT denoising queries')
        coefficients = active['coefficients']
        masked_features = [feature * augmentation_mask(
            targets, feature.shape[-2], feature.shape[-1], coefficients).to(feature.dtype)
            for feature in features]
        with frozen_bn_statistics(self.decoder):
            memory, shapes = self.decoder._get_encoder_input(masked_features)
            if tuple(shapes) != tuple(replay.spatial_shapes):
                raise RuntimeError('intervention feature spatial shape changed')
            replay_input = DecoderReplayInputs(
                initial_query=replay.initial_query,
                initial_reference_unactivated=replay.initial_reference_unactivated,
                memory=memory, spatial_shapes=shapes,
                attention_mask=replay.attention_mask,
                denoising_metadata=replay.denoising_metadata,
                normal_query_count=300,
            )
            raw, _ = _run_decoder_with_capture(self.decoder, replay_input)
        intervention = replay_detection_output(self.decoder, normal, replay, raw)
        return {'normal': normal, 'intervention': intervention}

    module.forward = MethodType(dual_forward, module)


def x5_branch_choice(runtime, epoch, index, steps, targets, device):
    global_step = epoch * steps + index
    selected = torch.zeros(1, device=device, dtype=torch.int32)
    if rank() == 0:
        generator = torch.Generator(device='cpu').manual_seed(910000 + global_step)
        selected[0] = int(runtime.args.smoke or torch.rand((), generator=generator).item() < .5)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(selected, src=0)
    has_gt = torch.tensor([int(any(len(item['labels']) for item in targets))], device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(has_gt, op=torch.distributed.ReduceOp.MAX)
    runtime.eligible_microbatches += int(has_gt.item())
    enabled = bool(selected.item() and has_gt.item())
    if not enabled:
        return None
    runtime.intervention_microbatches += 1
    generator = torch.Generator(device='cpu').manual_seed(920000 + global_step * world_size() + rank())
    coefficients = (.3 + .5 * torch.rand(len(targets), generator=generator)).tolist()
    return {'coefficients': coefficients}


def x5_eval_identity_parity(runtime):
    model = _module(runtime.solver.model)
    saved_rng = snapshot_rng()
    model.eval()
    try:
        batch_path = resolve_calibration_batch_path(
            runtime.args.manifest, runtime.manifest['batches'][0]['path'])
        batch = torch.load(batch_path, map_location='cpu', weights_only=True)
        samples = batch['samples'].to(runtime.solver.device)
        targets = _move_targets(batch['targets'], runtime.solver.device)
        with torch.no_grad():
            features = model.forward_features(samples)
            normal, _, replay = run_student_with_capture(model.decoder, features, targets)
            masked = [feature * augmentation_mask(
                [{'boxes': torch.empty(0, 4, device=feature.device)} for _ in targets],
                feature.shape[-2], feature.shape[-1], [.5] * len(targets)).to(feature.dtype)
                for feature in features]
            with frozen_bn_statistics(model.decoder):
                memory, shapes = model.decoder._get_encoder_input(masked)
                direct = DecoderReplayInputs(replay.initial_query, replay.initial_reference_unactivated,
                                             memory, shapes, replay.attention_mask,
                                             replay.denoising_metadata, 300)
                raw, _ = _run_decoder_with_capture(model.decoder, direct)
            logits_delta = float((normal['pred_logits'] - raw[1][-1]).abs().max())
            boxes_delta = float((normal['pred_boxes'] - raw[0][-1]).abs().max())
            if logits_delta > 1e-5 or boxes_delta > 1e-5:
                raise RuntimeError(f'X5 eval all-one replay parity failed: {logits_delta}, {boxes_delta}')
            return {'logits_max_abs': logits_delta, 'boxes_max_abs': boxes_delta}
    finally:
        model.train()
        restore_rng(saved_rng)


def calibrate(runtime):
    if runtime.arm not in ('ECX1', 'ECX2', 'ECX3', 'ECX4') or runtime.coefficient is not None:
        return
    saved_rng = snapshot_rng()
    teacher_before = ema_state_snapshot(runtime.teacher)
    ema_updates = runtime.solver.ema.updates
    model = copy.deepcopy(_module(runtime.solver.model)).to(runtime.solver.device)
    model.train()
    criterion = copy.deepcopy(runtime.solver.criterion).to(runtime.solver.device)
    criterion.train()
    l3 = [parameter for name, parameter in model.named_parameters()
          if parameter.requires_grad and name.startswith('decoder.decoder.layers.3.')]
    if not l3:
        raise RuntimeError('decoder L3 calibration parameter bucket is empty')
    rows, ratios = [], []
    classes = {str(index): 0 for index in range(4)}
    try:
        for index, record in enumerate(runtime.manifest['batches']):
            batch_path = resolve_calibration_batch_path(runtime.args.manifest, record['path'])
            if sha256(batch_path) != record['sha256']:
                raise RuntimeError(f'calibration batch hash changed: {record["path"]}')
            batch = torch.load(batch_path, map_location='cpu', weights_only=True)
            samples = batch['samples'].to(runtime.solver.device)
            targets = _move_targets(batch['targets'], runtime.solver.device)
            absolute = _targets_absolute_xyxy(batch['targets'], samples.shape[-2:], runtime.solver.device)
            with torch.autocast(device_type='cuda', enabled=True):
                outputs, _trace, replay, topk = _student_forward(model, samples, targets)
                _, audit = _teacher_forward(teacher_runtime(runtime.teacher, runtime.arm),
                                            replay, topk, samples, absolute, True)
            with torch.autocast(device_type='cuda', enabled=False):
                det_terms = criterion(outputs, targets, epoch=10, step=index,
                                      global_step=10 * len(runtime.solver.train_dataloader) + index,
                                      epoch_step=len(runtime.solver.train_dataloader))
                det = sum(det_terms.values())
                result = objective(runtime.arm, outputs, audit['teacher_logits'], targets,
                                   criterion.matcher, global_images=len(targets), ddp_world=1)
            valid = (result['lesion_active_count'] + result['negative_active_dimension_count']
                     if runtime.arm == 'ECX4' else result['valid_pair_count'])
            row = {'image_ids': record['image_ids'], 'class_ids': record['class_ids'],
                   'purpose': record['purpose'], 'valid_targets': valid,
                   'det_loss': float(det.detach()), 'new_loss': float(result['loss'].detach()),
                   'topk_calls': audit['topk_calls'], 'alignment': audit['alignment']}
            if record['purpose'] == 'empty_zero_check':
                if valid or row['new_loss'] != 0 or any(len(item['labels']) for item in targets):
                    raise RuntimeError('locked empty calibration batch is not zero')
            elif valid:
                det_norm = grad_norm(det, l3)
                new_norm = grad_norm(result['loss'], l3)
                if not all(math.isfinite(item) and item > 0 for item in (det_norm, new_norm)):
                    raise RuntimeError('zero/nonfinite calibration gradient')
                row['rho'] = new_norm / det_norm
                ratios.append(row['rho'])
                for key, count in result['matched_class_counts'].items():
                    classes[key] += int(count)
            rows.append(row)
        if len(ratios) < 3 or any(value == 0 for value in classes.values()):
            raise RuntimeError(f'insufficient effective calibration: valid={len(ratios)}, classes={classes}')
        ratios_by_rank = [None for _ in range(world_size())]
        if torch.distributed.is_initialized():
            torch.distributed.all_gather_object(ratios_by_rank, ratios)
        else:
            ratios_by_rank[0] = ratios
        coefficient, median = calibration_coefficient_from_rank_ratios(ratios_by_rank)
        pooled_ratios = [value for rank_values in ratios_by_rank for value in rank_values]
        runtime.coefficient = coefficient
        runtime.solver.calibration = {'coefficient': coefficient, 'median_raw_ratio': median,
                                      'valid_batches': len(pooled_ratios), 'matched_classes': classes,
                                      'manifest_sha256': sha256(runtime.args.manifest),
                                      'weighted_median_ratio': statistics.median([coefficient * item for item in pooled_ratios]),
                                      'ratios_by_rank': ratios_by_rank,
                                      'rows': rows}
        if rank() == 0:
            path = runtime.output / 'calibration.json'
            path.write_text(json.dumps(runtime.solver.calibration, indent=2, default=str) + '\n')
    finally:
        del model, criterion
        restore_rng(saved_rng)
    assert_state_unchanged(teacher_before, runtime.teacher)
    if runtime.solver.ema.updates != ema_updates:
        raise RuntimeError('calibration updated EMA counter')
    value = torch.tensor([runtime.coefficient], device=runtime.solver.device)
    if torch.distributed.is_initialized():
        all_values = [torch.empty_like(value) for _ in range(world_size())]
        torch.distributed.all_gather(all_values, value)
        if not all(torch.allclose(other, value, rtol=1e-6, atol=0) for other in all_values):
            raise RuntimeError('calibration coefficient differs across ranks')


def train_one_epoch_ecfull(self_lr_scheduler, lr_scheduler, model, criterion, data_loader,
                           optimizer, device, epoch, max_norm=0, **kwargs):
    runtime = ACTIVE
    if runtime is None:
        raise RuntimeError('EC full-cycle runtime missing')
    model.train()
    criterion.train()
    if runtime.teacher is not None:
        runtime.teacher.eval()
    ema = kwargs.get('ema')
    scaler = kwargs.get('scaler')
    if scaler is None:
        raise RuntimeError('full-cycle protocol requires AMP GradScaler')
    accumulation = max(1, int(kwargs.get('gradient_accumulation_steps', 1)))
    start_step = int(kwargs.get('start_step', 0))
    logger = MetricLogger(delimiter='  ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    optimizer.zero_grad(set_to_none=True)
    steps = len(data_loader)
    counts = {'matched': 0, 'valid_pairs': 0, 'reverse_pairs': 0,
              'positive_active': 0, 'negative_active': 0, 'empty_images': 0,
              **{f'matched_class_{category}': 0 for category in range(4)},
              **{f'active_class_{category}': 0 for category in range(4)}}
    x5_before = runtime.intervention_microbatches
    x5_eligible_before = runtime.eligible_microbatches
    for offset, (samples, raw_targets) in enumerate(
        logger.log_every(_BatchSlice(data_loader, start_step=start_step),
                         kwargs.get('print_freq', 500), f'Epoch: [{epoch}]')
    ):
        index = start_step + offset
        ramp = stage_ramp(epoch, index, steps, runtime.solver.cfg.epochs)
        if runtime.args.smoke and runtime.arm != 'ECX0':
            ramp = .5
        if ramp > 0 and runtime.arm in ('ECX1', 'ECX2', 'ECX3', 'ECX4') and runtime.coefficient is None:
            calibrate(runtime)
        samples = samples.to(device)
        targets = _move_targets(raw_targets, device)
        meta = {'epoch': epoch, 'step': index, 'global_step': epoch * steps + index,
                'epoch_step': steps}
        with torch.autocast(device_type='cuda', enabled=True):
            branch = None
            if ramp > 0 and runtime.arm == 'ECX5':
                branch = x5_branch_choice(runtime, epoch, index, steps, targets, device)
                runtime.active_branch = branch
            if ramp > 0 and runtime.arm in ('ECX1', 'ECX2', 'ECX3', 'ECX4'):
                outputs, _trace, replay, topk = _student_forward(model, samples, targets)
                teacher_before = ema_state_snapshot(runtime.teacher) if runtime.diagnostic_rows < 2 else None
                absolute = _targets_absolute_xyxy(raw_targets, samples.shape[-2:], device)
                _, audit = _teacher_forward(teacher_runtime(runtime.teacher, runtime.arm),
                                            replay, topk, samples, absolute, True)
                if teacher_before is not None:
                    assert_state_unchanged(teacher_before, runtime.teacher)
                    runtime.diagnostic_rows += 1
            else:
                try:
                    outputs = model(samples, targets=targets)
                finally:
                    runtime.active_branch = None
                audit = None
        with torch.autocast(device_type='cuda', enabled=False):
            normal_outputs = outputs['normal'] if branch is not None else outputs
            losses = criterion(normal_outputs, targets, **meta)
            det = sum(losses.values())
            extra = det.sum() * 0
            result = None
            if audit is not None:
                image_count = torch.tensor([len(targets)], device=device)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(image_count)
                result = objective(runtime.arm, outputs, audit['teacher_logits'], targets,
                                   criterion.matcher, global_images=int(image_count.item()),
                                   ddp_world=world_size())
                extra = ramp * runtime.coefficient * result['loss']
            if branch is not None:
                intervention_losses = criterion(outputs['intervention'], targets, **meta)
                normal_encoder, normal_decoder = partition_detection_losses(losses)
                intervention_encoder, intervention_decoder = partition_detection_losses(intervention_losses)
                if not normal_encoder or set(normal_encoder) != set(intervention_encoder) or set(normal_decoder) != set(intervention_decoder):
                    raise RuntimeError('X5 detection-loss family mismatch')
                for key in normal_encoder:
                    if not torch.allclose(normal_encoder[key], intervention_encoder[key], rtol=1e-5, atol=1e-6):
                        raise RuntimeError(f'X5 encoder loss changed during decoder replay: {key}')
                alpha = .25 * ramp
                extra = alpha * (sum(intervention_decoder.values()) - sum(normal_decoder.values()))
                runtime.x5_seen += 1
            total = det + extra
        if result is not None:
            for category, value in result['matched_class_counts'].items():
                counts[f'matched_class_{category}'] += int(value)
            counts['empty_images'] += int(result['empty_image_count'])
            if runtime.arm == 'ECX4':
                counts['matched'] += int(result['lesion_count'])
                counts['positive_active'] += int(result['lesion_active_count'])
                counts['negative_active'] += int(result['negative_active_dimension_count'])
                for category, value in result['active_class_counts'].items():
                    counts[f'active_class_{category}'] += int(value)
            else:
                counts['matched'] += sum(int(value) for value in result['matched_class_counts'].values())
                counts['valid_pairs'] += int(result['valid_pair_count'])
                counts['reverse_pairs'] += int(result['reverse_pair_count'])
                for category, value in result['valid_class_counts'].items():
                    counts[f'active_class_{category}'] += int(value)
        gradient_record = None
        if (epoch in (10, 20, 50, 90) and index == 0 and ramp > 0 and
                (result is not None or branch is not None)):
            l3 = [parameter for name, parameter in _module(model).named_parameters()
                  if parameter.requires_grad and name.startswith('decoder.decoder.layers.3.')]
            gradient_record = {'det_l3': grad_norm(det, l3),
                               'weighted_new_l3': grad_norm(extra, l3)}
            gradient_record['ratio'] = (gradient_record['weighted_new_l3'] /
                                        gradient_record['det_l3'] if gradient_record['det_l3'] else None)
        scaler.scale(total / accumulation).backward()
        step_due = _optimizer_step_due(index, steps, accumulation)
        success = False
        if step_due:
            scaler.unscale_(optimizer)
            if not finite_gradients(_module(model)):
                # GradScaler may skip an overflowing update, but never silently
                # accept a nonfinite gradient as a successful model update.
                pass
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            success = float(scaler.get_scale()) >= before
            if torch.distributed.is_initialized():
                status = torch.tensor([int(success)], device=device)
                minimum, maximum = status.clone(), status.clone()
                torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
                if int(minimum) != int(maximum):
                    raise RuntimeError('AMP update success differed across DDP ranks')
            optimizer.zero_grad(set_to_none=True)
            if success:
                runtime.updates += 1
                if ema is not None:
                    ema.update(model)
                if self_lr_scheduler:
                    optimizer = lr_scheduler.step(epoch * steps + index, optimizer)
                elif kwargs.get('lr_warmup_scheduler') is not None:
                    kwargs['lr_warmup_scheduler'].step()
            else:
                runtime.skips += 1
            interval = int(kwargs.get('checkpoint_interval_steps', 0))
            callback = kwargs.get('checkpoint_callback')
            if success and callback and interval and (index + 1) % interval == 0:
                callback(epoch, index + 1)
        if not torch.isfinite(total).all():
            raise FloatingPointError('nonfinite full-cycle loss')
        losses['loss_ec_new_weighted'] = extra.detach()
        reduced = dist_utils.reduce_dict(losses)
        logger.update(loss=sum(reduced.values()), **reduced)
        logger.update(lr=optimizer.param_groups[0]['lr'])
        if rank() == 0 and (index == 0 or index % 500 == 0 or runtime.args.smoke):
            row = {'arm': runtime.arm, 'epoch': epoch, 'batch': index,
                   'ramp': ramp, 'det': float(det.detach()), 'extra_raw': float(result['loss'].detach()) if result else 0.,
                   'extra_weighted': float(extra.detach()), 'coefficient': runtime.coefficient,
                   'updates': runtime.updates, 'amp_skips': runtime.skips,
                   'ema_updates': ema.updates if ema is not None else None,
                   'teacher_topk_calls': audit['topk_calls'] if audit else None,
                   'teacher_memory_delta': audit['memory_max_abs_delta'] if audit else None,
                   'x5_intervention': branch is not None,
                   'x5_coefficients': branch['coefficients'] if branch else None,
                   'x5_eligible_microbatches': runtime.eligible_microbatches,
                   'x5_intervention_microbatches': runtime.intervention_microbatches,
                   'gradient_l3_unscaled': gradient_record}
            with (runtime.output / 'ecfull_diagnostics.jsonl').open('a') as handle:
                handle.write(json.dumps(row, default=str) + '\n')
        if runtime.args.smoke and runtime.updates >= 5:
            break
        if runtime.args.smoke and index >= 80:
            raise RuntimeError('smoke failed to obtain five real AMP optimizer updates')
    logger.synchronize_between_processes()
    if runtime.fixed_teacher_state is not None:
        for key, value in runtime.teacher.state_dict().items():
            if not torch.equal(runtime.fixed_teacher_state[key], value.detach().cpu()):
                raise RuntimeError(f'fixed teacher changed during epoch: {key}')
    if runtime.teacher is not None and any(parameter.grad is not None for parameter in runtime.teacher.parameters()):
        raise RuntimeError('teacher received gradient')
    keys = list(counts)
    tensor = torch.tensor([counts[key] for key in keys], device=device, dtype=torch.long)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor)
    counts = dict(zip(keys, [int(value) for value in tensor.tolist()]))
    if rank() == 0:
        with (runtime.output / 'ecfull_epoch_summary.jsonl').open('a') as handle:
            handle.write(json.dumps({'arm': runtime.arm, 'epoch': epoch,
                                     'updates': runtime.updates, 'amp_skips': runtime.skips,
                                     'ema_updates': ema.updates if ema is not None else None,
                                     'counts': counts,
                                     'x5_eligible_batches': runtime.eligible_microbatches - x5_eligible_before,
                                     'x5_intervention_batches': runtime.intervention_microbatches - x5_before,
                                     'coefficient': runtime.coefficient}, sort_keys=True) + '\n')
    return {key: meter.global_avg for key, meter in logger.meters.items()}


class ECFullSolver(ECSolver):
    def __init__(self, cfg, args):
        super().__init__(cfg)
        self.args = args
        self.calibration = {}

    def state_dict(self):
        state = super().state_dict()
        state['ec_fullcycle_calibration'] = self.calibration
        state['ec_fullcycle_rng'] = snapshot_rng()
        state['ec_fullcycle_init_sha256'] = self.args.init_sha256
        return state

    def load_state_dict(self, state):
        if state.get('ec_fullcycle_init_sha256', self.args.init_sha256) != self.args.init_sha256:
            raise RuntimeError('resume initialization hash differs')
        super().load_state_dict(state)
        self.calibration = dict(state.get('ec_fullcycle_calibration', {}))
        if 'ec_fullcycle_rng' in state:
            restore_rng(state['ec_fullcycle_rng'])

    def train(self):
        global ACTIVE
        super().train()
        ACTIVE = Runtime(self, self.args)
        if self.args.arm == 'ECX5':
            install_dual_forward(ACTIVE)
        if self.args.arm != 'ECX0' and self.args.arm != 'ECX5' and ACTIVE.teacher is None:
            raise RuntimeError('required teacher missing')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--arm', required=True, choices=[f'ECX{i}' for i in range(6)])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--fixed-teacher')
    parser.add_argument('--init-sha256', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--print-rank', type=int, default=0)
    parser.add_argument('--print-method', default='builtin')
    parser.add_argument('--local-rank', type=int)
    return parser.parse_args()


def main(args):
    if args.seed != 42 or not args.use_amp:
        raise RuntimeError('locked EC full-cycle recipe is seed42/AMP')
    if args.arm == 'ECX1' and (not args.fixed_teacher or not Path(args.fixed_teacher).is_file()):
        raise RuntimeError('fixed mature teacher missing')
    if sha256(args.resume) != args.init_sha256 and not args.smoke:
        # A genuine resumable training checkpoint has its own hash; the
        # initialization hash is verified from checkpoint metadata instead.
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('ec_fullcycle_init_sha256') != args.init_sha256:
            raise RuntimeError('resume checkpoint belongs to a different common init')
    mp.set_sharing_strategy(os.environ.get('EC_MP_SHARING_STRATEGY', 'file_descriptor'))
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    cfg = YAMLConfig(args.config, resume=args.resume, seed=args.seed, use_amp=True,
                     output_dir=args.output_dir)
    actual = cfg.yaml_cfg
    if actual['ec_fullcycle_arm'] != args.arm or actual['epochs'] != 100 or actual['early_stop_patience'] != 0:
        raise RuntimeError('arm/full-cycle configuration mismatch')
    if actual['train_dataloader']['dataset']['transforms']['mosaic_epoch'] != 24 or actual['train_dataloader']['collate_fn']['mixup_epoch'] != 24 or actual['train_dataloader']['dataset']['transforms']['stop_epoch'] != 98:
        raise RuntimeError('augmentation schedule mismatch')
    if actual['train_dataloader']['total_batch_size'] != 16 or actual['gradient_accumulation_steps'] != 2 or world_size() != 2:
        raise RuntimeError('DDP/effective batch contract mismatch')
    ec_solver_module.train_one_epoch = train_one_epoch_ecfull
    install_atomic_saver()
    solver = ECFullSolver(cfg, args)
    if args.smoke:
        solver.train()
        solver.train_dataloader.set_epoch(0)
        parity = x5_eval_identity_parity(ACTIVE) if args.arm == 'ECX5' else None
        solver.model.train()
        train_one_epoch_ecfull(False, solver.lr_scheduler, solver.model, solver.criterion,
                               solver.train_dataloader, solver.optimizer, solver.device, 0,
                               max_norm=solver.cfg.clip_max_norm, ema=solver.ema,
                               scaler=solver.scaler, gradient_accumulation_steps=2,
                               print_freq=1000)
        if ACTIVE.updates < 5 or ACTIVE.teacher is not None and any(p.grad is not None for p in ACTIVE.teacher.parameters()):
            raise RuntimeError('smoke update/teacher-gradient gate failed')
        if args.arm == 'ECX5' and ACTIVE.x5_seen == 0:
            raise RuntimeError('X5 smoke never executed the intervention branch')
        if rank() == 0:
            (Path(args.output_dir) / 'SMOKE_COMPLETED.json').write_text(json.dumps({
                'arm': args.arm, 'updates': ACTIVE.updates, 'amp_skips': ACTIVE.skips,
                'calibration': solver.calibration,
                'teacher_gradient_count': sum(p.grad is not None for p in ACTIVE.teacher.parameters()) if ACTIVE.teacher else 0,
                'x5_eval_identity_parity': parity,
            }, indent=2, default=str) + '\n')
    else:
        solver.fit()
    dist_utils.cleanup()


if __name__ == '__main__':
    main(parse_args())
