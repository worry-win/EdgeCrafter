"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .ec_engine import evaluate, train_one_epoch


def _metric_values(value):
    """Return evaluator output as a list for logging and best-score handling."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, 'tolist'):
        value = value.tolist()
        return value if isinstance(value, list) else [value]
    return [value]


class ECSolver(BaseSolver):

    def _primary_eval_key(self):
        return f'coco_eval_{self.iou_type}'

    def _primary_eval_metric_name(self):
        return f'{self._primary_eval_key()}_map50'

    def _primary_eval_score(self, test_stats):
        metric_values = test_stats.get(self._primary_eval_key())
        if metric_values is None:
            return None
        if isinstance(metric_values, (list, tuple)):
            if len(metric_values) > 1:
                return float(metric_values[1])
            if len(metric_values) == 1:
                return float(metric_values[0])
            return None
        return float(metric_values)

    def _record_eval_best(self, test_stats, epoch):
        state = dict(getattr(self, '_best_eval_state', {}))
        primary_score = self._primary_eval_score(test_stats)
        primary_improved = (
            primary_score is not None
            and primary_score > float(state.get('best_map50', float('-inf')))
        )
        if primary_improved:
            state['best_map50'] = primary_score
            state['best_map50_epoch'] = epoch
            state['best_map50_yolo_f1'] = float(
                test_stats.get('yolo_f1_iou50', {}).get('f1', 0.0)
            )
            state['best_map50_yolo_f1_iou95'] = float(
                test_stats.get('yolo_f1_iou95', {}).get('f1', 0.0)
            )
            state['best_map50_yolo_f1_iou50_95'] = float(
                test_stats.get('yolo_f1_iou50_95', {}).get('f1', 0.0)
            )

        metric_values = _metric_values(test_stats.get(self._primary_eval_key()))
        map50_95 = float(metric_values[0]) if metric_values else None
        if (
            map50_95 is not None
            and map50_95 > float(state.get('best_map50_95', float('-inf')))
        ):
            state['best_map50_95'] = map50_95
            state['best_map50_95_epoch'] = epoch

        self._best_eval_state = state
        return primary_improved

    def _update_early_stop(self, test_stats, min_delta):
        state = dict(getattr(self, '_early_stop_state', {}))
        best_primary_score = float(
            state.get('best_primary_score', float('-inf'))
        )
        no_improve_epochs = int(state.get('no_improve_epochs', 0))
        primary_score = self._primary_eval_score(test_stats)
        if (
            primary_score is not None
            and primary_score > best_primary_score + min_delta
        ):
            best_primary_score = primary_score
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        self._early_stop_state = {
            'best_primary_score': best_primary_score,
            'no_improve_epochs': no_improve_epochs,
        }
        return no_improve_epochs

    def state_dict(self):
        state = super().state_dict()
        if hasattr(self, '_early_stop_state'):
            state['early_stop_state'] = dict(self._early_stop_state)
        if hasattr(self, '_best_eval_state'):
            state['best_eval_state'] = dict(self._best_eval_state)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._best_eval_state = dict(state.get('best_eval_state', {}))
        if self._best_eval_state:
            self._early_stop_state = dict(state.get('early_stop_state', {}))
        else:
            # Older checkpoints monitored mAP50-95 in this field. Rebuild it
            # from the resumed evaluation instead of reusing changed semantics.
            self._early_stop_state = {}

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)
        
        stop_aug_epoch = self.train_dataloader.dataset._transforms.stop_epoch  # epoch to stop augmentation
        if args.lrsheduler is not None:
            no_aug_epochs = args.epochs - stop_aug_epoch
            flat_epochs = self.train_dataloader.dataset._transforms.mosaic_epoch if args.flat_epoch is None else args.flat_epoch
            iter_per_epoch = len(self.train_dataloader)
            warmup_iter = min(args.warmup_iter, 3 * iter_per_epoch)  
            
            print(f'FlatCosineLRScheduler with flat_epochs: {flat_epochs}, no_aug_epochs: {no_aug_epochs}, warmup_iter: {args.warmup_iter}')
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epochs, 
                                                warmup_iter=warmup_iter, flat_epochs=flat_epochs, no_aug_epochs=no_aug_epochs)
            self.self_lr_scheduler = True
        else:
            self.self_lr_scheduler = False

        self._best_eval_state = dict(getattr(self, '_best_eval_state', {}))
        self._early_stop_state = dict(getattr(self, '_early_stop_state', {}))
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            self._record_eval_best(test_stats, self.last_epoch)
            if not self._early_stop_state:
                primary_score = self._primary_eval_score(test_stats)
                self._early_stop_state = {
                    'best_primary_score': (
                        primary_score
                        if primary_score is not None
                        else float('-inf')
                    ),
                    'no_improve_epochs': 0,
                }
            print(f'best_stat: {self._best_eval_state}')

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
                
            if epoch == stop_aug_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / 'best.pth'))
                self.last_epoch = epoch - 1
                self._early_stop_state = {
                    'best_primary_score': float(
                        self._best_eval_state.get('best_map50', float('-inf'))
                    ),
                    'no_improve_epochs': 0,
                }

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < stop_aug_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            for k, value in test_stats.items():
                values = _metric_values(value)
                if not values:
                    continue
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(values):
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f'Test/{k}_{i}', v, epoch)

            primary_improved = self._record_eval_best(test_stats, epoch)
            no_improve_epochs = self._update_early_stop(
                test_stats,
                args.early_stop_min_delta,
            )
            if primary_improved and self.output_dir:
                dist_utils.save_on_master(
                    self.state_dict(),
                    self.output_dir / 'best.pth',
                )
            print(f'best_stat: {self._best_eval_state}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if self.iou_type in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval[self.iou_type].eval,
                                    self.output_dir / "eval" / name)
            if self.output_dir:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')
            if args.early_stop_patience and no_improve_epochs >= args.early_stop_patience:
                print(
                    f'Early stopping after {no_improve_epochs} epochs without improvement '
                    f'on {self._primary_eval_metric_name()}.'
                )
                break
            if torch.cuda.is_available():  # Just for clearing up GPU memory. You can remove it if you have enough GPU memory.
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval[self.iou_type].eval, self.output_dir / "eval.pth")

        return
