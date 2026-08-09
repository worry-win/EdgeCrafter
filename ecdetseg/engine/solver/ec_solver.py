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

    def state_dict(self):
        state = super().state_dict()
        if hasattr(self, '_early_stop_state'):
            state['early_stop_state'] = dict(self._early_stop_state)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._early_stop_state = dict(state.get('early_stop_state', {}))

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

        top1 = float('-inf')
        best_stat = {'epoch': -1, }
        early_state = getattr(self, '_early_stop_state', {})
        best_primary_score = float(early_state.get('best_primary_score', float('-inf')))
        no_improve_epochs = int(early_state.get('no_improve_epochs', 0))
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
            for k, value in test_stats.items():
                values = _metric_values(value)
                if not values:
                    continue
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = values[0]
                if k == f'coco_eval_{self.iou_type}':
                    top1 = values[0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
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
                no_improve_epochs = 0

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

            primary_key = f'coco_eval_{self.iou_type}'
            for k, value in test_stats.items():
                values = _metric_values(value)
                if not values:
                    continue
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(values):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                current_value = values[0]
                if k in best_stat:
                    if current_value > best_stat[k]:
                        best_stat[k] = current_value
                        if k == primary_key:
                            best_stat['epoch'] = epoch
                else:
                    best_stat[k] = current_value
                    if k == primary_key:
                        best_stat['epoch'] = epoch

                # Checkpoint selection follows the configured COCO primary metric,
                # while auxiliary scalar metrics remain logging-only.
                if k == primary_key and current_value > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = current_value
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

            primary_values = _metric_values(test_stats.get(f'coco_eval_{self.iou_type}'))
            primary_score = float(primary_values[0]) if primary_values else float('-inf')
            if primary_score > best_primary_score + args.early_stop_min_delta:
                best_primary_score = primary_score
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            self._early_stop_state = {
                'best_primary_score': best_primary_score,
                'no_improve_epochs': no_improve_epochs,
            }

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
                    f'on coco_eval_{self.iou_type}[0].'
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
