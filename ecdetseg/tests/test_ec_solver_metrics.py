from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.solver import ec_solver  # noqa: E402
from engine.solver.ec_solver import ECSolver  # noqa: E402


class _TrainLoader:
    def __init__(self):
        self.dataset = SimpleNamespace(
            _transforms=SimpleNamespace(stop_epoch=99, mosaic_epoch=0)
        )

    def __len__(self):
        return 1

    def set_epoch(self, epoch):
        self.epoch = epoch


class EcSolverMetricsTest(unittest.TestCase):
    def setUp(self):
        self.solver = object.__new__(ECSolver)
        self.solver.iou_type = "bbox"

    def test_primary_eval_score_is_map50(self):
        test_stats = {"coco_eval_bbox": [0.91, 0.42, 0.73]}

        self.assertEqual(self.solver._primary_eval_key(), "coco_eval_bbox")
        self.assertEqual(
            self.solver._primary_eval_metric_name(),
            "coco_eval_bbox_map50",
        )
        self.assertEqual(self.solver._primary_eval_score(test_stats), 0.42)

    def test_best_eval_state_tracks_both_maps_and_resumes_all_values(self):
        first_stats = {
            "coco_eval_bbox": [0.50, 0.60],
            "yolo_f1_iou50": {"f1": 0.70},
            "yolo_f1_iou95": {"f1": 0.10},
            "yolo_f1_iou50_95": {"f1": 0.40},
        }
        second_stats = {
            "coco_eval_bbox": [0.55, 0.59],
            "yolo_f1_iou50": {"f1": 0.99},
            "yolo_f1_iou95": {"f1": 0.98},
            "yolo_f1_iou50_95": {"f1": 0.97},
        }

        self.assertTrue(self.solver._record_eval_best(first_stats, epoch=3))
        self.assertFalse(self.solver._record_eval_best(second_stats, epoch=4))

        expected = {
            "best_map50": 0.60,
            "best_map50_epoch": 3,
            "best_map50_95": 0.55,
            "best_map50_95_epoch": 4,
            "best_map50_yolo_f1": 0.70,
            "best_map50_yolo_f1_iou95": 0.10,
            "best_map50_yolo_f1_iou50_95": 0.40,
        }
        self.assertEqual(self.solver._best_eval_state, expected)

        self.solver._early_stop_state = {
            "best_primary_score": 0.60,
            "no_improve_epochs": 2,
        }
        self.solver.last_epoch = 4
        checkpoint = self.solver.state_dict()
        resumed = object.__new__(ECSolver)
        resumed.iou_type = "bbox"
        resumed.last_epoch = -1
        resumed.load_state_dict(checkpoint)

        self.assertEqual(resumed._best_eval_state, expected)
        self.assertEqual(resumed._early_stop_state, self.solver._early_stop_state)

    def test_early_stopping_counts_map50_not_map50_95(self):
        self.assertEqual(
            self.solver._update_early_stop(
                {"coco_eval_bbox": [0.40, 0.50]},
                min_delta=0.0,
            ),
            0,
        )
        self.assertEqual(
            self.solver._update_early_stop(
                {"coco_eval_bbox": [0.90, 0.40]},
                min_delta=0.0,
            ),
            1,
        )
        self.assertEqual(
            self.solver._update_early_stop(
                {"coco_eval_bbox": [0.95, 0.39]},
                min_delta=0.0,
            ),
            2,
        )
        self.assertEqual(
            self.solver._early_stop_state,
            {
                "best_primary_score": 0.50,
                "no_improve_epochs": 2,
            },
        )

    def test_resume_eval_can_be_skipped_for_memory_safe_resume(self):
        self.solver.last_epoch = 1
        self.solver.cfg = SimpleNamespace(skip_resume_eval=True)

        self.assertFalse(self.solver._should_evaluate_before_resume())

        self.solver.cfg.skip_resume_eval = False
        self.assertTrue(self.solver._should_evaluate_before_resume())

    def test_step_checkpoint_progress_round_trips(self):
        self.solver.last_epoch = 1
        self.solver._train_progress = {"epoch": 2, "step": 5000}
        checkpoint = self.solver.state_dict()

        resumed = object.__new__(ECSolver)
        resumed.iou_type = "bbox"
        resumed.last_epoch = -1
        resumed.load_state_dict(checkpoint)

        self.assertEqual(resumed._train_progress, {"epoch": 2, "step": 5000})
        self.assertEqual(resumed._resume_position(), (2, 5000))

    def test_fit_saves_best_and_stops_on_map50(self):
        with tempfile.TemporaryDirectory() as output_dir:
            solver = object.__new__(ECSolver)
            solver.iou_type = "bbox"
            solver.cfg = SimpleNamespace(
                lrsheduler=None,
                epochs=3,
                checkpoint_freq=100,
                clip_max_norm=0.0,
                print_freq=10,
                early_stop_patience=1,
                early_stop_min_delta=0.0,
                gradient_accumulation_steps=1,
            )
            solver.train = mock.Mock()
            solver.train_dataloader = _TrainLoader()
            solver.val_dataloader = object()
            solver.model = object()
            solver.ema = None
            solver.criterion = object()
            solver.postprocessor = object()
            solver.evaluator = object()
            solver.device = "cpu"
            solver.output_dir = Path(output_dir)
            solver.last_epoch = -1
            solver.writer = None
            solver.optimizer = object()
            solver.lr_scheduler = object()
            solver.lr_warmup_scheduler = SimpleNamespace(finished=lambda: False)
            solver.scaler = None

            eval_results = [
                ({"coco_eval_bbox": [0.50, 0.60]}, None),
                ({"coco_eval_bbox": [0.70, 0.50]}, None),
                ({"coco_eval_bbox": [0.80, 0.40]}, None),
            ]
            best_saves = []

            def record_save(state, path):
                if Path(path).name == "best.pth":
                    best_saves.append((state["last_epoch"], state["best_eval_state"]))

            with (
                mock.patch.object(ec_solver, "stats", return_value=(0, "model")),
                mock.patch.object(ec_solver, "train_one_epoch", return_value={}),
                mock.patch.object(ec_solver, "evaluate", side_effect=eval_results) as evaluate,
                mock.patch.object(
                    ec_solver.dist_utils,
                    "is_dist_available_and_initialized",
                    return_value=False,
                ),
                mock.patch.object(ec_solver.dist_utils, "is_main_process", return_value=False),
                mock.patch.object(
                    ec_solver.dist_utils,
                    "save_on_master",
                    side_effect=record_save,
                ),
                mock.patch.object(ec_solver.torch.cuda, "is_available", return_value=False),
            ):
                solver.fit()

            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(len(best_saves), 1)
            self.assertEqual(best_saves[0][0], 0)
            self.assertEqual(best_saves[0][1]["best_map50"], 0.60)
            self.assertEqual(solver._best_eval_state["best_map50_epoch"], 0)
            self.assertEqual(solver._best_eval_state["best_map50_95_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
