from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.solver.ec_engine import (  # noqa: E402
    evaluate,
    summarize_pr_curve_f1,
    summarize_yolo_pr_curve_metrics,
)


class EcEngineF1Test(unittest.TestCase):
    def test_evaluate_keeps_legacy_fields_and_adds_yolo_fields(self):
        recalls = np.array([0.0, 0.5, 1.0])
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.ones((10, 3, 1, 1, 1), dtype=np.float64) * 0.8
        scores = np.ones_like(precision) * 0.6
        bbox_eval = SimpleNamespace(
            eval={"precision": precision, "scores": scores},
            params=SimpleNamespace(
                recThrs=recalls,
                iouThrs=iou_thresholds,
                catIds=[0],
            ),
            stats=np.arange(12, dtype=np.float64),
        )
        coco_gt = SimpleNamespace(loadCats=lambda _: [{"id": 0, "name": "zero"}])

        class FakeCocoEvaluator:
            labels = None
            iou_types = ["bbox"]

            def __init__(self):
                self.coco_eval = {"bbox": bbox_eval}
                self.coco_gt = coco_gt

            def cleanup(self):
                pass

            def update(self, results):
                pass

            def synchronize_between_processes(self):
                pass

            def accumulate(self):
                pass

            def summarize(self):
                pass

        class FakeModel:
            def eval(self):
                pass

            def __call__(self, samples):
                return {}

        class FakeCriterion:
            def eval(self):
                pass

        data_loader = [(
            torch.zeros((1, 3, 2, 2)),
            [{
                "orig_size": torch.tensor([2, 2]),
                "image_id": torch.tensor(1),
            }],
        )]
        expected_legacy = summarize_pr_curve_f1(bbox_eval)

        stats, _ = evaluate(
            FakeModel(),
            FakeCriterion(),
            lambda outputs, sizes: [{}],
            data_loader,
            FakeCocoEvaluator(),
            torch.device("cpu"),
        )

        self.assertEqual(
            stats["coco_eval_bbox_f1"],
            expected_legacy["f1_iou50"],
        )
        self.assertEqual(
            stats["coco_eval_bbox_f1_recall"],
            expected_legacy["recall_iou50"],
        )
        self.assertIn("yolo_per_class", stats)
        self.assertIn("yolo_overall", stats)

    def test_yolo_per_class_uses_only_evaluator_category_ids(self):
        recalls = np.array([0.0, 1.0])
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.ones((10, 2, 7, 1, 1), dtype=np.float64)
        scores = np.ones_like(precision)
        evaluated_cat_ids = list(range(7))
        all_categories = {
            cat_id: {"id": cat_id, "name": f"class-{cat_id}"}
            for cat_id in range(8)
        }
        coco_eval = SimpleNamespace(
            eval={"precision": precision, "scores": scores},
            params=SimpleNamespace(
                recThrs=recalls,
                iouThrs=iou_thresholds,
                catIds=evaluated_cat_ids,
            ),
        )
        coco_gt = SimpleNamespace(
            loadCats=lambda cat_ids: [all_categories[cat_id] for cat_id in cat_ids]
        )

        summary = summarize_yolo_pr_curve_metrics(coco_eval, coco_gt)

        self.assertEqual(coco_eval.params.catIds, evaluated_cat_ids)
        self.assertEqual(set(summary["yolo_per_class"]), set(evaluated_cat_ids))
        self.assertEqual(len(summary["yolo_per_class"]), 7)
        self.assertNotIn(7, summary["yolo_per_class"])

    def test_yolo_summary_reports_iou50_iou95_and_sweep_mean(self):
        recalls = np.array([0.0, 1.0])
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.empty((10, 2, 1, 1, 1), dtype=np.float64)
        scores = np.ones_like(precision)
        per_iou_precision = np.linspace(0.1, 1.0, 10)
        for index, value in enumerate(per_iou_precision):
            precision[index, :, 0, 0, 0] = value

        coco_eval = SimpleNamespace(
            eval={"precision": precision, "scores": scores},
            params=SimpleNamespace(
                recThrs=recalls,
                iouThrs=iou_thresholds,
                catIds=[0],
            ),
        )
        coco_gt = SimpleNamespace(loadCats=lambda _: [{"id": 0, "name": "zero"}])

        summary = summarize_yolo_pr_curve_metrics(coco_eval, coco_gt)
        expected_f1 = 2 * per_iou_precision / (per_iou_precision + 1.0)

        self.assertAlmostEqual(summary["yolo_f1_iou50"]["f1"], expected_f1[0])
        self.assertAlmostEqual(summary["yolo_f1_iou95"]["f1"], expected_f1[-1])
        self.assertAlmostEqual(
            summary["yolo_f1_iou50_95"]["f1"],
            expected_f1.mean(),
        )

    def test_yolo_overall_uses_all_class_mean_pr_then_harmonic_f1(self):
        recalls = np.array([0.0, 0.5, 1.0])
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.zeros((10, 3, 3, 1, 1), dtype=np.float64)
        scores = np.zeros_like(precision)

        precision[:, :, 0, 0, 0] = [0.9, 0.8, 0.1]
        precision[:, :, 1, 0, 0] = [0.9, 0.9, 0.8]
        coco_eval = SimpleNamespace(
            eval={"precision": precision, "scores": scores},
            params=SimpleNamespace(
                recThrs=recalls,
                iouThrs=iou_thresholds,
                catIds=[0, 1, 2],
            ),
        )
        coco_gt = SimpleNamespace(
            loadCats=lambda cat_ids: [
                {"id": cat_id, "name": str(cat_id)} for cat_id in cat_ids
            ]
        )

        summary = summarize_yolo_pr_curve_metrics(coco_eval, coco_gt)
        overall = summary["yolo_overall"]
        class_metrics = summary["yolo_per_class"]
        expected_precision = (0.8 + 0.8 + 0.0) / 3
        expected_recall = (0.5 + 1.0 + 0.0) / 3
        expected = (
            2 * expected_precision * expected_recall
            / (expected_precision + expected_recall)
        )
        mean_class_f1 = np.mean([
            class_metrics[0]["f1"],
            class_metrics[1]["f1"],
        ])

        self.assertEqual(class_metrics[2]["f1"], 0.0)
        self.assertAlmostEqual(overall["precision"], expected_precision)
        self.assertAlmostEqual(overall["recall"], expected_recall)
        self.assertAlmostEqual(overall["f1"], expected)
        self.assertNotAlmostEqual(overall["f1"], mean_class_f1)

    def test_yolo_summary_selects_each_class_best_recall_and_confidence(self):
        recalls = np.array([0.0, 0.5, 1.0])
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.full((10, 3, 2, 1, 1), -1.0, dtype=np.float64)
        scores = np.full_like(precision, -1.0)

        precision[:, :, 0, 0, 0] = [0.9, 0.8, 0.1]
        scores[:, :, 0, 0, 0] = [0.95, 0.60, 0.10]
        precision[:, :, 1, 0, 0] = [0.9, 0.9, 0.8]
        scores[:, :, 1, 0, 0] = [0.90, 0.50, 0.20]

        coco_eval = SimpleNamespace(
            eval={"precision": precision, "scores": scores},
            params=SimpleNamespace(
                recThrs=recalls,
                iouThrs=iou_thresholds,
                catIds=[0, 1],
            ),
        )
        coco_gt = SimpleNamespace(
            loadCats=lambda cat_ids: [
                {"id": cat_id, "name": f"class-{cat_id}"}
                for cat_id in cat_ids
            ]
        )

        summary = summarize_yolo_pr_curve_metrics(coco_eval, coco_gt)

        self.assertEqual(summary["yolo_per_class"][0]["recall"], 0.5)
        self.assertEqual(summary["yolo_per_class"][0]["confidence"], 0.60)
        self.assertEqual(summary["yolo_per_class"][1]["recall"], 1.0)
        self.assertEqual(summary["yolo_per_class"][1]["confidence"], 0.20)

    def test_summary_reports_iou50_iou95_and_sweep_mean(self):
        recalls = np.linspace(0.0, 1.0, 101)
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        precision = np.empty((10, 101, 1, 1, 3), dtype=np.float64)
        per_iou_precision = np.linspace(0.1, 1.0, 10)
        for index, value in enumerate(per_iou_precision):
            precision[index, :, :, :, :] = value

        coco_eval = SimpleNamespace(
            eval={"precision": precision},
            params=SimpleNamespace(recThrs=recalls, iouThrs=iou_thresholds),
        )
        summary = summarize_pr_curve_f1(coco_eval)
        expected_f1 = 2 * per_iou_precision / (per_iou_precision + 1.0)

        self.assertAlmostEqual(summary["f1_iou50"], expected_f1[0])
        self.assertAlmostEqual(summary["f1_iou95"], expected_f1[-1])
        self.assertAlmostEqual(summary["f1_iou50_95_mean"], expected_f1.mean())


if __name__ == "__main__":
    unittest.main()
