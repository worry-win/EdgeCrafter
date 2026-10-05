import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ablation.analyze_cmp5L_privileged_ema_oracle import analyze_payload


class PrivilegedEmaAnalysisTest(unittest.TestCase):
    def test_paired_binding_and_shortcut_deltas_are_reported(self):
        def metrics(equal, gt, wrong, rank, iou):
            return {
                "q_iou_equals_q_cls": equal,
                "q_iou_equals_q_hungarian": equal,
                "q_cls_equals_q_hungarian": equal,
                "gt_logit_q_iou": gt,
                "wrong_logit_q_iou": wrong,
                "class_margin_q_iou": gt - wrong,
                "has_iou_05_candidate": iou >= 0.5,
                "joint_success_iou05_score05": iou >= 0.5 and gt >= 0.0,
                "rank_q_iou_final_score": rank,
                "rank_q_iou_gt_class": rank,
                "max_iou": iou,
                "iou_q_cls": iou,
                "iou_q_hungarian": iou,
                "top1_best_iou": iou - 0.2,
                "top5_best_iou": iou,
                "top1_top5_iou_gap": 0.2,
                "normal_q_iou_current_iou": iou,
                "normal_q_iou_gt_logit": gt,
                "normal_q_iou_wrong_logit": wrong,
                "normal_q_iou_class_margin": gt - wrong,
                "normal_q_iou_final_rank": rank,
                "normal_q_iou_gt_class_rank": rank,
                "auxiliary_q_iou": {
                    "sampling_inside": [0.5],
                    "attention_foreground_mass": [0.4],
                    "attention_context_mass": [0.2],
                    "query_update_norm": [2.0],
                    "query_norm": [3.0],
                    "ffn_activation_norm": [4.0],
                    "ffn_activation_entropy": [0.8],
                    "ffn_top10pct_energy": [0.6],
                },
            }

        payload = {
            "meta": {"conditions": [{"name": "normal"}, {"name": "mild"}]},
            "lesions": [
                {"group": "rank_fixable", "conditions": {
                    "normal": metrics(False, -1.0, 0.0, 8, 0.8),
                    "mild": metrics(True, 0.5, 0.1, 2, 0.79),
                }},
                {"group": "common_failure", "conditions": {
                    "normal": metrics(False, -2.0, -0.5, 10, 0.7),
                    "mild": metrics(True, 0.0, -0.4, 3, 0.69),
                }},
            ],
            "images": [
                {"image_id": 1, "condition": "normal", "is_negative": False,
                 "all_logit_mean": -2.0, "all_query_score_mean": 0.1,
                 "background_query_score_mean": 0.05, "positive_query_score_mean": 0.6,
                 "background_fp_count_05": 0, "background_fp_count_01": 0,
                 "top1_query_score": 0.7},
                {"image_id": 1, "condition": "mild", "is_negative": False,
                 "all_logit_mean": -1.9, "all_query_score_mean": 0.11,
                 "background_query_score_mean": 0.05, "positive_query_score_mean": 0.7,
                 "background_fp_count_05": 0, "background_fp_count_01": 0,
                 "top1_query_score": 0.8},
            ],
        }
        result = analyze_payload(payload, seed=3, bootstrap=100)
        mild = result["conditions"]["mild"]["overall"]
        self.assertEqual(mild["q_iou_equals_q_cls"]["mean"], 1.0)
        self.assertEqual(mild["delta_vs_normal"]["q_iou_equals_q_cls"]["mean"], 1.0)
        self.assertEqual(mild["delta_vs_normal"]["rank_q_iou_final_score"]["mean"], -6.5)
        self.assertEqual(mild["delta_vs_normal"]["gt_logit_q_iou"]["mean"], 1.75)
        self.assertAlmostEqual(mild["delta_vs_normal"]["wrong_logit_q_iou"]["mean"], 0.1)
        self.assertAlmostEqual(
            result["shortcut"]["mild"]["positive_images"]["all_logit_mean"]["mean"],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
