import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ablation" / "create_aligned_ecl_breast_inits.py"


def load_module():
    spec = importlib.util.spec_from_file_location("aligned_ecl_breast_inits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AlignedECLBreastInitTest(unittest.TestCase):
    def test_target_keeps_its_backbone_and_receives_the_complete_canonical_detector_side(self):
        transfer = load_module()
        target = {
            "backbone.backbone.weight": torch.full((2, 2), 1.0),
            "backbone.projector.weight": torch.full((2, 2), 2.0),
            "encoder.fpn.weight": torch.full((2, 2), 3.0),
            "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 4.0),
            "decoder.enc_score_head.weight": torch.full((4, 2), 5.0),
            "decoder.enc_bbox_head.layers.2.weight": torch.full((4, 2), 6.0),
        }
        canonical = {
            "backbone.backbone.weight": torch.full((2, 2), 11.0),
            "backbone.projector.weight": torch.full((2, 2), 12.0),
            "encoder.fpn.weight": torch.full((2, 2), 13.0),
            "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 14.0),
            "decoder.enc_score_head.weight": torch.full((4, 2), 15.0),
            "decoder.enc_bbox_head.layers.2.weight": torch.full((4, 2), 16.0),
        }

        merged, audit = transfer.assemble_target_state(target, canonical)

        self.assertTrue(torch.equal(merged["backbone.backbone.weight"], target["backbone.backbone.weight"]))
        self.assertTrue(torch.equal(merged["backbone.projector.weight"], target["backbone.projector.weight"]))
        for key in (
            "encoder.fpn.weight",
            "decoder.decoder.layers.0.linear1.weight",
            "decoder.enc_score_head.weight",
            "decoder.enc_bbox_head.layers.2.weight",
        ):
            self.assertTrue(torch.equal(merged[key], canonical[key]))
        self.assertEqual(audit["aligned_by_family"], {"encoder": 1, "decoder": 3})
        self.assertEqual(audit["missing"], [])
        self.assertEqual(audit["shape_mismatch"], [])

    def test_canonical_decoder_uses_source_body_but_one_shared_four_class_output_state(self):
        transfer = load_module()
        target = {
            "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 1.0),
            "decoder.dec_bbox_head.0.layers.0.weight": torch.full((2, 2), 2.0),
            "decoder.dec_bbox_head.0.layers.2.weight": torch.full((4, 2), 3.0),
            "decoder.enc_score_head.weight": torch.full((4, 2), 4.0),
            "decoder.denoising_class_embed.weight": torch.full((5, 2), 5.0),
        }
        source = {
            "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 11.0),
            "decoder.dec_bbox_head.0.layers.0.weight": torch.full((2, 2), 12.0),
            "decoder.dec_bbox_head.0.layers.2.weight": torch.full((4, 2), 13.0),
            "decoder.enc_score_head.weight": torch.full((80, 2), 14.0),
            "decoder.denoising_class_embed.weight": torch.full((81, 2), 15.0),
        }

        canonical, audit = transfer.build_canonical_state(target, source)

        for key in (
            "decoder.decoder.layers.0.linear1.weight",
            "decoder.dec_bbox_head.0.layers.0.weight",
        ):
            self.assertTrue(torch.equal(canonical[key], source[key]))
        for key in (
            "decoder.dec_bbox_head.0.layers.2.weight",
            "decoder.enc_score_head.weight",
            "decoder.denoising_class_embed.weight",
        ):
            self.assertTrue(torch.equal(canonical[key], target[key]))
        self.assertEqual(audit["missing"], [])
        self.assertEqual(audit["shape_mismatch"], [])


if __name__ == "__main__":
    unittest.main()
