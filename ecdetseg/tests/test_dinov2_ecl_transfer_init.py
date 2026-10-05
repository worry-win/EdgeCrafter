import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ablation" / "create_dinov2_ecl_transfer_init.py"


def _load_transfer_module():
    spec = importlib.util.spec_from_file_location("dinov2_ecl_transfer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DinoV2ECLTransferInitTest(unittest.TestCase):
    def test_merge_preserves_dino_and_random_class_heads_while_loading_ecl_detector(self):
        transfer = _load_transfer_module()
        target = {
        "backbone.backbone.blocks.0.weight": torch.full((2, 2), 1.0),
        "backbone.projector.0.conv.weight": torch.full((2, 2), 2.0),
        "encoder.lateral_convs.0.conv.weight": torch.full((2, 2), 3.0),
        "encoder.fpn_blocks.0.weight": torch.full((2, 2), 4.0),
        "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 5.0),
        "decoder.dec_bbox_head.0.layers.0.weight": torch.full((2, 2), 6.0),
        "decoder.dec_bbox_head.0.layers.2.weight": torch.full((2, 2), 6.5),
        "decoder.enc_bbox_head.layers.2.weight": torch.full((2, 2), 6.6),
        "decoder.pre_bbox_head.layers.2.weight": torch.full((2, 2), 6.7),
        "decoder.enc_score_head.weight": torch.full((7, 2), 7.0),
        "decoder.dec_score_head.0.weight": torch.full((7, 2), 8.0),
        "decoder.denoising_class_embed.weight": torch.full((8, 2), 9.0),
        }
        source = {
        "backbone.backbone.blocks.0.weight": torch.full((2, 2), 11.0),
        "backbone.projector.0.conv.weight": torch.full((2, 2), 12.0),
        "encoder.lateral_convs.0.conv.weight": torch.full((2, 2), 13.0),
        "encoder.fpn_blocks.0.weight": torch.full((3, 2), 14.0),
        "decoder.decoder.layers.0.linear1.weight": torch.full((2, 2), 15.0),
        "decoder.dec_bbox_head.0.layers.0.weight": torch.full((2, 2), 16.0),
        "decoder.dec_bbox_head.0.layers.2.weight": torch.full((2, 2), 16.5),
        "decoder.enc_bbox_head.layers.2.weight": torch.full((2, 2), 16.6),
        "decoder.pre_bbox_head.layers.2.weight": torch.full((2, 2), 16.7),
        "decoder.enc_score_head.weight": torch.full((80, 2), 17.0),
        "decoder.dec_score_head.0.weight": torch.full((80, 2), 18.0),
        "decoder.denoising_class_embed.weight": torch.full((81, 2), 19.0),
        }

        merged, audit = transfer.merge_transfer_state(target, source)

        self.assertTrue(torch.equal(merged["backbone.backbone.blocks.0.weight"], target["backbone.backbone.blocks.0.weight"]))
        self.assertTrue(torch.equal(merged["backbone.projector.0.conv.weight"], target["backbone.projector.0.conv.weight"]))
        self.assertTrue(torch.equal(merged["encoder.lateral_convs.0.conv.weight"], target["encoder.lateral_convs.0.conv.weight"]))
        self.assertTrue(torch.equal(merged["decoder.decoder.layers.0.linear1.weight"], source["decoder.decoder.layers.0.linear1.weight"]))
        self.assertTrue(torch.equal(merged["decoder.dec_bbox_head.0.layers.0.weight"], source["decoder.dec_bbox_head.0.layers.0.weight"]))
        self.assertTrue(torch.equal(merged["decoder.dec_bbox_head.0.layers.2.weight"], target["decoder.dec_bbox_head.0.layers.2.weight"]))
        self.assertTrue(torch.equal(merged["decoder.enc_bbox_head.layers.2.weight"], target["decoder.enc_bbox_head.layers.2.weight"]))
        self.assertTrue(torch.equal(merged["decoder.pre_bbox_head.layers.2.weight"], target["decoder.pre_bbox_head.layers.2.weight"]))
        self.assertTrue(torch.equal(merged["decoder.enc_score_head.weight"], target["decoder.enc_score_head.weight"]))
        self.assertTrue(torch.equal(merged["decoder.dec_score_head.0.weight"], target["decoder.dec_score_head.0.weight"]))
        self.assertTrue(torch.equal(merged["decoder.denoising_class_embed.weight"], target["decoder.denoising_class_embed.weight"]))
        self.assertNotIn("encoder.fpn_blocks.0.weight", audit["transferred"])
        self.assertIn("decoder.dec_bbox_head.0.layers.2.weight", audit["preserved_output_initializers"])
        self.assertIn("decoder.enc_bbox_head.layers.2.weight", audit["preserved_output_initializers"])
        self.assertIn("decoder.pre_bbox_head.layers.2.weight", audit["preserved_output_initializers"])
        self.assertEqual(audit["transferred_by_family"], {"encoder": 0, "decoder": 2})


if __name__ == "__main__":
    unittest.main()
