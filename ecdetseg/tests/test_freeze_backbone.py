import unittest
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ecdetseg.engine.core.yaml_config import freeze_backbone_parameters


class FreezeBackboneTest(unittest.TestCase):
    def test_freezes_only_backbone(self):
        model = nn.Module()
        model.backbone = nn.Linear(4, 4)
        model.decoder = nn.Linear(4, 2)
        freeze_backbone_parameters(model)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        self.assertFalse(model.backbone.training)


if __name__ == "__main__":
    unittest.main()
