import unittest

from ecdetseg.engine.edgecrafter.decoder import ECTransformer


class LayerwiseDeformablePointsTest(unittest.TestCase):
    def test_each_decoder_layer_keeps_its_configured_points_per_level(self):
        model = ECTransformer(
            num_classes=2,
            hidden_dim=256,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[[3, 6, 3], [6, 3, 3], [6, 3, 3], [3, 6, 3]],
            num_layers=4,
        )
        observed = [layer.cross_attn.num_points_list for layer in model.decoder.layers]
        self.assertEqual(observed, [[3, 6, 3], [6, 3, 3], [6, 3, 3], [3, 6, 3]])


if __name__ == "__main__":
    unittest.main()
