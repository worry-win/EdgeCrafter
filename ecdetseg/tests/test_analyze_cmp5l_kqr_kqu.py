"""Read-only arm and contrast contracts for KQ-R/KQ-U paired validation."""

import unittest
from argparse import Namespace
from pathlib import Path

from scripts.ablation.analyze_cmp5L_kqr_kqu import GROUPS, CONTRASTS, arm_roots


class KQRKQUAnalysisContractTest(unittest.TestCase):
    def test_comparison_directions_and_roots_are_locked(self):
        roots = arm_roots(Namespace(original_root="/old", extension_root="/ext", new_root="/new"))
        self.assertEqual(GROUPS, ("KQ0", "KQ2", "KQO", "KQR", "KQU"))
        self.assertEqual(roots, {
            "KQ0": Path("/old/KQ0"), "KQ2": Path("/old/KQ2"),
            "KQO": Path("/ext/KQO"), "KQR": Path("/new/KQR"), "KQU": Path("/new/KQU"),
        })
        for arm in ("KQR", "KQU"):
            for control in ("KQO", "KQ2", "KQ0"):
                self.assertEqual(CONTRASTS[f"{arm}_minus_{control}"], (control, arm))
        self.assertEqual(len(CONTRASTS), 6)


if __name__ == "__main__":
    unittest.main()
