import tempfile
import unittest
from pathlib import Path

from scripts.ablation.summarize_cmp5L_ec_fullcycle_cross import discover_arms


class CrossArmSummaryTests(unittest.TestCase):
    def test_reports_all_missing_arms_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ECX0').mkdir()
            (root / 'ECX0' / 'COMPLETED.json').write_text('{}')
            ready, pending = discover_arms(root)
            self.assertEqual(ready, ['ECX0'])
            self.assertEqual(pending, ['ECX1', 'ECX2', 'ECX3', 'ECX4', 'ECX5'])


if __name__ == '__main__':
    unittest.main()
