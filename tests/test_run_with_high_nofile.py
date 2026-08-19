import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path("/cobot/Code/wanrui/EdgeCrafter")
WRAPPER = PROJECT_ROOT / "scripts/ablation/run_with_high_nofile.py"


class RunWithHighNofileTest(unittest.TestCase):
    def test_target_can_import_a_module_from_its_own_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_dir = Path(directory)
            (fixture_dir / "sibling_module.py").write_text(
                'VALUE = "sibling-import-ok"\n', encoding="utf-8"
            )
            target = fixture_dir / "target.py"
            target.write_text(
                "from sibling_module import VALUE\nprint(VALUE)\n", encoding="utf-8"
            )

            result = subprocess.run(
                [sys.executable, str(WRAPPER), str(target)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("sibling-import-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
