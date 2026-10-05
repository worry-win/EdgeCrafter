import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "build_liver_perturbation_coco.py"


class LiverPerturbationCocoCliTest(unittest.TestCase):
    def test_builds_aligned_original_and_spatially_perturbed_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "perturb"
            original = base / "original" / "case_a"
            cropped = base / "random_crop" / "case_a"
            original.mkdir(parents=True)
            cropped.mkdir(parents=True)

            (original / "scan.png").write_bytes(b"image")
            (cropped / "scan_frame000.png").write_bytes(b"image")
            (original / "scan.json").write_text(json.dumps({
                "imagePath": "scan.png", "imageWidth": 100, "imageHeight": 80,
                "shapes": [
                    {"label": "9", "shape_type": "rectangle", "points": [[10, 20], [30, 50]]},
                    {"label": "000", "shape_type": "point", "points": [[5, 5]]},
                ],
            }))
            (cropped / "scan_frame000.json").write_text(json.dumps({
                "imagePath": "scan_frame000.png", "imageWidth": 100, "imageHeight": 80,
                "shapes": [
                    {"label": "9", "shape_type": "rectangle", "points": [[8, 18], [29, 49]]},
                    {"label": "000", "shape_type": "point", "points": [[5, 5]]},
                ],
            }))
            reference = root / "test.json"
            reference.write_text(json.dumps({
                "images": [{"id": 7, "file_name": "test__abc123__case_a_scan.png", "width": 100, "height": 80}],
                "annotations": [{"id": 11, "image_id": 7, "category_id": 1, "bbox": [10, 20, 20, 30], "area": 600, "iscrowd": 0}],
                "categories": [{"id": i, "name": str(i)} for i in range(7)],
            }))
            output = root / "out"

            subprocess.run([
                sys.executable, str(SCRIPT), "--base-dir", str(base),
                "--reference-json", str(reference), "--output-dir", str(output),
                "--strategies", "original", "random_crop",
            ], check=True)

            clean = json.loads((output / "original.json").read_text())
            perturbed = json.loads((output / "random_crop.json").read_text())
            self.assertEqual(clean["images"][0]["file_name"], "case_a/scan.png")
            self.assertEqual(clean["annotations"][0]["category_id"], 1)
            self.assertEqual(clean["annotations"][0]["bbox"], [10.0, 20.0, 20.0, 30.0])
            self.assertEqual(perturbed["images"][0]["file_name"], "case_a/scan_frame000.png")
            self.assertEqual(perturbed["annotations"][0]["bbox"], [8.0, 18.0, 21.0, 31.0])
            self.assertEqual(len(perturbed["annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
