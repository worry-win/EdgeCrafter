import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "infer_detection_frame_video.py"


class RenderDetectionsTest(unittest.TestCase):
    def test_draws_box_and_places_label_below_box_when_space_allows(self):
        spec = importlib.util.spec_from_file_location("video_infer", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        rendered = module.draw_detections(
            image,
            [{"label": 0, "score": 0.91, "box": [10, 10, 50, 40]}],
        )
        self.assertTrue(np.any(rendered[10:41, 10:51] != 0))
        self.assertTrue(np.any(rendered[42:70, 10:80] != 0))


if __name__ == "__main__":
    unittest.main()
