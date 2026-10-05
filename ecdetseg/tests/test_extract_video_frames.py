import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "extract_video_frames.py"


class ExtractVideoFramesCliTest(unittest.TestCase):
    def test_extracts_at_requested_rate_and_records_source_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 3.0, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            for value in range(6):
                writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
            writer.release()

            output = root / "frames"
            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(video),
                "--output-dir", str(output), "--fps", "1.5",
            ], check=True)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["source_frame_count"], 6)
            self.assertEqual(manifest["extracted_frame_count"], 3)
            self.assertEqual(manifest["source_fps"], 3.0)
            self.assertEqual(manifest["output_fps"], 1.5)
            self.assertEqual([x["source_frame_index"] for x in manifest["frames"]], [0, 2, 4])
            self.assertEqual(
                [x["file_name"] for x in manifest["frames"]],
                ["frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
