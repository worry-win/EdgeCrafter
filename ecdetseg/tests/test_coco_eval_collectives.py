from pathlib import Path
import sys
import unittest
from unittest import mock


ECDETSEG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECDETSEG_ROOT))

from engine.data.dataset import coco_eval  # noqa: E402


class CocoEvalCollectiveTest(unittest.TestCase):
    def test_picklable_data_uses_the_shared_object_collective(self):
        payload = {"img_ids": [1, 2, 3]}
        gathered = [payload, {"img_ids": [4]}]

        with mock.patch.object(
            coco_eval.dist_utils,
            "all_gather",
            return_value=gathered,
        ) as shared_all_gather:
            result = coco_eval.all_gather(payload)

        self.assertIs(result, gathered)
        shared_all_gather.assert_called_once_with(payload)


if __name__ == "__main__":
    unittest.main()
