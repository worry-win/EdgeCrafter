import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ecdetseg.engine.data.dataset.sqlite_coco_dataset import SqliteCocoDetection
from scripts.data.build_coco_sqlite import build_coco_sqlite


class SqliteCocoDatasetTest(unittest.TestCase):
    def test_build_and_read_an_image_with_normal_and_ignore_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (10, 8), color="white").save(root / "sample.png")
            annotation_path = root / "train.json"
            annotation_path.write_text(json.dumps({
                "images": [{"id": 101, "file_name": "sample.png", "width": 10, "height": 8}],
                "categories": [
                    {"id": 9, "name": "ignore"},
                    {"id": 1, "name": "lesion"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 101, "category_id": 1, "bbox": [2, 2, 4, 3], "area": 12},
                    {"id": 2, "image_id": 101, "category_id": 9, "bbox": [1, 1, 2, 3], "area": 6},
                ],
            }))
            sqlite_path = root / "train.sqlite"

            summary = build_coco_sqlite(annotation_path, sqlite_path)
            dataset = SqliteCocoDetection(
                img_folder=str(root),
                sqlite_file=str(sqlite_path),
                transforms=None,
            )
            image, target = dataset[0]

            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["annotations"], 2)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(image.size, (10, 8))
            self.assertEqual(target["image_id"].tolist(), [101])
            self.assertEqual(target["labels"].tolist(), [1])
            self.assertEqual(target["boxes"].tolist(), [[2.0, 2.0, 6.0, 5.0]])
            self.assertEqual(target["ignore_boxes"].tolist(), [[1.0, 1.0, 3.0, 4.0]])
            self.assertEqual(dataset.category2name, {9: "ignore", 1: "lesion"})
            self.assertEqual(dataset.category2label, {9: 0, 1: 1})

    def test_builder_excludes_missing_images_and_their_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (10, 8), color="white").save(root / "present.png")
            annotation_path = root / "train.json"
            annotation_path.write_text(json.dumps({
                "images": [
                    {"id": 1, "file_name": "present.png"},
                    {"id": 2, "file_name": "missing.png"},
                ],
                "categories": [{"id": 1, "name": "lesion"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]},
                ],
            }))

            summary = build_coco_sqlite(
                annotation_path,
                root / "train.sqlite",
                image_root=root,
            )

            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["annotations"], 1)
            self.assertEqual(summary["excluded_images"], 1)
            self.assertEqual(summary["excluded_annotations"], 1)


if __name__ == "__main__":
    unittest.main()
