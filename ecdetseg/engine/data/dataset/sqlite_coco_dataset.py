"""Memory-bounded COCO training dataset backed by SQLite."""

import json
import os
import sqlite3

import numpy as np
import torch
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset
from .coco_dataset import ConvertCocoPolysToMask


Image.MAX_IMAGE_PIXELS = None


@register()
class SqliteCocoDetection(DetDataset):
    """COCO-format detection data that only keeps image ids in RAM.

    The SQLite connection is opened lazily, so each DataLoader process owns a
    small read-only connection instead of inheriting a full COCO Python index.
    """

    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(
        self,
        img_folder,
        sqlite_file,
        transforms,
        ann_file=None,
        return_masks=False,
        remap_mscoco_category=False,
    ):
        self.img_folder = img_folder
        self.sqlite_file = os.path.abspath(sqlite_file)
        self._transforms = transforms
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self._connection = None

        connection = self._connect()
        try:
            self.ids = np.fromiter(
                (int(row[0]) for row in connection.execute('SELECT id FROM images ORDER BY id')),
                dtype=np.int64,
            )
            self._categories = [
                {'id': int(category_id), 'name': name}
                for category_id, name in connection.execute(
                    'SELECT id, name FROM categories ORDER BY position'
                )
            ]
        finally:
            connection.close()
            self._connection = None
        self.auto_ignore_category_ids = {
            category['id']
            for category in self._categories
            if str(category['name']).strip().lower() == 'ignore'
        }
        self.prepare = ConvertCocoPolysToMask(
            return_masks,
            ignore_category_ids=self.auto_ignore_category_ids,
        )

    def _connect(self):
        if self._connection is None:
            uri = f'file:{self.sqlite_file}?mode=ro'
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._connection.execute('PRAGMA query_only = ON')
            self._connection.execute('PRAGMA cache_size = -2048')
        return self._connection

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_connection'] = None
        return state

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image, target = self.load_item(idx)
        if self._transforms is not None:
            self._transforms.set_epoch(self.epoch)
            image, target = self._transforms(image, target)
        return image, target

    def load_item(self, idx):
        image_id = int(self.ids[idx])
        connection = self._connect()
        image_row = connection.execute(
            'SELECT file_name FROM images WHERE id = ?', (image_id,)
        ).fetchone()
        if image_row is None:
            raise IndexError(f'Unknown image id: {image_id}')
        image_path = os.path.join(self.img_folder, image_row[0])
        image = Image.open(image_path).convert('RGB')

        rows = connection.execute(
            '''SELECT id, category_id, x, y, width, height, area, iscrowd,
                      segmentation, keypoints
               FROM annotations WHERE image_id = ? ORDER BY id''',
            (image_id,),
        )
        annotations = []
        for row in rows:
            annotation = {
                'id': int(row[0]),
                'image_id': image_id,
                'category_id': int(row[1]),
                'bbox': [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                'area': float(row[6]),
                'iscrowd': int(row[7]),
            }
            if row[8] is not None:
                annotation['segmentation'] = json.loads(row[8])
            if row[9] is not None:
                annotation['keypoints'] = json.loads(row[9])
            annotations.append(annotation)

        target = {'image_id': image_id, 'annotations': annotations}
        image, target = self.prepare(image, target)
        target['idx'] = torch.tensor([idx])
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'], key='boxes', spatial_size=image.size[::-1]
            )
        if 'ignore_boxes' in target:
            target['ignore_boxes'] = convert_to_tv_tensor(
                target['ignore_boxes'], key='boxes', spatial_size=image.size[::-1]
            )
        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        return image, target

    @property
    def categories(self):
        return self._categories

    @property
    def category2name(self):
        return {category['id']: category['name'] for category in self.categories}

    @property
    def category2label(self):
        return {category['id']: index for index, category in enumerate(self.categories)}

    @property
    def label2category(self):
        return {index: category['id'] for index, category in enumerate(self.categories)}
