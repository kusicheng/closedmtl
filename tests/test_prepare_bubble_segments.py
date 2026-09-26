import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from training_scripts.prepare_bubble_segments import add_row, book_splits, cropped_mask, decode_runs, polygon_from_mask
from training_scripts.prepare_bubble_evaluation import prepare as prepare_evaluation


class PreparationTests(unittest.TestCase):
    def test_cumulative_images_keep_distinct_annotation_rows(self):
        images={}
        ids=set()
        first={'id':0, 'file_name':'A/0.jpg'}
        second={'id':1, 'file_name':'B/0.jpg'}
        add_row({'images':[first], 'annotations':[{'id':5, 'image_id':0}]}, images, ids)
        add_row({'images':[first, second], 'annotations':[{'id':6, 'image_id':1}]}, images, ids)
        self.assertEqual(len(images), 2)
        self.assertEqual(ids, {5, 6})
        with self.assertRaises(ValueError):
            add_row({'images':[], 'annotations':[{'id':5, 'image_id':0}]}, images, ids)
        with self.assertRaises(ValueError):
            add_row({'images':[{'id':0, 'file_name':'wrong'}], 'annotations':[]}, images, ids)

    def test_rle_decodes_tight_crop_without_page_allocation(self):
        self.assertEqual(decode_runs('5220003'), [5, 2, 2, 2, 2, 2, 5])
        ann={'segmentation':{'size':[4, 5], 'counts':'5220003'}, 'bbox':[1, 1, 3, 2], 'area':6}
        np.testing.assert_array_equal(cropped_mask(ann, {'width':5, 'height':4}), np.ones((2, 3)))
        ann['area']=7
        with self.assertRaises(ValueError):
            cropped_mask(ann, {'width':5, 'height':4})
        with self.assertRaises(ValueError):
            decode_runs('p')

    def test_rle_spanning_columns_is_exact(self):
        ann={'segmentation':{'size':[4, 5], 'counts':'0d0'}, 'bbox':[0, 0, 5, 4], 'area':20}
        np.testing.assert_array_equal(cropped_mask(ann, {'width':5, 'height':4}), np.ones((4, 5)))

    def test_hole_loss_rejects_inaccurate_polygon(self):
        mask=np.ones((40, 40), dtype=np.uint8)
        mask[10:30, 10:30]=0
        line, stats=polygon_from_mask(mask, [10, 20, 40, 40], {'width':100, 'height':100})
        self.assertIsNone(line)
        self.assertAlmostEqual(stats['iou'], .75)
        self.assertEqual(stats['holes'], 1)

    def test_disconnected_contours_are_measured(self):
        mask=np.zeros((20, 50), dtype=np.uint8)
        mask[:, :20]=1
        mask[:, 30:]=1
        line, stats=polygon_from_mask(mask, [0, 0, 50, 20], {'width':50, 'height':20})
        self.assertIsNotNone(line)
        self.assertEqual(stats['components'], 2)
        self.assertGreaterEqual(stats['iou'], .98)
        self.assertGreater(stats['false_fill_pixels'], 0)

    def test_book_leakage_fails(self):
        source={'splits':{'train':{'books':['A']}, 'validation':{'books':['A']}, 'test':{'books':['C']}}}
        with self.assertRaises(ValueError):
            book_splits(source)

    def test_full_evaluation_retains_polygon_rejected_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            output=root/'output'
            output.mkdir()
            source=root/'training_data/manga_segmentation'
            source.mkdir(parents=True)
            splits=root/'training_data/manga109_ocr'
            splits.mkdir()
            provenance={'splits':{'train':{'books':['A']}, 'validation':{'books':['B']}, 'test':{'books':['C']}}}
            (splits/'provenance.json').write_text(json.dumps(provenance))
            (output/'provenance.json').write_text(json.dumps({'exclusions':[{'image_id':1, 'reason':'polygon_quality'}]}))
            images=[]
            annotations=[]
            for ident, book in enumerate(('B', 'B', 'C')):
                name=f'{book}/{ident}.jpg'
                image=root/'models/huggingface/distill-Manga109-s/Manga109s_released_2026_05_21/images'/name
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (5, 4)).save(image)
                images.append({'id':ident, 'file_name':name, 'width':5, 'height':4})
                annotations.append({'id':ident, 'image_id':ident, 'category_id':5, 'bbox':[1, 1, 3, 2],
                                    'area':6, 'iscrowd':0, 'segmentation':{'size':[4, 5], 'counts':'5220003'}})
            pq.write_table(pa.Table.from_pylist([{'images':images, 'annotations':annotations}]), source/'0000.parquet')
            result=prepare_evaluation(root, output)
            self.assertEqual(result['counts']['validation']['images'], 2)
            self.assertEqual(result['counts']['validation']['polygon_rejected_pages_retained'], 1)
            rows=[json.loads(line) for line in (output/'full_validation_masks.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['annotations'][0]['segmentation']['counts'], '5220003')
            with self.assertRaises(FileExistsError):
                prepare_evaluation(root, output)


if __name__=='__main__':
    unittest.main()
