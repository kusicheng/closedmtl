"""Write exact-mask evaluation pages independently of training-polygon quality."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

try:
    from training_scripts.prepare_bubble_segments import add_row, book_splits, digest, write_json
except ModuleNotFoundError:
    from prepare_bubble_segments import add_row, book_splits, digest, write_json


def prepare(root, output):
    root=Path(root).resolve()
    output=Path(output).resolve()
    split_source=root/'training_data/manga109_ocr/provenance.json'
    mapping=book_splits(json.loads(split_source.read_text(encoding='utf-8')))
    provenance=json.loads((output/'provenance.json').read_text(encoding='utf-8'))
    polygon_rejected={row['image_id'] for row in provenance['exclusions'] if row['reason']=='polygon_quality'}
    manga=root/'models/huggingface/distill-Manga109-s/Manga109s_released_2026_05_21/images'
    paths={split:output/f'full_{split}_masks.jsonl' for split in ('validation', 'test')}
    if any(path.exists() for path in paths.values()) or (output/'evaluation_provenance.json').exists():
        raise FileExistsError('Evaluation outputs already exist')
    streams={split:path.open('x', encoding='utf-8') for split, path in paths.items()}
    images={}
    ids=set()
    seen_pages=set()
    counts=defaultdict(Counter)
    exclusions=[]
    hashes={str(split_source):digest(split_source), str(Path(__file__).resolve()):digest(__file__),
            str(output/'provenance.json'):digest(output/'provenance.json')}
    try:
        for path in sorted((root/'training_data/manga_segmentation').glob('*.parquet')):
            hashes[str(path)]=digest(path)
            for row_index, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=1)):
                row=batch.to_pylist()[0]
                add_row(row, images, ids)
                grouped=defaultdict(list)
                for ann in row['annotations']:
                    if ann['category_id']==5:
                        grouped[ann['image_id']].append(ann)
                for ident, annotations in grouped.items():
                    image=images[ident]
                    book=image['file_name'].split('/')[0]
                    split=mapping.get(book)
                    if split not in streams:
                        continue
                    if ident in seen_pages:
                        raise ValueError('Duplicate evaluation page')
                    seen_pages.add(ident)
                    source=manga/image['file_name']
                    reason=None
                    if not source.is_file():
                        reason='missing_image'
                    else:
                        with Image.open(source) as opened:
                            if opened.size!=(image['width'], image['height']):
                                reason='image_dimension_mismatch'
                    if reason:
                        exclusions.append({'image_id':ident, 'source':str(source), 'reason':reason})
                        continue
                    record={'image_id':ident, 'book':book, 'source_image':str(source), 'source_sha256':digest(source),
                            'image_path':str(source), 'width':image['width'], 'height':image['height'],
                            'source_ref':{'parquet':str(path), 'row':row_index}, 'annotations':annotations,
                            'training_polygon_rejected':ident in polygon_rejected}
                    streams[split].write(json.dumps(record, ensure_ascii=False)+'\n')
                    counts[split]['images']+=1
                    counts[split]['balloons']+=len(annotations)
                    counts[split]['polygon_rejected_pages_retained']+=int(ident in polygon_rejected)
    finally:
        for stream in streams.values():
            stream.close()
    report={'status':'complete', 'counts':dict(counts), 'exclusions':exclusions, 'source_hashes':hashes,
            'manifest_sha256':{path.name:digest(path) for path in paths.values()},
            'policy':'All available dimension-matching positive validation/test pages, regardless of polygon representation quality. Exact original COCO RLE retained. No source-negative claim. Book-disjoint local fine-tuning split; upstream model exposure remains unknown.'}
    write_json(output/'evaluation_provenance.json', report)
    print(json.dumps(report['counts']), flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='training_data/bubble_joint_20260922')
    args=parser.parse_args()
    prepare(args.root, args.output)
