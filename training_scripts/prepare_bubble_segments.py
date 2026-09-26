"""Prepare local, book-disjoint balloon masks and current one-class boxes."""
import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode_runs(text):
    """Decode signed delta COCO RLE (cocodataset/cocoapi/common/maskApi.c)."""
    result=[]
    value=0
    shift=0
    for char in text:
        code=ord(char)-48
        if not 0<=code<=63:
            raise ValueError('Invalid RLE character')
        value|=(code&31)<<shift
        shift+=5
        if code&32:
            continue
        if code&16:
            value|=-1<<shift
        if len(result)>2:
            value+=result[-2]
        if value<0:
            raise ValueError('Negative RLE run')
        result.append(value)
        value=0
        shift=0
    if shift or not result:
        raise ValueError('Empty or truncated RLE')
    return result


def cropped_mask(annotation, image):
    """Decode exact foreground directly into its COCO bounding rectangle."""
    segmentation=annotation['segmentation']
    height, width=segmentation['size']
    if [height, width]!=[image['height'], image['width']]:
        raise ValueError('Mask/image dimensions differ')
    x, y, w, h=annotation['bbox']
    if min(w, h)<=0 or min(x, y)<0 or x+w>width or y+h>height:
        raise ValueError('Invalid mask bounding box')
    runs=decode_runs(segmentation['counts'])
    if sum(runs)!=height*width or sum(runs[1::2])!=annotation['area']:
        raise ValueError('RLE area or image coverage differs')
    mask=np.zeros((h, w), dtype=np.uint8)
    offset=0
    for index, run in enumerate(runs):
        if index%2 and run:
            first_x, first_y=divmod(offset, height)
            last_x, last_y=divmod(offset+run-1, height)
            if first_x<x or last_x>=x+w:
                raise ValueError('RLE exceeds bounding box columns')
            if first_x==last_x:
                if first_y<y or last_y>=y+h:
                    raise ValueError('RLE exceeds bounding box rows')
                mask[first_y-y:last_y-y+1, first_x-x]=1
            else:
                if y!=0 or h!=height:
                    raise ValueError('Column-spanning run exceeds bounding box')
                mask[first_y:, first_x-x]=1
                mask[:, first_x-x+1:last_x-x]=1
                mask[:last_y+1, last_x-x]=1
        offset+=run
    ys, xs=np.nonzero(mask)
    if not len(xs) or [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]!=[0, 0, w, h]:
        raise ValueError('RLE does not match tight bounding box')
    return mask


def polygon_from_mask(mask, bbox, image, minimum_iou=0.98):
    """Measure filled/merged YOLO polygon against exact pixels before accepting."""
    from ultralytics.data.converter import merge_multi_segment
    from ultralytics.utils.ops import resample_segments

    contours, hierarchy=cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError('No contour')
    external=[contour.reshape(-1, 2) for contour, parent in zip(contours, hierarchy[0]) if parent[3]<0]
    external=[points for points in external if len(points)>=3]
    if not external:
        raise ValueError('No polygon with three vertices')
    points=external[0] if len(external)==1 else np.concatenate(merge_multi_segment(external))
    # Pixel-center vertices survive decimal serialization and int32 rasterization.
    offset=np.array(bbox[:2], dtype=np.float32)
    scale=np.array([image['width'], image['height']], dtype=np.float32)
    normalized=(points.astype(np.float32)+offset+0.5)/scale
    values=['0']+[f'{float(value):.10f}' for value in normalized.reshape(-1)]
    serialized=np.array([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
    sample_count=max(1000, len(serialized)+1)
    sampled=resample_segments([serialized.copy()], n=sample_count)[0]
    raster=np.zeros_like(mask)
    cv2.fillPoly(raster, [(sampled*scale-offset).astype(np.int32)], 1)
    intersection=int(np.count_nonzero(raster&mask))
    union=int(np.count_nonzero(raster|mask))
    iou=intersection/union
    stats={'iou':iou, 'vertices':len(points), 'components':len(external),
           'holes':int(np.count_nonzero(hierarchy[0, :, 3]>=0)),
           'false_fill_pixels':int(np.count_nonzero(raster&(1-mask))),
           'lost_pixels':int(np.count_nonzero(mask&(1-raster)))}
    if iou<minimum_iou:
        return None, stats
    return ' '.join(values), stats


def add_row(row, images, annotation_ids):
    """Image metadata are cumulative; annotation rows belong to distinct books."""
    for image in row['images']:
        old=images.get(image['id'])
        if old is not None and old!=image:
            raise ValueError(f"Conflicting image ID {image['id']}")
        images[image['id']]=image
    for annotation in row['annotations']:
        if annotation['id'] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID {annotation['id']}")
        if annotation['image_id'] not in images:
            raise ValueError('Annotation references unknown image')
        annotation_ids.add(annotation['id'])


def book_splits(provenance):
    mapping={}
    for split, values in provenance['splits'].items():
        for book in values['books']:
            if book in mapping:
                raise ValueError(f'Book appears in multiple splits: {book}')
            mapping[book]=split
    if set(mapping.values())!={'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation and test book splits')
    return mapping


def link_image(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return 'hardlink'
    except OSError:
        shutil.copy2(source, target)
        return 'copy'


def box_line(bbox, width, height):
    x, y, w, h=bbox
    return f'0 {(x+w/2)/width:.10f} {(y+h/2)/height:.10f} {w/width:.10f} {h/height:.10f}'


def write_labels(output, kind, split, stem, lines):
    path=output/kind/'labels'/split/(stem+'.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def prepare(args):
    output=Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root=Path(args.root).resolve()
    manga=root/'models/huggingface/distill-Manga109-s/Manga109s_released_2026_05_21/images'
    split_path=root/'training_data/manga109_ocr/provenance.json'
    split_source=json.loads(split_path.read_text(encoding='utf-8'))
    mapping=book_splits(split_source)
    images={}
    annotation_ids=set()
    processed=set()
    counts=defaultdict(Counter)
    link_methods=Counter()
    exclusions=[]
    manifests=defaultdict(list)
    rows=[]
    all_iou=[]
    accepted_iou=[]
    topology=Counter()
    source_hashes={str(split_path):digest(split_path), str(Path(__file__).resolve()):digest(__file__)}
    gt={split:(output/f'{split}_masks.jsonl').open('x', encoding='utf-8') for split in ('train', 'validation', 'test')}
    quality=(output/'polygon_quality.jsonl').open('x', encoding='utf-8')
    try:
        for path in sorted((root/'training_data/manga_segmentation').glob('*.parquet')):
            source_hashes[str(path)]=digest(path)
            for row_index, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=1)):
                row=batch.to_pylist()[0]
                add_row(row, images, annotation_ids)
                grouped=defaultdict(list)
                for ann in row['annotations']:
                    if ann['category_id']==5:
                        grouped[ann['image_id']].append(ann)
                rows.append({'shard':str(path), 'row':row_index, 'balloons':sum(map(len, grouped.values()))})
                for image_id, annotations in grouped.items():
                    if image_id in processed:
                        raise ValueError('Balloon page appears in multiple annotation rows')
                    processed.add(image_id)
                    image=images[image_id]
                    name=image['file_name']
                    book=name.split('/')[0]
                    split=mapping.get(book)
                    source=manga/name
                    reason=None
                    if split is None:
                        reason='book_not_in_available_ocr_split'
                    elif book=='PrayerHaNemurenai':
                        reason='excluded_book_dimension_mismatch'
                    elif not source.is_file():
                        reason='missing_image'
                    else:
                        with Image.open(source) as opened:
                            if opened.size!=(image['width'], image['height']):
                                reason='image_dimension_mismatch'
                    if reason:
                        exclusions.append({'image_id':image_id, 'source':name, 'balloons':len(annotations), 'reason':reason})
                        continue
                    lines=[]
                    page_stats=[]
                    failures=[]
                    for ann in annotations:
                        try:
                            mask=cropped_mask(ann, image)
                            line, stats=polygon_from_mask(mask, ann['bbox'], image, args.minimum_iou)
                            if ann['iscrowd']:
                                line=None
                                stats['error']='crowd_annotation'
                        except ValueError as error:
                            line=None
                            stats={'iou':0.0, 'error':str(error)}
                        stats.update({'annotation_id':ann['id'], 'image_id':image_id})
                        page_stats.append(stats)
                        all_iou.append(stats['iou'])
                        if line is None:
                            failures.append(ann['id'])
                        else:
                            lines.append(line)
                    accepted=not failures
                    for stats in page_stats:
                        quality.write(json.dumps({**stats, 'page_accepted':accepted})+'\n')
                    if failures:
                        exclusions.append({'image_id':image_id, 'source':name, 'balloons':len(annotations),
                                           'reason':'polygon_quality', 'failed_annotations':failures,
                                           'minimum_iou':min(value['iou'] for value in page_stats)})
                        continue
                    stem=f'manga_{image_id:05d}'
                    targets={kind:output/kind/'images'/split/(stem+'.jpg') for kind in ('segments', 'boxes')}
                    for kind, target in targets.items():
                        link_methods[link_image(source, target)]+=1
                        manifests[f'{kind}_{split}'].append(target.as_posix())
                    write_labels(output, 'segments', split, stem, lines)
                    write_labels(output, 'boxes', split, stem,
                                 [box_line(ann['bbox'], image['width'], image['height']) for ann in annotations])
                    record={'image_id':image_id, 'book':book, 'source_image':str(source), 'source_sha256':digest(source),
                            'image_path':targets['segments'].as_posix(), 'width':image['width'], 'height':image['height'],
                            'source_ref':{'parquet':str(path), 'row':row_index}, 'annotations':annotations}
                    gt[split].write(json.dumps(record, ensure_ascii=False)+'\n')
                    counts[split]['images']+=1
                    counts[split]['balloons']+=len(annotations)
                    accepted_iou.extend(value['iou'] for value in page_stats)
                    topology['instances_with_holes']+=sum(value['holes']>0 for value in page_stats)
                    topology['instances_with_multiple_components']+=sum(value['components']>1 for value in page_stats)
                print(json.dumps({'row':len(rows), 'counts':dict(counts), 'excluded_pages':len(exclusions)}), flush=True)
    finally:
        for stream in gt.values():
            stream.close()
        quality.close()
    for image_id, image in images.items():
        if image_id not in processed:
            exclusions.append({'image_id':image_id, 'source':image['file_name'], 'balloons':0,
                               'reason':'no_balloon_annotation_not_assumed_negative'})
    current=prepare_current_boxes(root, output, manifests, link_methods, source_hashes, exclusions)
    for name, paths in manifests.items():
        (output/f'{name}.txt').write_text('\n'.join(paths)+'\n', encoding='utf-8')
    for kind in ('segments', 'boxes'):
        write_yaml(output/f'{kind}.yaml', output, f'{kind}_train.txt', f'{kind}_validation.txt', f'{kind}_test.txt')
    write_yaml(output/'current_boxes.yaml', output, 'current_train.txt', 'current_validation.txt', 'current_test.txt')
    manifest_hashes={path.name:digest(path) for path in output.glob('*.txt')}
    mask_hashes={path.name:digest(path) for path in output.glob('*_masks.jsonl')}
    quantiles=lambda values:dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'max'),
                                    np.quantile(values, [0, .01, .05, .5, .95, 1]).tolist())) if values else {}
    report={'status':'complete', 'root':str(root), 'output':str(output), 'source_hashes':source_hashes,
            'source_metadata':{key:row[key] for key in ('info', 'licenses', 'categories')},
            'split_books':{split:sorted(book for book, value in mapping.items() if value==split)
                           for split in ('train', 'validation', 'test')},
            'counts':dict(counts), 'current_boxes':current, 'source_rows':rows, 'exclusions':exclusions,
            'exclusion_counts':dict(Counter(item['reason'] for item in exclusions)), 'image_link_methods':dict(link_methods),
            'polygon_quality':{'minimum_required_iou':args.minimum_iou, 'attempted':quantiles(all_iou),
                               'accepted':quantiles(accepted_iou), 'accepted_topology':dict(topology),
                               'method':'External contours; Ultralytics merge_multi_segment for disconnected contours; pixel-center normalized vertices; serialized float32+resample_segments then fillPoly. Holes are filled only when measured IoU passes. Reject entire page on any failed instance.'},
            'manifest_sha256':manifest_hashes, 'exact_mask_sha256':mask_hashes,
            'limitations':['Book-disjoint local fine-tuning split; upstream model training overlap is unknown.',
                           'Positive pages only. Missing balloon labels are not treated as verified negatives.',
                           'Current speech validation/test are development-only and not independent acceptance.',
                           'No scantrad data included. Current validation -1047 excluded as duplicate art of training -1044.',
                           'Training polygons approximate source RLE; use exact_mask JSONL for full-resolution evaluation.',
                           'Local Manga109-s research use; retain source attribution and supplied annotation license.']}
    write_json(output/'provenance.json', report)
    print(json.dumps({'prepared':str(output), 'counts':dict(counts), 'polygon_quality':report['polygon_quality']}), flush=True)
    return report


def prepare_current_boxes(root, output, manifests, link_methods, source_hashes, exclusions):
    summary={}
    for source_split, split in (('train', 'train'), ('valid', 'validation'), ('test', 'test')):
        folder=root/'training_data/speech-bubbles-detection'/source_split
        path=folder/'_annotations.coco.json'
        source_hashes[str(path)]=digest(path)
        data=json.loads(path.read_text(encoding='utf-8'))
        grouped=defaultdict(list)
        for ann in data['annotations']:
            if ann['category_id'] not in range(1, 7):
                raise ValueError('Unexpected current-data bubble category')
            grouped[ann['image_id']].append(ann)
        rows=[]
        for image in data['images']:
            source=folder/image['file_name']
            if source_split=='valid' and image['file_name'].startswith('-1047-_jpg.'):
                exclusions.append({'source':str(source), 'reason':'current_valid_duplicate_of_train_-1044'})
                continue
            annotations=[]
            for ann in grouped[image['id']]:
                x, y, w, h=ann['bbox']
                if min(w, h)<=0:
                    exclusions.append({'source':str(source), 'annotation_id':ann['id'], 'reason':'current_nonpositive_box'})
                    continue
                if min(x, y)<0 or x+w>image['width']+1 or y+h>image['height']+1:
                    raise ValueError('Current box exceeds image bounds')
                annotations.append(ann)
            with Image.open(source) as opened:
                if opened.size!=(image['width'], image['height']):
                    raise ValueError('Current image dimensions differ')
            stem=f'speech_{source_split}_{image["id"]:05d}'
            target=output/'boxes'/'images'/('current_'+split)/(stem+'.jpg')
            link_methods[link_image(source, target)]+=1
            write_labels(output, 'boxes', 'current_'+split, stem,
                         [box_line(ann['bbox'], image['width'], image['height']) for ann in annotations])
            manifests[f'current_{split}'].append(target.as_posix())
            if split=='train':
                manifests['boxes_train'].append(target.as_posix())
            rows.append({'image_path':target.as_posix(), 'source_image':str(source), 'source_sha256':digest(source),
                         'width':image['width'], 'height':image['height'], 'annotations':annotations})
        with (output/f'current_{split}_boxes.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        summary[split]={'images':len(rows), 'boxes':sum(len(row['annotations']) for row in rows)}
    return summary


def write_yaml(path, output, train, validation, test):
    path.write_text(f'path: {output.as_posix()}\ntrain: {train}\nval: {validation}\ntest: {test}\nnames:\n  0: balloon\n', encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='training_data/bubble_joint_20260922')
    parser.add_argument('--minimum-iou', type=float, default=0.98)
    args=parser.parse_args()
    if not 0<args.minimum_iou<=1:
        parser.error('--minimum-iou must be in (0,1]')
    prepare(args)
