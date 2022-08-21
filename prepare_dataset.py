import random
import click
import cv2
from loguru import logger
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


@logger.catch
def crop_minAreaRect(img, xc, yc, w, h, a):
    box = cv2.boxPoints(((xc, yc), (w, h), -a))
    w, h = int(w), int(h)
    box = np.int0(box)
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h - 1],
                        [0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped


@click.command()
@click.option('-a', '--annotations-path',
              type=click.Path(exists=True,
                              file_okay=False,
                              readable=True,
                              path_type=Path),
              default=Path('dataset_info/'))
@click.option('-s', '--save_path',
              type=click.Path(file_okay=False,
                              writable=True,
                              path_type=Path),
              default=Path('cropped/'))
@click.option('--no-split', is_flag=True)
@click.option('--reduce', type=float)
@logger.catch
def main(annotations_path: Path, save_path: Path, no_split: bool, reduce: float):
    annotations: list[Path]
    outputs: list[Path]
    if no_split:
        annotations = [annotations_path / 'imgur5k_annotations.json']
        outputs = [save_path / 'whole']
    else:
        annotations = [
            annotations_path / 'imgur5k_annotations_train.json',
            annotations_path / 'imgur5k_annotations_val.json',
            annotations_path / 'imgur5k_annotations_test.json',
        ]
        outputs = [
            save_path / 'train',
            save_path / 'val',
            save_path / 'test'
        ]

    annotations_path.mkdir(parents=True, exist_ok=True)
    for annotation_path, output_path in tqdm(zip(annotations, outputs)):
        words = {}
        output_path.mkdir(parents=True, exist_ok=True)
        annotation = json.load(annotation_path.open('r'))
        annotations = list(annotation['index_to_ann_map'].items())
        if reduce is not None:
            random.shuffle(annotations)
            annotations = annotations[:int(len(annotations) * reduce)]
        for index_id, ann_ids in tqdm(annotations, leave=False):
            img_info = annotation['index_id'][index_id]
            img = cv2.imread(img_info['image_path'])
            if img is None:
                continue
            for ann_id in ann_ids:
                info = annotation['ann_id'][ann_id]
                info['word'] = str(info['word'])
                if len(info['word']) == 0:
                    continue

                words[ann_id] = info['word']

                if (output_path / f'{ann_id}.png').exists():
                    continue
                img_cropped = crop_minAreaRect(img, *info['bounding_box'])
                cv2.imwrite(str(output_path / f'{ann_id}.png'), img_cropped)
        json.dump(words, (output_path / 'words.json').open('w'))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
