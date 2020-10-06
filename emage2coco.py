import json
import os
import argparse
from functools import *
import numpy as np


def get_bbox(xs, ys):
    return [
        np.min(xs),
        np.min(ys),
        np.max(xs) - np.min(xs),
        np.max(ys) - np.min(ys)
    ]


def refactor_annotations(config_path):
    image_id, config = config_path
    with open(config, 'r') as f:
        config = json.load(f)

    categories = [
        'Background',
        'Bridging defect',
        'Bridging defect 1',
        'Overkill',
        'Single Overkill',
    ]
    annotations = [{
        'image_id': image_id,
        'segmentation': [reduce(lambda x, y:
                                x + [y[0], y[1]],
                                zip(config['regions'][k]['List_X'], config['regions'][k]['List_Y']),
                                [])],
        'iscrowd': 0,
        'bbox': get_bbox(config['regions'][k]['List_X'], config['regions'][k]['List_Y']),
        'category_id': categories.index(config['classId'][int(k)]),
    } for k in config['regions'].keys()]

    return annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser('New Solutions - EmageAI')
    parser.add_argument('-e', '--emage_path', type=str, default='...', help='emage data path')
    parser.add_argument('-o', '--out_path', type=str, default='datasets/emage', help='output directory')
    parser.add_argument('-p', '--project_name', type=str, default='emage', help='Project name')
    args = parser.parse_args()

    train_path = os.path.join(args.emage_path, 'Train', 'OriginImage')
    val_path = os.path.join(args.emage_path, 'Validation')
    ann_path = os.path.join(args.out_path, 'annotations')

    os.makedirs(ann_path, exist_ok=True)

    train_set = list(filter(lambda x: x.endswith('.json'), os.listdir(train_path)))
    val_set = list(filter(lambda x: x.endswith('.json'), os.listdir(val_path)))
    train_set = [os.path.join(train_path, x) for x in train_set]
    val_set = [os.path.join(val_path, x) for x in val_set]

    train_annotations = {
        'info': {
            'description': 'Emage Dataset 2020',
            'version': 1.0,
            'year': 2020,
        },
        'categories': [
            {'supercategory': 'reject', 'id': 1, 'name': 'Bridging defect'},
            {'supercategory': 'reject', 'id': 2, 'name': 'Bridging defect 1'},
            {'supercategory': 'pass', 'id': 3, 'name': 'Overkill'},
            {'supercategory': 'pass', 'id': 4, 'name': 'Single Overkill'},
        ],
        'images': reduce(lambda x, y:
                         x + [{
                             'id': len(x),
                             'width': 128,
                             'height': 128,
                             'file_name': os.path.basename(y.split('.json')[0])
                         }], train_set, []),
        'annotations': reduce(lambda annots, config_path:
                              annots + list(
                                  map(lambda ann: {
                                      'id': len(annots) + ann[0],
                                      **ann[1],
                                  }, enumerate(refactor_annotations(config_path)))
                              ),
                              enumerate(train_set), [])
    }
    val_annotations = {
        'info': {
            'description': 'Emage Dataset 2020',
            'version': 1.0,
            'year': 2020,
        },
        'categories': [
            {'supercategory': 'reject', 'id': 1, 'name': 'Bridging defect'},
            {'supercategory': 'reject', 'id': 2, 'name': 'Bridging defect 1'},
            {'supercategory': 'pass', 'id': 3, 'name': 'Overkill'},
            {'supercategory': 'pass', 'id': 4, 'name': 'Single Overkill'},
        ],
        'images': reduce(lambda x, y:
                         x + [{
                             'id': len(x),
                             'width': 128,
                             'height': 128,
                             'file_name': os.path.basename(y.split('.json')[0])
                         }], val_set, []),
        'annotations': reduce(lambda annots, config_path:
                              annots + list(
                                  map(lambda ann: {
                                      'id': len(annots) + ann[0],
                                      **ann[1],
                                  }, enumerate(refactor_annotations(config_path)))
                              ),
                              enumerate(val_set), [])
    }

    json.dump(train_annotations, open(os.path.join(ann_path, 'instances_train.json'), 'w'))
    json.dump(val_annotations, open(os.path.join(ann_path, 'instances_val.json'), 'w'))
