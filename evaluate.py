# Author: Shinre

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import argparse
import os
import time
from datetime import datetime
from shutil import copy

import torch
import cv2
import numpy as np
from matplotlib import colors
from tqdm import tqdm

from backbone import EfficientDetBackbone
from torch.backends import cudnn
from detection_module.utils import BBoxTransform, ClipBoxes
from utils.params import YamlParams
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box, boolean_string

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)


def display(preds, imgs, config, label=None, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = config.obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, config.obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(label or f'test/img_inferred_d{config.compound_coef}_this_repo_{i}.jpg', imgs[i])


def compute_intersection(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = np.asarray([b if box_area >= b else box_area for b in boxes_area])
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_intersection(box2, boxes1, area2[i], area1)
    return overlaps


def define_args():
    parser = argparse.ArgumentParser('New Solutions - EmageAI')
    parser.add_argument('command', help='One of the commands\'rois\' | \'report\'')
    parser.add_argument('-p', '--project', type=str, default='emage', help='project file that contains parameters')
    parser.add_argument('-d', '--dataset', type=str, default='datasets/emage/test', help='dataset path')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-w', '--weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, '
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--sets', type=str, default="['Reject', 'Pass']", help='List of set to evaluate')
    parser.add_argument('--thresholds', type=str, default='[0.05, 0.05, 0.4, 0.4]', help='Evaluate threshold')
    parser.add_argument('--fail_ids', type=str, default='[0, 1]', help='Fail ids')
    parser.add_argument('--pass_ids', type=str, default='[2, 3]', help='Pass ids')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = define_args()

    # Config
    params = YamlParams(f'projects/{opt.project}.yml')


    class InferenceConfig(object):
        compound_coef = opt.compound_coef
        force_input_size = None  # set None to use default size
        obj_list = params.obj_list

        # replace this part with your project's anchor config
        anchor_ratios = eval(params.anchor_ratios)
        anchor_scales = eval(params.anchor_scales)
        crop_size = params.crop_size

        threshold = params.threshold
        iou_threshold = params.iou_threshold


    config = InferenceConfig()

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    color_list = standard_to_bgr(STANDARD_COLORS)
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    model = EfficientDetBackbone(compound_coef=config.compound_coef,
                                 num_classes=len(config.obj_list),
                                 ratios=config.anchor_ratios,
                                 scales=config.anchor_scales)
    model.load_state_dict(torch.load(opt.weights))
    model.requires_grad_(False)
    model.eval()
    if opt.command == 'report':
        square_size = params.square_size
        red_box = ((config.crop_size - square_size) / 2, (config.crop_size - square_size) / 2, (config.crop_size + square_size) / 2, (config.crop_size + square_size) / 2)
        fail_ids = eval(opt.fail_ids)
        pass_ids = eval(opt.pass_ids)
        thresholds = eval(opt.thresholds)

        if os.path.isfile(opt.dataset):
            raise ValueError("dataset must be a folder")

        assert 'Pass' in os.listdir(opt.dataset) and 'Reject' in os.listdir(opt.dataset)

        metrics = {
            f'{set_name} Folder': {
                'Pass': 0,
                'Reject': 0,
                'Unknown': 0
            }
            for set_name in eval(opt.sets)
        }

        sets = eval(opt.sets)

        evaluated_path = f'test/evaluated_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        underkill_path = os.path.join(evaluated_path, 'UKs')
        overkill_path = os.path.join(evaluated_path, 'OKs')
        unknown_path = os.path.join(evaluated_path, 'UNs')
        os.makedirs(evaluated_path, exist_ok=True)
        os.makedirs(underkill_path, exist_ok=True)
        os.makedirs(overkill_path, exist_ok=True)
        os.makedirs(unknown_path, exist_ok=True)

        for set_name in sets:
            img_folder = os.path.join(opt.dataset, set_name)
            img_ids = [os.path.join(img_folder, f) for f in
                       filter(lambda file: file.endswith('.bmp'), os.listdir(os.path.join(img_folder)))]

            progress_bar = tqdm(range(len(img_ids)))
            for img_id in img_ids:
                # tf bilinear interpolation is different from any other's, just make do
                input_size = config.force_input_size or input_sizes[config.compound_coef]
                ori_imgs, framed_imgs, framed_metas = preprocess(img_id, crop_size=params.crop_size,
                                                                 max_size=input_size)

                if use_cuda:
                    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
                else:
                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

                x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
                if use_cuda:
                    model = model.cuda()
                if use_float16:
                    model = model.half()

                with torch.no_grad():
                    features, regression, classification, anchors = model(x)
                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()
                    out = postprocess(x,
                                      anchors, regression, classification,
                                      regressBoxes, clipBoxes,
                                      config.threshold, config.iou_threshold)
                out = invert_affine(framed_metas, out)

                if opt.debug:
                    display(out, ori_imgs, config, label=os.path.basename(img_id), imshow=True, imwrite=False)

                mark = []  # None for Unknown, True for Pass, False for Reject
                for roi, class_id, score in zip(out[0]['rois'], out[0]['class_ids'], out[0]['scores']):
                    iou = compute_overlaps(np.asarray([roi]), np.asarray([red_box]))
                    if iou > 0.05:
                        if class_id in pass_ids and score > thresholds[class_id]:
                            mark.append('Pass')
                        if class_id in fail_ids and score > thresholds[class_id]:
                            mark.append('Fail')
                    else:
                        mark.append('Unknown')

                if 'Fail' in mark:
                    metrics['%s Folder' % set_name]['Reject'] += 1
                    # Write OKs
                    if set_name == 'Pass':
                        copy(img_id, overkill_path)
                elif all([m == 'Unknown' for m in mark]):
                    metrics['%s Folder' % set_name]['Unknown'] += 1
                    # Write Unknown
                    copy(img_id, unknown_path)
                else:
                    metrics['%s Folder' % set_name]['Pass'] += 1
                    # Write UKs
                    if set_name == 'Reject':
                        copy(img_id, underkill_path)

                progress_bar.set_description(f"{set_name} directory "
                                             f"[{', '.join([str(v) for v in metrics['%s Folder' % set_name].values()])}]")
                progress_bar.update()

        with open(f'{evaluated_path}/metrics.csv', 'w+') as f:
            data = 'Set Name, Pass, Reject, Unknown\n'
            data += '\n'.join([','.join(list([set_name] + [str(v) for v in metrics[set_name].values()]))
                               for set_name in metrics.keys()])
            f.write(data)
            f.write('\nFail Threshold, Pass Threshold')
            f.write(f'\n{thresholds[0]}, {thresholds[2]}')
    elif opt.command == 'rois':
        while True:
            data_path = input('Data path (.quit to finish)')

            if data_path == '.quit':
                break

            if data_path.endswith('.bmp'):
                img_ids = [data_path]
            else:
                data_dir = data_path
                img_ids = [os.path.join(data_dir, f) for f in
                           filter(lambda file: file.endswith('.bmp'), os.listdir(data_path))]

            progress_bar = tqdm(range(len(img_ids)))
            for img_id in img_ids:
                progress_bar.update()

                # tf bilinear interpolation is different from any other's, just make do
                input_size = config.force_input_size or input_sizes[config.compound_coef]
                ori_imgs, framed_imgs, framed_metas = preprocess(img_id, crop_size=params.crop_size,
                                                                 max_size=input_size)

                if use_cuda:
                    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
                else:
                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

                x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

                if use_cuda:
                    model = model.cuda()
                if use_float16:
                    model = model.half()

                with torch.no_grad():
                    features, regression, classification, anchors = model(x)
                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()
                    out = postprocess(x,
                                      anchors, regression, classification,
                                      regressBoxes, clipBoxes,
                                      config.threshold, config.iou_threshold)

                out = invert_affine(framed_metas, out)

                print(out)
                if opt.debug:
                    display(out, ori_imgs, config, label=os.path.basename(img_id), imshow=False, imwrite=True)
