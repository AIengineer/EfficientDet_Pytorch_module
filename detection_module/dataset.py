import glob
import imghdr
import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from imgaug import augmenters as iaa, BoundingBoxesOnImage, BoundingBox
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


class DataGenerator(Dataset):
    def __init__(self, data_path, class_ids, transform=None, pre_augments=None):
        self.data_path = data_path
        self.transform = transform
        self.pre_augments = pre_augments
        self.image_ids = self.load_image_ids()
        self.classes, self.labels = self.load_classes(class_ids)

    def load_image_ids(self):
        if self.pre_augments:
            data_path = os.path.join(self.data_path, 'OriginImage')
        else:
            data_path = self.data_path

        # Get image name for image id
        list_ids = []
        for image in glob.glob(data_path + "/*.bmp"):
            if imghdr.what(image):
                image_name = os.path.split(image)[1]
                list_ids.append(image_name)
        return list_ids

    def load_classes(self, class_ids):
        # load class names (name -> label)
        classes = {}
        for i, cls in enumerate(class_ids):
            classes[cls] = i

        # also load the reverse (label -> name)
        labels = {}
        for key, value in classes.items():
            labels[value] = key

        return classes, labels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img, prefix = self.load_image(idx)
        annot = self.load_annotations(prefix, idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        prefix, data_path = '', self.data_path
        if self.pre_augments:
            prefix = np.random.choice(self.pre_augments)
            #hard code for non augmentation
            prefix = ''
            if prefix == '':
                data_path = os.path.join(self.data_path, 'OriginImage')
            else:
                data_path = os.path.join(self.data_path, 'TransformImage')
        image_info = prefix + self.image_ids[image_index]
        path = os.path.join(data_path, image_info)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, prefix

    def load_annotations(self, prefix, idx):
        data_path = self.data_path
        if self.pre_augments:
            if prefix == '':
                data_path = os.path.join(self.data_path, 'OriginImage')
            else:
                data_path = os.path.join(self.data_path, 'TransformImage')

        # Get annotations
        with open(os.path.join(data_path, prefix + self.image_ids[idx] + '.json'), 'r') as f:
            data = json.load(f)
            regions = data['regions'].values()
            class_ids = data['classId']

            annotations = np.zeros((0, 5))
            for r, c in zip(regions, class_ids):
                if len(r['List_X']) == 0 or len(r['List_Y']) == 0:
                    continue
                tlx, tly = np.min(r['List_X']), np.min(r['List_Y'])
                brx, bry = np.max(r['List_X']), np.max(r['List_Y'])
                cls_id = self.classes[c]
                annotations = np.append(annotations, [[tlx, tly, brx, bry, cls_id]], axis=0)
        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.augmenter = iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
        ])

    def __call__(self, sample, flip_x=0.5):
        image, annots = sample['img'], sample['annot']
        image = self.augmenter(images=[image])[0]

        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample

# class Augmenter(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __init__(self):
#         self.augmenter = iaa.OneOf([
#             iaa.Identity(),
#             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255)),
#             iaa.Fliplr(1),
#             iaa.Flipud(1),
#             iaa.Rot90(k=1),
#             iaa.Rot90(k=2),
#             iaa.Rot90(k=3),
#             iaa.Sequential([
#                 iaa.Flipud(1),
#                 iaa.Rot90(k=1),
#             ]),
#             iaa.Sequential([
#                 iaa.Flipud(1),
#                 iaa.Rot90(k=2),
#             ]),
#             iaa.Sequential([
#                 iaa.Flipud(1),
#                 iaa.Rot90(k=3),
#             ])
#         ])
#
#     def __call__(self, sample):
#         image, annots = sample['img'], sample['annot']
#         bboxes = BoundingBoxesOnImage([BoundingBox(*annot[:4]) for annot in annots], shape=image.shape)
#         image_aug, bboxes_aug = self.augmenter(images=[image], bounding_boxes=bboxes)[0]
#         sample = {'img': image_aug, 'annot': annots}
#         return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) / 255. - self.mean) / self.std), 'annot': annots}


class AdvProp(object):
    def __call__(self, sample, flip_x=0.5):
        image, annots = sample['img'], sample['annot']

        return {'img': image.astype(np.float32) / 255 * 2 - 1, 'annot': annots}
