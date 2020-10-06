import glob
import imghdr
import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
import cv2


class DataGenerator(Dataset):
    def __init__(self, data_path, class_ids, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_ids = self.load_image_ids()
        self.classes, self.labels = self.load_classes(class_ids)

    def load_image_ids(self):
        # Get image name for image id
        list_ids = []
        for image in glob.glob(os.path.join(self.data_path, "/*.bmp")):
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
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.image_ids[image_index]
        path = os.path.join(self.data_path, image_info)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_annotations(self, idx):
        # Get annotations
        with open(os.path.join(self.data_path, self.image_ids[idx] + '.json'), 'r') as f:
            data = json.load(f)
            class_ids = data['classId']
        return np.asarray(self.classes[class_ids[0]])

    def statistics(self):
        distribution = {label: 0 for label in self.classes.keys()}
        for img_name in self.image_ids:
            with open(os.path.join(self.data_path, img_name + '.json'), 'r') as f:
                data = json.load(f)
                label = data['classId'][0]
                distribution[label] += 1
        return distribution

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    annots = torch.from_numpy(np.stack(annots, axis=0)).to(torch.long)
    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annots, 'scale': scales}


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

        # annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': annots, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.augmenter = iaa.OneOf([
            iaa.Identity(),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255)),
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Rot90(k=1),
            iaa.Rot90(k=2),
            iaa.Rot90(k=3),
            iaa.Sequential([
                iaa.Flipud(1),
                iaa.Rot90(k=1),
            ]),
            iaa.Sequential([
                iaa.Flipud(1),
                iaa.Rot90(k=2),
            ]),
            iaa.Sequential([
                iaa.Flipud(1),
                iaa.Rot90(k=3),
            ])
        ])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = self.augmenter(images=[image])[0]
        sample = {'img': image, 'annot': annots}
        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) / 255. - self.mean) / self.std), 'annot': annots}
