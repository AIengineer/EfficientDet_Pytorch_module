# repo: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git
import argparse
import os
import random

import numpy as np
import torch

from detection_module.script import train_det
from classification_module.script import train_cls
from utils.params import YamlParams, JsonConfig
from utils.utils import boolean_string
from imgaug import augmenters as iaa
import imgaug

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reproducibility
def make_reproducibility(config):
    manual_seed = config.manual_seed
    assert manual_seed is not None
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    imgaug.seed(manual_seed)
    iaa.iarandom.seed(manual_seed)
    if config.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
    else:
        torch.manual_seed(manual_seed)


def get_args():
    parser = argparse.ArgumentParser('New Solutions - EmageAI')
    parser.add_argument('command', type=str, help='Command expected values of `det` and `cls`')
    parser.add_argument('-c', '--config', type=str, default='projects/config.json', help='configuration')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, '
                             'set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('-d', '--data_path', type=str, default='datasets', help='the root folder of dataset')

    # Optional arguments for detailed customization
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training '
                             'will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--saved_path', type=str, default='logs')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    config = JsonConfig(opt.config)

    if config.manual_seed is None:
        manual_seed = config.params['manual_seed'] = random.randint(0, 1e7)
    else:
        manual_seed = config.manual_seed
    make_reproducibility(config)

    if opt.command == 'det':
        train_det(opt, config)
    elif opt.command == 'cls':
        train_cls(opt, config)
