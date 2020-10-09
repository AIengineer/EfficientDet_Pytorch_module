import datetime
import json
import os
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from detection_module.dataset import Resizer, Normalizer, Augmenter, collater, DataGenerator
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from detection_module.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, save_checkpoint
from optimizers.srsgd import SRSGD

class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train_det(opt, cfg):
    # # Write history
    # if 'backlog' not in opt.config:
    #     with open(os.path.join(opt.saved_path, f'{opt.project}_backlog.yml'), 'w') as f:
    #         doc = open(f'projects/{opt.project}.yml', 'r')
    #         f.write('#History log file')
    #         f.write(f'\n__backlog__: {now.strftime("%Y/%m/%d %H:%M:%S")}\n')
    #         f.write(doc.read())
    #         f.write('\n# Manual seed used')
    #         f.write(f'\nmanual_seed: {cfg.manual_seed}')
    # else:
    #     with open(os.path.join(opt.saved_path, f'{opt.project}_history.yml'), 'w') as f:
    #         doc = open(f'projects/{opt.project}.yml', 'r')
    #         f.write(doc.read())

    training_params = {'batch_size': cfg.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': cfg.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    training_set = DataGenerator(
        data_path=os.path.join(opt.data_path, 'Train'),
        class_ids=cfg.dictionary_class_name.keys(),
        transform=transforms.Compose([Augmenter(),
                                      Normalizer(mean=cfg.mean, std=cfg.std),
                                      Resizer(input_sizes[cfg.compound_coef])]),
        pre_augments=['', *[f'{aug}_' for aug in cfg.augment_list]] if cfg.augment_list else None
    )
    training_generator = DataLoader(training_set, **training_params)

    val_set = DataGenerator(
        # root_dir=os.path.join(opt.data_path, cfg.project_name),
        data_path=os.path.join(opt.data_path, 'Validation'),
        class_ids=cfg.dictionary_class_name.keys(),
        transform=transforms.Compose([Normalizer(mean=cfg.mean, std=cfg.std),
                                      Resizer(input_sizes[cfg.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(cfg.dictionary_class_name), compound_coef=cfg.compound_coef,
                                 ratios=eval(cfg.anchor_ratios), scales=eval(cfg.anchor_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print('[Warning] Don\'t panic if you see this, '
                  'this might be because you load a pretrained weights with different number of classes. '
                  'The rest of the weights should be loaded already.'
                  )

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if cfg.training_layer.lower() == 'heads':
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if cfg.num_gpus > 1 and cfg.batch_size // cfg.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if cfg.num_gpus > 0:
        model = model.cuda()
        if cfg.num_gpus > 1:
            model = CustomDataParallel(model, cfg.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), cfg.learning_rate)
    if cfg.optimizer.lower() == 'srsgd':
        optimizer = SRSGD(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4, iter_count=100)
    else:
        optimizer = torch.optim.SGD(model.parameters(), cfg.learning_rate, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Setup complete, then start training
    now = datetime.datetime.now()
    opt.saved_path = opt.saved_path + f'/trainlogs_{now.strftime("%Y%m%d_%H%M%S")}'
    if opt.log_path is None:
        opt.log_path = opt.saved_path
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    # Write history
    if 'backlog' not in opt.config:
        with open(os.path.join(opt.saved_path, f'{now.strftime("%Y%m%d%H%M%S")}.backlog.json'), 'w') as f:
            backlog = dict(cfg.to_pascal_case())
            backlog['__metadata__'] = 'Backlog at ' + now.strftime("%Y/%m/%d %H:%M:%S")
            json.dump(backlog, f)
    else:
        with open(os.path.join(opt.saved_path, f'{now.strftime("%Y%m%d%H%M%S")}.history.json'), 'w') as f:
            history = dict(cfg.to_pascal_case())
            history['__metadata__'] = now.strftime("%Y/%m/%d %H:%M:%S")
            json.dump(history, f)

    writer = SummaryWriter(opt.log_path + f'/tensorboard')

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(cfg.no_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.set_description(f'Skip {iter} < {step} - {last_epoch} * {num_iter_per_epoch}')
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if cfg.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=cfg.dictionary_class_name.keys())
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. '
                        'Total loss: {:.5f}'.format(step, epoch, cfg.no_epochs, iter + 1,
                                                    num_iter_per_epoch, cls_loss.item(),
                                                    reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classification_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(val_generator):
                with torch.no_grad():
                    imgs = data['img']
                    annot = data['annot']

                    if cfg.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    cls_loss, reg_loss = model(imgs, annot, obj_list=cfg.dictionary_class_name.keys())
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss_classification_ls.append(cls_loss.item())
                    loss_regression_ls.append(reg_loss.item())

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            progress_bar.set_description(
                'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}.'
                ' Total loss: {:1.5f}'
                .format(epoch, cfg.no_epochs, cls_loss, reg_loss, loss))

            writer.add_scalars('Loss', {'val': loss}, step)
            writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
            writer.add_scalars('Classification_loss', {'val': cls_loss}, step)

            if cfg.only_best_weights:
                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    save_checkpoint(model, f"{opt.saved_path}/det_d{cfg.compound_coef}_{epoch}_{step}.pth")
            else:
                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                save_checkpoint(model, f"{opt.saved_path}/det_d{cfg.compound_coef}_{epoch}_{step}.pth")

            model.train()

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                break
        print(f'[Info] Finished training. Best loss achieved {best_loss} at epoch {best_epoch}.')
    except KeyboardInterrupt:
        save_checkpoint(model, f"{opt.saved_path}/d{cfg.compound_coef}_{epoch}_{step}.pth")
        writer.close()
    writer.close()
