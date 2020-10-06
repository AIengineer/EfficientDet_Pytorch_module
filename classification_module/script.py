import datetime
import json
import os
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import Resizer, Normalizer, Augmenter, collater, DataGenerator
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm
from .model import EfficientNet as EffNet

from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, save_checkpoint


class EfficientNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = model

    def forward(self, imgs, annotations):
        logits = self.model(imgs)
        loss = self.criterion(logits, annotations)
        return logits, loss


def train_cls(opt, cfg):
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

    input_sizes = [224, 240, 260, 300, 380, 456, 528, 600]

    # training_set = CocoDataset(
    #     # root_dir=os.path.join(opt.data_path, params.project_name),
    #     root_dir=opt.data_path,
    #     set=params.train_set,
    #     transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                   # AdvProp(),
    #                                   Augmenter(),
    #                                   Resizer(input_sizes[cfg.compound_coef])]))

    training_set = DataGenerator(
        data_path=os.path.join(opt.data_path, 'Train', 'OriginImage'),
        class_ids=cfg.dictionary_class_name.keys(),
        transform=transforms.Compose([Augmenter(),
                                      Normalizer(mean=cfg.mean, std=cfg.std),
                                      Resizer(input_sizes[cfg.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    # val_set = CocoDataset(
    #     # root_dir=os.path.join(opt.data_path, params.project_name),
    #     root_dir=opt.data_path,
    #     set=params.val_set,
    #     transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                   Resizer(input_sizes[cfg.compound_coef])]))

    val_set = DataGenerator(
        # root_dir=os.path.join(opt.data_path, params.project_name),
        data_path=os.path.join(opt.data_path, 'Validation'),
        class_ids=cfg.dictionary_class_name.keys(),
        transform=transforms.Compose([Normalizer(mean=cfg.mean, std=cfg.std),
                                      Resizer(input_sizes[cfg.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EffNet.from_name(f'efficientnet-b{cfg.compound_coef}',
                             override_params={'num_classes': len(cfg.dictionary_class_name.keys())})

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
            print(ret)
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
    model = EfficientNetWrapper(model)

    if cfg.num_gpus > 0:
        model = model.cuda()
        if cfg.num_gpus > 1:
            model = CustomDataParallel(model, cfg.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), cfg.learning_rate)
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
            # metrics
            correct_preds = 0.

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

                    # if params.num_gpus == 1:
                    #     # if only one gpu, just send it to cuda:0
                    #     # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                    imgs = imgs.cuda()
                    annot = annot.cuda()

                    optimizer.zero_grad()
                    logits, loss = model(imgs, annot)
                    loss = loss.mean()

                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    epoch_loss.append(float(loss))

                    _, preds = torch.max(logits, dim=1)
                    correct_preds += torch.sum(preds == annot)
                    acc = correct_preds / ((step % num_iter_per_epoch + 1) * cfg.batch_size)

                    progress_bar.set_description('Step: {}. Epoch: {}/{}. Iteration: {}/{}. '
                                                 'Loss: {:.5f}. Accuracy: {:.5f}.'
                                                 .format(step, epoch, cfg.no_epochs, iter + 1, num_iter_per_epoch,
                                                         float(loss), float(acc)))
                    writer.add_scalars('Loss', {'train': float(loss)}, step)
                    writer.add_scalars('Accuracy', {'train': float(acc)}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                correct_preds = 0.
                fusion_matrix = torch.zeros(len(cfg.dictionary_class_name), len(cfg.dictionary_class_name)).cuda()
                model.eval()
                val_losses = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        # if params.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                        logits, loss = model(imgs, annot)
                        loss = loss.mean()

                        _, preds = torch.max(logits, dim=1)
                        correct_preds += torch.sum(preds == annot)

                        # Update matrix
                        for i, j in zip(preds, annot):
                            fusion_matrix[i, j] += 1

                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                val_acc = float(correct_preds) / (len(val_generator) * cfg.batch_size)

                progress_bar.set_description(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}. Accuracy: {:1.5f}. '
                        .format(epoch, cfg.no_epochs, val_loss.item(), val_acc)
                )

                # Calculate predictions and recalls
                preds_total = torch.sum(fusion_matrix, dim=1)
                recall_total = torch.sum(fusion_matrix, dim=0)
                predictions = {l: float(fusion_matrix[i, i]) / max(1, preds_total[i].item()) for l, i in val_set.classes.items()}
                recalls = {l: float(fusion_matrix[i, i]) / max(1, recall_total[i].item()) for l, i in val_set.classes.items()}

                writer.add_scalars('Loss', {'val': val_loss}, step)
                writer.add_scalars('Accuracy', {'val': val_acc}, step)
                writer.add_scalars('Predictions', predictions, step)
                writer.add_scalars('Recalls', recalls, step)

                print(fusion_matrix)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                save_checkpoint(model, f"{opt.saved_path}/cls_b{cfg.compound_coef}_{epoch}_{step}.pth")

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
        print(f'[Info] Finished training. Best loss achieved {best_loss} at epoch {best_epoch}.')
    except KeyboardInterrupt:
        save_checkpoint(model, f"{opt.saved_path}/cls_b{cfg.compound_coef}_{epoch}_{step}.pth")
        writer.close()
    writer.close()
