import datetime
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import loss as utl
from misc import init as cfg
from Dataload.ThumosDataset import THUMOSDataSet
from loss.evaluate import Colar_evaluate
from model.ColarModel import Colar_dynamic, Colar_static
import torch.nn.functional as F
import numpy as np
from misc.utils import backup_code


def train_one_epoch(model_dynamic,
                    model_static,
                    criterion,
                    data_loader, optimizer,
                    device, max_norm):
    model_static.train()
    model_dynamic.train()
    criterion.train()
    losses = 0
    i = 0
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        enc_target = enc_target.argmax(dim=-1)
        target = enc_target.to(device=device)

        # todo
        optimizer.zero_grad()

        enc_score_static = model_static(inputs[:, -1:, :], device)
        loss_static = criterion(enc_score_static, target[:, -1:], 'CE')

        enc_score_dynamic = model_dynamic(inputs)
        loss_dynamic = criterion(enc_score_dynamic[:, :, -1:], target[:, -1:], 'CE')

        loss_KL = criterion(enc_score_dynamic[:, :21, -1:], enc_score_static, 'KL')
        loss = loss_static + loss_dynamic + loss_KL

        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_static.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model_dynamic.parameters(), max_norm)

        optimizer.step()
        losses += loss
        i = i + 1
        print('\r train-------------------{:.4f}%'.format((i / 1415) * 100), end='')
    return losses / i, losses


def evaluate(model_dynamic,
             model_static,
             data_loader, device):
    model_static.eval()
    model_dynamic.eval()

    score_val_x = []
    target_val_x = []

    i = 0
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        target = enc_target.to(device)
        target_val = target[:, -1:, :21]

        with torch.no_grad():
            enc_score_dynamic = model_dynamic(inputs)
            enc_score_static = model_static(inputs[:, -1:, :], device)

        enc_score_static = enc_score_static.permute(0, 2, 1)
        enc_score_static = enc_score_static[:, :, :21]

        enc_score_dynamic = enc_score_dynamic.permute(0, 2, 1)
        enc_score_dynamic = enc_score_dynamic[:, -1:, :21]

        score_val = enc_score_static * 0.3 + enc_score_dynamic * 0.7
        score_val = F.softmax(score_val, dim=-1)

        score_val = score_val.contiguous().view(-1, 21).cpu().numpy()
        target_val = target_val.contiguous().view(-1, 21).cpu().numpy()

        score_val_x += list(score_val)
        target_val_x += list(target_val)
        print('\r train-------------------{:.4f}%'.format((i / 1600) * 100), end='')
        i += 1
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T
    print(all_probs.shape, all_classes.shape)
    results = {'probs': all_probs, 'labels': all_classes}

    return results


def main(args):
    log_file = backup_code(args.exp_name)
    seed = args.seed + cfg.get_rank()
    cfg.set_seed(seed)

    device = torch.device('cuda:' + str(args.cuda_id))
    model_static = Colar_static(args.input_size, args.numclass, device, args.kmean)
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)

    model_dynamic.apply(cfg.weight_init)
    model_dynamic.to(device)
    model_static.apply(cfg.weight_init)
    model_static.to(device)

    criterion = utl.SetCriterion().to(device)
    optimizer = torch.optim.Adam([
        {"params": model_static.parameters()},
        {"params": model_dynamic.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = THUMOSDataSet(flag='train', args=args)
    dataset_val = THUMOSDataSet(flag='test', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, loss = train_one_epoch(
            model_dynamic,
            model_static,
            criterion, data_loader_train, optimizer, device, args.clip_max_norm)

        lr_scheduler.step()
        print('epoch:{}------loss:{}'.format(epoch, train_loss))

        test_stats = evaluate(
            model_dynamic,
            model_static,
            data_loader_val, device)
        print('---------------Calculation of the map-----------------')
        Colar_evaluate(test_stats, epoch, args.command, log_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if '__main__' == __name__:
    args = cfg.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['THUMOS']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
