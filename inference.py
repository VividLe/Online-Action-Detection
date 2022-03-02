import json
from loss.evaluate import Colar_evaluate

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from Dataload.ThumosDataset import THUMOSDataSet
from model.ColarModel import Colar_dynamic, Colar_static
from misc import init


def evaluate(args):
    device = torch.device(args.cuda_id)
    model_static = Colar_static(args.input_size, args.numclass, device, args.kmean)
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)

    model_dict = torch.load(args.checkpoint)
    model_static.load_state_dict(model_dict['model_static'])

    model_dynamic.load_state_dict(model_dict['model_dynamic'])

    model_dynamic.to(device)
    model_static.to(device)

    dataset_val = THUMOSDataSet(flag='test', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    model_dynamic.eval()
    model_static.eval()

    score_val_x = []
    target_val_x = []
    i = 0
    for camera_inputs, enc_target in data_loader_val:
        inputs = camera_inputs.to(device)
        target_val = enc_target[:, -1:, :21].to(device)

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

    score_val_x = np.asarray(score_val_x)
    target_val_x = np.asarray(target_val_x)

    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T
    print(all_probs.shape, all_classes.shape)
    results = {'probs': all_probs, 'labels': all_classes}
    Colar_evaluate(results, 1, 'test', '')


if '__main__' == __name__:
    args = init.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['THUMOS']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']

    evaluate(args)
