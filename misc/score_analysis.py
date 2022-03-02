import json
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from misc import config
from model.ColarModel import Colar_dynamic, Colar_static


def score_analy(args):
    np.set_printoptions(threshold=np.inf, precision=3)

    device = torch.device('cuda:1')
    model_static = Colar_static(args.input_size, args.numclass, device, args.kmean)
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)

    model_dict = torch.load(args.checkpoint)
    model_static.load_state_dict(model_dict['model_static'])
    model_dynamic.load_state_dict(model_dict['model_dynamic'])

    model_dynamic.to(device)
    model_static.to(device)

    target = pickle.load(
        open(osp.join(args.data_root, 'thumos_test_anno.pickle'), 'rb'))

    feature = pickle.load(open(osp.join(
        args.data_root, 'thumos_all_feature_test_V3.pickle'), 'rb'))

    target = target['video_test_0000615']
    feature = feature['video_test_0000615']

    model_dynamic.eval()
    model_static.eval()

    static_s_x = []
    dynamic_s_x = []
    for start, end in zip(
            range(503, 631, 1),
            range(567, 631, 1)):
        enc_target = torch.tensor(target['anno'][start:end])
        enc_feature_rgb = torch.tensor(feature['rgb'][start:end])
        enc_feature_flow = torch.tensor(feature['flow'][start:end])
        enc_feature = torch.cat((enc_feature_rgb, enc_feature_flow), dim=-1)

        inputs = enc_feature.to(device).unsqueeze(0)

        with torch.no_grad():
            enc_score_dynamic = model_dynamic(inputs)
            enc_score_static = model_static(inputs[:, -1:, :], device)

        enc_score_static = enc_score_static.permute(0, 2, 1)
        enc_score_static = enc_score_static[:, :, :21]
        static_s = F.softmax(enc_score_static, dim=-1)

        enc_score_dynamic = enc_score_dynamic.permute(0, 2, 1)
        enc_score_dynamic = enc_score_dynamic[:, -1:, :21]
        dynamic_s = F.softmax(enc_score_dynamic, dim=-1)

        static_s = static_s.contiguous().view(-1, 21).cpu().numpy()
        dynamic_s = dynamic_s.contiguous().view(-1, 21).cpu().numpy()

        static_s_x += list(static_s)
        dynamic_s_x += list(dynamic_s)

    print(static_s_x, dynamic_s_x)


if '__main__' == __name__:
    args = config.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['THUMOS']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']

    score_analy(args)
