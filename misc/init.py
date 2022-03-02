import torch
import random
import numpy as np
import argparse
import os
import torch.distributed as dist
import torch.nn.init as init
import torch.nn as nn
import torch.backends.cudnn as cudnn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True
    print(cudnn.benchmark, cudnn.deterministic, cudnn.enabled)


def parse_args():
    parser = argparse.ArgumentParser(description='Online action detection')
    parser.add_argument('--exp_name', default='ColarTHUMOS14')

    # please adjust the path
    parser.add_argument('--data_root', default='./data/thumos14')
    parser.add_argument('--dataset_file', type=str, default='./data/thumos14/data_info.json')
    parser.add_argument('--kmean', type=str, default='./data/thumos14/exemplar.pickle')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/THUMOS-TSN-Kinetics.pth')

    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--cuda_id', default=1)
    parser.add_argument('--lr_drop', default=1, type=int)
    parser.add_argument('--input_size', default=4096, type=int)
    parser.add_argument('--enc_layers', default=64, type=int)
    parser.add_argument('--numclass', default=22, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--overlap', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--clip_max_norm', default=1., type=float)
    parser.add_argument('--feature_type', type=str, default='kinetics')
    parser.add_argument('--command', type=str, default='kinetics')
    args = parser.parse_args()

    return args


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
