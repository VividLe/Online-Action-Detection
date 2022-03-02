import numpy as np
from scipy.ndimage import label
import os
import shutil
import datetime
import torch
import torch.distributed as dist


def get_actionness(ti_anno):
    class_line = np.argmax(ti_anno, axis=1)
    actionness_line = (class_line > 0).astype(np.float32)
    return actionness_line


def get_relevance(ti_anno):
    class_line = np.argmax(ti_anno, axis=1)
    binary_line = (class_line == class_line[-1]).astype(np.int32)
    indexed_line, _ = label(binary_line)
    t0_idx = indexed_line[-1]
    relation_line = (indexed_line == t0_idx).astype(np.float32)
    return relation_line


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


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def frame_level_map_n_cap(results):
    all_probs = results['probs']
    all_labels = results['labels']

    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(1, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum / np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    map = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)
    return map, all_cls_ap, cap, all_cls_acp


def backup_code(exp_name):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S_") + exp_name
    res_dir = 'output/' + time_str + '/'
    os.makedirs(res_dir)
    log_file = res_dir + 'results.txt'

    file_list = ['main.py']
    fol_list = ['Dataload', 'loss', 'model', 'tool', 'misc']

    for file in file_list:
        shutil.copyfile(file, res_dir + file)
    for folname in fol_list:
        shutil.copytree(folname, res_dir + folname)
    print('Codea backup at %s' % res_dir)
    return log_file
