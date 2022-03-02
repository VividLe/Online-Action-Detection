import torch
import torch.nn.functional as F
import numpy as np


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


@torch.no_grad()
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
