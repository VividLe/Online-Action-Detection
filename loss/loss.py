import torch
import torch.nn.functional as F
from torch import nn


class SetCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion_CE = nn.CrossEntropyLoss(reduction=reduction)
        self.criterion_MSE = nn.MSELoss(reduction=reduction)
        self.criterion_KL = nn.KLDivLoss(reduction=reduction)

    def forward(self, outputs, targets, type):
        if type == 'CE':
            loss = self.criterion_CE(outputs, targets)
        elif type == 'MSE':
            outputs = F.softmax(outputs, dim=1)
            targets = F.softmax(targets, dim=1)
            loss = self.criterion_MSE(outputs, targets)
        elif type == 'KL':
            outputs = F.softmax(outputs, dim=1)
            targets = F.softmax(targets, dim=1)
            loss1 = self.criterion_KL(outputs, targets)
            loss2 = self.criterion_KL(targets, outputs)
            loss = loss1 + loss2
        return loss


if __name__ == '__main__':
    x_input = torch.randn(128, 21, 1)
    y_input = torch.randn(128, 21, 1)
    x = SetCriterion()

    y = x(x_input, y_input, 'KL')
    print(y)
