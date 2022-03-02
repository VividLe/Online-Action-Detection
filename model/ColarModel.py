import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Colar_static(nn.Module):
    def __init__(self, ch_in, ch_out, device, kmean_path):
        super(Colar_static, self).__init__()
        chennel = 1024
        self.conv1_3_Ek = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_Ev = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_k = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_v = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv2_1_W = nn.Conv1d(chennel, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv1d(chennel * 2, 21, kernel_size=1, stride=1, padding=0)
        self.opt = nn.ReLU()
        self.static_feature = list()

        static = np.load(kmean_path, allow_pickle=True)
        for i in range(0, 21, 1):
            x = np.asarray(static[i])
            x = torch.from_numpy(x).squeeze().unsqueeze(0)
            self.static_feature.append(x.permute(0, 2, 1))
            self.static_feature[i] = self.static_feature[i].to(device)

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.matmul(value, y_weight)
        return y_sum

    def forward(self, x, device):
        x = x[:, -1:, :]
        x = x.permute(0, 2, 1)
        k = self.conv1_3_k(x)
        v = self.conv1_3_v(x)

        feature_w = torch.empty(x.shape[0], 21).to(device)
        for i in range(0, 21, 1):
            static_feature = self.static_feature[i]

            Ek = self.conv1_3_Ek(static_feature)
            Ev = self.conv1_3_Ev(static_feature)

            weight = self.weight(Ek, k)
            sum = self.sum(Ev, weight)
            if i == 0:
                feature_E = sum
            else:
                feature_E = torch.cat((feature_E, sum), dim=-1)

            feature_w[:, i:i + 1] = self.conv2_1_W(sum).squeeze(-1)

        feature_E = feature_E.to(device)
        feature_w = F.softmax(feature_w, dim=-1).unsqueeze(-1)
        feature_E = torch.bmm(feature_E, feature_w)
        out = torch.cat((v, feature_E), dim=1)
        out = self.opt(out)

        out = self.conv2_1(out)

        return out


class Colar_dynamic(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Colar_dynamic, self).__init__()
        chennel = 1024
        self.conv1_3_k = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_v = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_feature = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=1)

        self.conv2_3_k = nn.Conv1d(chennel, chennel, kernel_size=3, stride=1, padding=1)
        self.conv2_3_v = nn.Conv1d(chennel, chennel, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv1d(chennel, ch_out, kernel_size=1, stride=1, padding=0)

        self.opt = nn.ReLU()

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.bmm(value, y_weight)
        sum = value[:, :, -1:] + y_sum
        return torch.cat((value[:, :, :-1], sum), dim=-1)

    def forward(self, input):
        input = input.permute(0, 2, 1)

        k = self.conv1_3_k(input)
        v = self.conv1_3_v(input)
        y_weight = self.weight(k, k[:, :, -1:])
        feat1 = self.sum(v, y_weight)
        feat1 = self.opt(feat1)

        k = self.conv2_3_k(feat1)
        v = self.conv2_3_v(feat1)
        y_weight = self.weight(k, k[:, :, -1:])
        feat2 = self.sum(v, y_weight)
        feat2 = self.opt(feat2)

        return self.conv3_1(feat2)


if __name__ == '__main__':
    import time
    
    dynamic_input = torch.randn(1, 64, 4096)
    static_input = torch.randn(1, 1, 4096)
    device = torch.device("cuda:0")
    dynamic_input = dynamic_input.to(device)
    static_input = static_input.to(device)

    model_d = Colar_dynamic(4096, 22)
    model_s = Colar_static(4096, 21, device)
    model_d.to(device)
    model_s.to(device)

    start = time.time()
    for i in range(290):
        out1 = model_d(dynamic_input)
        out2 = model_s(static_input, device)
    print(time.time() - start)
