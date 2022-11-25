import torch
import torch.nn as nn
import torch.nn.functional as F


class siamese_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 1)

    def hidden_forward(self, x):
        x = F.relu(self.l1(x))
        x = self.bn1(x)
        return x

    def forward(self, x1, x2):
        x = torch.abs(x1.view(-1, 512) - x2.view(-1, 512))
        return torch.sigmoid(self.out(self.hidden_forward(x)))
