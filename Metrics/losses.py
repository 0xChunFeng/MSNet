import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class T_Loss(nn.Module):
    def __init__(self, dim, epsilon=1e-8, reduce='mean'):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.nu_tilde = nn.Parameter(torch.tensor(1.0))
        self.reduce = reduce

    def forward(self, inputs, target):
        delta = inputs - target
        nu = torch.exp(self.nu_tilde) + self.epsilon

        # Compute terms of the loss function
        first = -torch.lgamma((nu + self.dim) / 2)
        second = torch.lgamma(nu / 2)
        third = (self.dim / 2) * torch.log(torch.tensor(np.pi) * nu)

        fraction = delta * delta / nu
        fourth = ((nu + self.dim) / 2) * torch.log(1 + fraction)

        # Compute total loss
        loss = first + second + third + fourth

        # Reduce loss over the batch
        if self.reduce == 'mean':
            return torch.mean(loss)
        elif self.reduce == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"The reduction {self.reduce} is not implemented")

