import torch


class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score


def calculate_metrics_seg(out, target, e=1e-6):
    out = torch.sigmoid(out)
    out_binary = (out > 0.5).int()
    target_binary = (target > 0.5).int()
    intersection = torch.logical_and(out_binary, target_binary).sum(dim=(2, 3))
    union_dice = out_binary.sum(dim=(2, 3)) + target_binary.sum(dim=(2, 3))
    union_iou = union_dice - intersection
    dice_score = 2 * intersection / (union_dice + e)
    iou_score = intersection / (union_iou + e)
    correct_pixels = torch.eq(out_binary, target_binary).sum()
    total_pixels = target_binary.numel()
    pixel_accuracy = correct_pixels / total_pixels

    return dice_score.mean(), iou_score.mean()
    # return dice_score.mean(), iou_score.mean(), pixel_accuracy

