import torch

def write2tensorboard(writer, epoch, loss, train_mDice, train_mIou, test_measure_mDice, test_measure_mIou, lr):
    writer.add_scalar(tag="train/loss", scalar_value=loss,
                                  global_step=epoch)
    writer.add_scalar(tag="train/mDIce", scalar_value=train_mDice,
                      global_step=epoch)
    writer.add_scalar(tag="train/mIou", scalar_value=train_mIou,
                      global_step=epoch)
    writer.add_scalar(tag="test/mDIce", scalar_value=test_measure_mDice,
                      global_step=epoch)
    writer.add_scalar(tag="test/mIou", scalar_value=test_measure_mIou,
                      global_step=epoch)
    writer.add_scalar(tag="train/learning rate", scalar_value=lr,
                      global_step=epoch)


def save_model(model_state_dict, optimizer_state_dict, epoch, loss, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "loss": loss,
        },
        save_path,
    )


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))