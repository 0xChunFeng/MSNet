import os
import time
import numpy as np
import glob

import torch

import arguments
from Data import dataloaders
from Models import bestmodel
from Metrics import performance_metrics


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        train_images = args.root + "/train/images/*"
        train_input_paths = sorted(glob.glob(train_images))
        train_masks = args.root + "/train/masks/*"
        train_target_paths = sorted(glob.glob(train_masks))

        val_images = args.root + "/val/images/*"
        val_input_paths = sorted(glob.glob(val_images))
        val_masks = args.root + "/val/masks/*"
        val_target_paths = sorted(glob.glob(val_masks))

        test_images = args.root + "/test/images/*"
        test_input_paths = sorted(glob.glob(test_images))
        test_masks = args.root + "/test/masks/*"
        test_target_paths = sorted(glob.glob(test_masks))
        train_dataloader, test_dataloader, val_dataloader = dataloaders.get_dataloaders(
            args.dataset, False, args.resolution, train_input_paths, train_target_paths, val_input_paths,
            val_target_paths, test_input_paths,
            test_target_paths, batch_size=1
        )

    perf = performance_metrics.DiceScore()
    model = bestmodel.XXFormer(size=args.resolution)
    state_dict = torch.load(
        "./Trained models/" + args.logs_dir + "/" + args.mt + "_model_{}.pt".format(args.train_dataset)
    )
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)

    return device, test_dataloader, perf, model


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model = build(args)
    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
        os.makedirs("./Predictions/Trained on {}".format(args.train_dataset))
    if not os.path.exists(
        "./Predictions/Trained on {}/Tested on {}".format(
            args.train_dataset, args.test_dataset
        )
    ):
        os.makedirs(
            "./Predictions/Trained on {}/Tested on {}".format(
                args.train_dataset, args.test_dataset
            )
        )

    t = time.time()
    model.eval()
    dice_accumulator = []
    iou_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        mDice, mIou = performance_metrics.calculate_metrics_seg(output, target)
        dice_accumulator.append(mDice.item())
        iou_accumulator.append(mIou.item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        if i + 1 < len(test_dataloader):
            print(
                "\rPredict: [{}/{} ({:.1f}%)]\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(dice_accumulator),
                    np.mean(iou_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rPredict: [{}/{} ({:.1f}%)]\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(dice_accumulator),
                    np.mean(iou_accumulator),
                    time.time() - t,
                )
            )


def main():
    args = arguments.get_test_args()
    predict(args)


if __name__ == "__main__":
    main()

