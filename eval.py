import os
import glob
import argparse
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score
from skimage.io import imread
from skimage.transform import resize

def binary_pa(pred, gt):
    """
        calculate the pixel accuracy of two N-d volumes.
        pre: the segmentation volume of numpy array
        gt: the ground truth volume of numpy array
        """
    pa = ((pred == gt).sum()) / gt.size
    return pa


def eval(args):

    if args.test_dataset == "Kvasir":
        prediction_files = sorted(
            glob.glob(
                "./Predictions/Trained on {}/Tested on {}/*".format(
                    args.train_dataset, args.test_dataset
                )
            )
        )
        depth_path = args.root + "/test/masks/*"
        test_files = sorted(glob.glob(depth_path))
    elif args.test_dataset == "data08":
        prediction_files = sorted(
            glob.glob(
                "./Predictions/Trained on {}/Tested on {}/*".format(
                    args.train_dataset, args.test_dataset
                )
            )
        )
        depth_path = args.root + "test_label/*"
        test_files = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC":
        prediction_files = sorted(
            glob.glob(
                "./Predictions/Trained on {}/Tested on {}/*".format(
                    args.train_dataset, args.test_dataset
                )
            )
        )
        depth_path = args.root + "/test/masks/*"
        test_files = sorted(glob.glob(depth_path))

    dice = []
    IoU = []
    accuracy = []
    precision = []
    recall = []

    for i in range(len(test_files)):
        pred = np.ndarray.flatten(imread(prediction_files[i]) / 255) > 0.5
        gt = (
            resize(imread(test_files[i]), (int(352), int(352)), anti_aliasing=False)
            > 0.5
        )

        if len(gt.shape) == 3:
            gt = np.mean(gt, axis=2)
        gt = np.ndarray.flatten(gt)


        dice.append(f1_score(gt, pred))
        IoU.append(jaccard_score(gt, pred))
        accuracy.append(accuracy_score(gt, pred))
        precision.append(precision_score(gt, pred))
        recall.append(recall_score(gt, pred))

        if i + 1 < len(test_files):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, accuracy={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    i + 1,
                    len(test_files),
                    100.0 * (i + 1) / len(test_files),
                    np.mean(dice),
                    np.mean(IoU),
                    np.mean(accuracy),
                    np.mean(precision),
                    np.mean(recall),
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, accuracy={:.6f}, precision={:.6f}, recall={:.6f}".format(
                    i + 1,
                    len(test_files),
                    100.0 * (i + 1) / len(test_files),
                    np.mean(dice),
                    np.mean(IoU),
                    np.mean(accuracy),
                    np.mean(precision),
                    np.mean(recall),
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train-dataset", type=str, default=""
    )
    parser.add_argument(
        "--test-dataset", type=str, default=""
    )
    parser.add_argument("--data-root", type=str, default="", dest="root")


    return parser.parse_args()


def main():
    args = get_args()
    eval(args)


if __name__ == "__main__":
    main()

