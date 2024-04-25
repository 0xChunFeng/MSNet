import argparse
import datetime

i = datetime.datetime.now()
amp = True
epochs = 200
logs_dir = ''
idea = ""

def get_tarin_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    # model
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data-root", default="", dest="root")
    parser.add_argument("--resolution", type=int, default=352)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument("--weight-decay", type=float, default=1e-3, dest="wd")
    parser.add_argument("--dropout-rate", type=float, default=0.2, dest="da")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    # AMP
    parser.add_argument("--if_AMP", type=bool, default=amp, dest="ifamp", help="if use AMP")

    # AugSub
    parser.add_argument(
        "--is-augsub", type=bool, default=False, dest="isaugsub", help="if use augsub")

    # MedAugment
    parser.add_argument(
        "--is-medaugment", type=bool, default=False, dest="isma", help="if use MedAugment")
    parser.add_argument(
        "--number_branch", type=int, default=3, dest="ma_num", help="MedAugment number of branch")

    parser.add_argument("--logs_dir", type=str, default=logs_dir, dest="logs_dir")
    parser.add_argument("--idea", type=str, default=idea, dest="idea")


    parser.add_argument("--is_demo", type=bool, default=False, dest="is_demo")



    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--train-dataset", type=str, default=""
    )
    parser.add_argument(
        "--test-dataset", type=str, default=""
    )
    parser.add_argument(
        "--dataset", type=str, default=""
    )
    parser.add_argument("--data-root", type=str, default="", dest="root")
    parser.add_argument("--resolution", type=int, default=352)
    parser.add_argument("--logs_dir", type=str, default=logs_dir, dest="logs_dir")
    parser.add_argument("--idea", type=str, default=idea, dest="idea")


    parser.add_argument("--model_type", type=str, default="best", dest="mt", help="best or last or epoch")
    parser.add_argument("--epoch_number", type=int, default=80, dest="epoch_num", help="trained model saved epochs")

    return parser.parse_args()
