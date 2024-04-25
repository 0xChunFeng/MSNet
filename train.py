import random
import sys
import os
import time
import numpy as np
import glob

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import warnings

import arguments
from Data import dataloaders
from Models import bestmodel
from Metrics import performance_metrics
from Metrics import losses

import torch.nn.functional as F

from utils.NativeScalerWithGradNormCount import NativeScalerWithGradNormCount as NativeScaler

from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

from utils.util import write2tensorboard, save_model

warnings.filterwarnings("ignore", category=UserWarning)


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

@torch.no_grad()
def test(model, device, test_loader=None, epoch=None, writer=None):
    t = time.time()
    model.eval()
    dice_accumulator = []
    iou_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        mDice, mIou = performance_metrics.calculate_metrics_seg(output, target)
        dice_accumulator.append(mDice.item())
        iou_accumulator.append(mIou.item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(dice_accumulator),
                    np.mean(iou_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(dice_accumulator),
                    np.mean(iou_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(dice_accumulator), np.mean(iou_accumulator)


@torch.no_grad()
def test_all(model, device, test_loader=None, epoch=None):
    all_dice_accumulator = []
    all_iou_accumulator = []
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        test_images = "D:/coder/data/all_polyp/TestDataset/{}/images/*".format(_data_name)
        test_input_paths = sorted(glob.glob(test_images))
        test_masks = "D:/coder/data/all_polyp/TestDataset/{}/masks/*".format(_data_name)
        test_target_paths = sorted(glob.glob(test_masks))
        train_dataloader, test_loader, val_dataloader = dataloaders.get_dataloaders(
            dataste='all', is_ma=False, resolution=352, test_input_paths=test_input_paths, test_target_paths=test_target_paths
        )
        t = time.time()
        model.eval()
        dice_accumulator = []
        iou_accumulator = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            mDice, mIou = performance_metrics.calculate_metrics_seg(output, target)
            dice_accumulator.append(mDice.item())
            iou_accumulator.append(mIou.item())
            if batch_idx + 1 < len(test_loader):
                print(
                    "\rTest  DataSet: {} [{}/{} ({:.1f}%)]\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                        _data_name,
                        batch_idx + 1,
                        len(test_loader),
                        100.0 * (batch_idx + 1) / len(test_loader),
                        np.mean(dice_accumulator),
                        np.mean(iou_accumulator),
                        time.time() - t,
                    ),
                    end="",
                )
            else:
                print(
                    "\rTest  DataSet: {} [{}/{} ({:.1f}%)]\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                        _data_name,
                        batch_idx + 1,
                        len(test_loader),
                        100.0 * (batch_idx + 1) / len(test_loader),
                        np.mean(dice_accumulator),
                        np.mean(iou_accumulator),
                        time.time() - t,
                    )
                )
        all_dice_accumulator.append(np.mean(dice_accumulator))
        all_iou_accumulator.append(np.mean(iou_accumulator))

    return all_dice_accumulator, all_iou_accumulator


def train_epoch(model, device, resolution, train_loader, optimizer, epoch, Dice_loss, BCE_loss, T_Loss, Iou_loss, scheduler, warmup_scheduler,
                mixup = False, isaugsub=False, ifamp=False):
    t = time.time()
    model.train()
    loss_accumulator = []
    bce_loss_accumulator = []
    dice_loss_accumulator = []
    loss_scaler = NativeScaler()
    i = 0
    dice_accumulator = []
    iou_accumulator = []

    # if ifamp:
    scaler = GradScaler()

    for batch_idx, (data, target) in enumerate(train_loader):
        i = i + 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if ifamp:
            with autocast():
                output = model(data)
                bce_loss = BCE_loss(output, target)
                dice_loss = Dice_loss(output, target)
                loss = bce_loss + dice_loss

        else:
            output = model(data)
            loss = BCE_loss(output, target) + Dice_loss(output, target)
        mDice, mIou = performance_metrics.calculate_metrics_seg(output, target)
        dice_accumulator.append(mDice.item())
        iou_accumulator.append(mIou.item())
        if isaugsub:
            if ifamp:
                scaler.scale(loss / 2).backward()
                scaler.step(optimizer)
                scaler.update()
                with autocast():
                    output_sub = model(data.detach(), augsub_type='masking')
                    loss_sub = BCE_loss(output_sub, output.detach())
                scaler.scale(loss_sub / 2).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_scaler(loss / 2, optimizer, parameters=model.parameters(), update_grad=False)
                output_sub = model(data.detach(), augsub_type='masking')
                loss_sub = BCE_loss(output_sub, output.detach())
                loss_scaler(loss_sub / 2, optimizer, parameters=model.parameters(), update_grad=True, retain_graph=False)

        else:
            if ifamp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_scaler(loss, optimizer, parameters=model.parameters())

        loss_accumulator.append(loss.item())

        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tAverage mDice: {:.6f}\tAverage mIou: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    np.mean(dice_accumulator),
                    np.mean(iou_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator), np.mean(dice_accumulator), np.mean(iou_accumulator)


def train(args):
    (
        device,
        resolution,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        Dice_loss,
        BCE_loss,
        T_Loss,
        Iou_loss,
        model,
        optimizer,
        scheduler,
        warmup_scheduler,
        logs_dir,
        mixup,
        isaugsub,
        writer,
        train_writer,
        test_writer,
    ) = build(args)

    if not os.path.exists("./Trained models/" + logs_dir):
        os.makedirs("./Trained models/" + logs_dir)

    best_dice = None
    best_iou = None
    all_best_dice = [None, None, None, None, None]
    all_best_iou = [None, None, None, None, None]
    all_best_epoch = [0, 0, 0, 0, 0]
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        try:
            loss, train_mDice, train_mIou = train_epoch(
                model, device, resolution, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss, T_Loss, Iou_loss, scheduler,
                warmup_scheduler, mixup, isaugsub=isaugsub, ifamp=args.ifamp
            )
            if args.dataset == 'all':
                all_test_measure_mDice, all_test_measure_mIou = test_all(
                    model, device, epoch)
            else:
                test_measure_mDice, test_measure_mIou = test(
                    model, device, val_dataloader, epoch, writer)
            if args.lrs == "true":
                warmup_scheduler.step()


        except KeyboardInterrupt:
            save_path = "./Trained models/" + logs_dir + "/best_model_" + args.dataset + ".pt"
            save_model(model.state_dict(), optimizer.state_dict(), epoch, loss, save_path)
            print("Training interrupted by user")
            sys.exit(0)

        if args.dataset == 'all':
            dataset_index = 0
            for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                if all_best_dice[dataset_index] == None or all_test_measure_mDice[dataset_index] > all_best_dice[dataset_index]:
                    save_path = "./Trained models/" + logs_dir + "/best_model_" + _data_name + ".pt"
                    print("{} Saving...".format(_data_name))
                    save_model(model.state_dict(), optimizer.state_dict(), epoch, loss, save_path)
                    all_best_dice[dataset_index] = all_test_measure_mDice[dataset_index]
                    all_best_iou[dataset_index] = all_test_measure_mIou[dataset_index]
                    all_best_epoch[dataset_index] = epoch
                dataset_index = dataset_index + 1
        else:
            if best_dice == None or test_measure_mDice > best_dice:
                save_path = "./Trained models/" + logs_dir + "/best_model_" + args.dataset + ".pt"
                print("Saving...")
                save_model(model.state_dict(), optimizer.state_dict(), epoch, loss, save_path)
                best_dice = test_measure_mDice
                best_iou = test_measure_mIou
                best_epoch = epoch

        if epoch == args.epochs:
            save_path = "./Trained models/" + logs_dir + "/last_model_" + args.dataset + ".pt"
            save_model(model.state_dict(), optimizer.state_dict(), epoch, loss, save_path)
    if args.dataset == 'all':
        writer_index = 0
        for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            test_writer.write(
                "\rDataSet: {}:\tEpoch: {}\tBest mDice: {:.6f}\tBest mIou: {:.6f}\n{}".format(
                    _data_name,
                    all_best_epoch[writer_index],
                    all_best_dice[writer_index],
                    all_best_iou[writer_index],
                    args.idea
                ))
            writer_index = writer_index + 1
    else:
        test_writer.write(
            "\rBEST RESULT:\tEpoch: {}\tBest mDice: {:.6f}\tBest mIou: {:.6f}\n{}".format(
                best_epoch,
                best_dice,
                best_iou,
                args.idea
            ))
    writer.close()
    train_writer.close()
    test_writer.close()


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    resolution = args.resolution

    if args.dataset == "Kvasir":
        if args.isma:
            train_images = args.root + "/medaugment-" + str(args.ma_num) + "/images/*"
            train_input_paths = sorted(glob.glob(train_images))
            train_masks = args.root + "/medaugment-" + str(args.ma_num) + "/masks/*"
            train_target_paths = sorted(glob.glob(train_masks))
        else:
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
            args.dataset, args.isma, args.resolution, train_input_paths, train_target_paths, val_input_paths, val_target_paths, test_input_paths,
            test_target_paths, batch_size=args.batch_size
        )
    elif args.dataset == "CVC":
        train_images = args.root + "/train/images/*"
        train_input_paths = sorted(glob.glob(train_images))
        train_masks = args.root + "/train/masks/*"
        train_target_paths = sorted(glob.glob(train_masks))
        test_images = args.root + "/test/images/*"
        test_input_paths = sorted(glob.glob(test_images))
        test_masks = args.root + "/test/masks/*"
        test_target_paths = sorted(glob.glob(test_masks))
        train_dataloader, test_dataloader, val_dataloader = dataloaders.get_dataloaders(
            args.dataset, False, args.resolution, train_input_paths, train_target_paths, None, None,
            test_input_paths,
            test_target_paths, args.batch_size
        )
        val_dataloader = None
    elif args.dataset == "all":
        train_images = args.root + "/TrainDataset/images/*"
        train_input_paths = sorted(glob.glob(train_images))
        train_masks = args.root + "/TrainDataset/masks/*"
        train_target_paths = sorted(glob.glob(train_masks))
        train_dataloader, test_dataloader, val_dataloader = dataloaders.get_dataloaders(
            args.dataset, False, args.resolution, train_input_paths, train_target_paths, batch_size=args.batch_size
        )

    Dice_loss = losses.SoftDiceLoss().cuda()
    BCE_loss = nn.BCEWithLogitsLoss().cuda()
    T_Loss = losses.T_Loss(dim=3)
    Iou_loss = losses.IoULoss().cuda()

    model = bestmodel.XXFormer(size=args.resolution, drop_rate=args.da)




    isaugsub = args.isaugsub

    if args.mixup > 0:
        mixup = True
    else:
        mixup = False
    logs_dir = args.logs_dir
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    writer = SummaryWriter(log_dir=logs_dir)

    train_writer = open(args.logs_dir + '/train_result', 'a', encoding='utf-8')
    test_writer = open(args.logs_dir + '/test_result', 'a', encoding='utf-8')

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=args.wd)



    if args.lrs == "true":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)
    else:
        scheduler = None
        warmup_scheduler = None

    return (
        device,
        resolution,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        Dice_loss,
        BCE_loss,
        T_Loss,
        Iou_loss,
        model,
        optimizer,
        scheduler,
        warmup_scheduler,
        logs_dir,
        mixup,
        isaugsub,
        writer,
        train_writer,
        test_writer,
    )


def main():
    args = arguments.get_tarin_args()
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train(args)


if __name__ == "__main__":
    main()
