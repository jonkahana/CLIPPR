import time
import os

import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

import clip

from dataset import *
from model import *
from utils import *

save_dir = 'weights'
os.makedirs(save_dir, exist_ok=True)
best_test_loss = np.inf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        help="choices are ['utk', 'stanford_cars', 'adience', 'cifar10']")
    parser.add_argument('--regression', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    args = parser.parse_args()
    args = DictX(vars(args))


    reg_str = 'classification' if not args.regression else f'regression'
    args.exp_name = f'{args.exp_name}__{reg_str}'
    device = args.device if torch.cuda.is_available() else 'cpu'

    transform = None

    if args.dataset == 'utk':
        train_set = UTK_Faces(target='age', split='train')
        test_set = UTK_Faces(target='age', split='test')
        prompt = PROMPTS['utk']
    elif args.dataset == 'adience':
        train_set = Adience(split='train')
        test_set = Adience(split='test')
        prompt = PROMPTS['adience']
    elif args.dataset == 'stanford_cars':
        train_set = Stanford_Cars(data_name='stanford_cars', label_name='year', split='train')
        test_set = Stanford_Cars(data_name='stanford_cars', label_name='year', split='test')
        prompt = PROMPTS['stanford_cars']
    elif args.dataset == 'cifar10':
        train_set = CIFAR10(split='train')
        test_set = CIFAR10(split='test')
        prompt = PROMPTS['cifar10']
    else:
        raise ValueError(f'dataset = {args.dataset}, is not supported at the moment')

    classes = None if args.regression else train_set.all_labels_names

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers)

    model = CLIP_Visual(classes=classes, device=device).to(device)
    model_parameters = model.classifier.parameters()

    optimizer = optim.Adam(model_parameters, lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=len(train_loader) * args.scheduler_epochs,
                       gamma=args.scheduler_gamma)
    crit = nn.L1Loss() if args.regression else nn.CrossEntropyLoss()

    exp_dir = os.path.join(save_dir, args.dataset, args.exp_name)
    if os.path.exists(exp_dir) and 'debug' not in args.exp_name:
        raise ValueError(f'Preventing delete of previous experiment!')
    os.makedirs(exp_dir, exist_ok=True)
    save_experiment_hyper_params(args, exp_dir)
    tens_dir = join(exp_dir, 'tensorboard')
    os.makedirs(tens_dir, exist_ok=True)
    writer = SummaryWriter(tens_dir)

    for epoch in range(args.epochs):
        model.train()
        total_loss, avg_loss = 0.0, 0.0
        correct, total_el = 0.0, 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if len(output.shape) == 1 and not args.regression:
                output = output.view(-1, 1)
            loss = crit(output, target)
            writer.add_scalar('train/Batch_Loss', loss.item(), global_step=epoch * len(train_loader) + batch_idx)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if not args.regression:
                cls_pred = output.argmax(dim=1, keepdim=True)
                correct += cls_pred.eq(target.view_as(cls_pred)).sum().item()
                total_el += data.shape[0]
            total_loss += loss.item()

        # region Log Epoch
        train_loss = total_loss / len(train_loader)
        if not args.regression:
            train_acc = 100. * correct / total_el
            writer.add_scalar('train/Epoch_Accuracy', train_acc, global_step=epoch)
            print_str = f'Epoch {epoch} || Train Accuracy: {train_acc} || Train Loss {train_loss}\n'
        else:
            train_acc = None
            print_str = f'Epoch {epoch} || Train MAE {train_loss}\n'
        writer.add_scalar('train/Epoch_Loss', train_loss, global_step=epoch)
        for param_group in optimizer.param_groups:
            writer.add_scalar('train/LR', param_group['lr'], global_step=epoch)
            break

        print(print_str)
        # endregion log epoch

        # region Eval Epoch
        if epoch % args.eval_every == 0:
            model.eval()
            test_total_loss = 0.0
            test_correct, test_total_el = 0.0, 0.0
            for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    if len(output.shape) == 1 and not args.regression:
                        output = output.view(-1, 1)
                    loss = crit(output, target)
                if not args.regression:
                    cls_pred = output.argmax(dim=1, keepdim=True)
                    test_correct += cls_pred.eq(target.view_as(cls_pred)).sum().item()
                test_total_loss += loss.item()
                test_total_el += data.shape[0]

            # region Log Evaluation + Save Best Weights

            test_loss = test_total_loss / len(test_loader)
            if not args.regression:
                test_acc = 100. * test_correct / test_total_el
                writer.add_scalar('test/Epoch_Accuracy', test_acc, global_step=epoch)
                print_str = f'Epoch {epoch} || Test Accuracy: {test_acc} || Test Loss {test_loss}\n'
            else:
                test_acc = None
                print_str = f'Epoch {epoch} || Test MAE {test_loss}\n'
            writer.add_scalar('test/Epoch_Loss', test_loss, global_step=epoch)
            print(print_str)

            if best_test_loss > test_loss:
                writer.add_scalar('test/best_epoch', epoch, global_step=epoch)
                torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_best.pth'))
                best_test_loss = test_loss

            # endregion Log Evaluation + Save Best Weights

        # endregion Eval Epoch

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_last.pth'))
