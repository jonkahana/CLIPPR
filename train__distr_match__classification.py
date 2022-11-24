import time
import os
import pandas as pd

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


def get_model_and_parameters(model_type, model_classes, regression, prompt=None,
                             device='cuda', inet_pretrain=False):
    if model_type == 'clip_vis':
        model = CLIP_Visual(classes=model_classes, device=device).to(device)
        params = model.classifier.parameters()
    elif model_type == 'clip_zero':
        if regression:
            raise ValueError(f'Cannot zero-shot with original CLIP in regression')
        model = CLIP_Zero_Shot(classes=model_classes, prompt=prompt, device=device).to(device)
        params = iter([])
    else:
        raise ValueError(f'model = {model_type}, is not supported at the moment')
    return model, params


if __name__ == '__main__':

    # region args
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        help="choices are ['utk', 'stanford_cars', 'adience', 'cifar10']")
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--assumed-dist-params', type=str, default=None)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--inet-pretrain', type=bool, default=False)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    args = parser.parse_args()
    args = DictX(vars(args))

    if args.inet_pretrain:
        assert args.model in ['clip_vis']

    args.exp_name = f'distr_match__{args.exp_name}__classification'
    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.assumed_dist_params is not None:
        args.assumed_dist_params = eval(args.assumed_dist_params)

    # endregion args

    # region Load Data

    if args.dataset == 'utk':
        train_set = UTK_Faces(target='age', split='train')
        test_set = UTK_Faces(target='age', split='test')
        prompt = PROMPTS['utk']
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

    classes = train_set.all_labels_names
    # train_regr_labels = train_set.regr_targets
    train_cls_targets = train_set.cls_targets
    # cls2regr = train_set.cls2regr

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # endregion data

    # region Model & Optimization

    model, model_parameters = get_model_and_parameters(model_type='clip_vis', model_classes=classes, regression=False,
                                                       prompt=prompt, device=device, inet_pretrain=args.inet_pretrain)
    labels_model, _ = get_model_and_parameters(model_type='clip_zero', model_classes=classes,
                                               regression=False, prompt=prompt, device=device, inet_pretrain=False)
    labels_model.eval()

    optimizer = optim.Adam(model_parameters, lr=0.001, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=len(train_loader) * args.scheduler_epochs / args.acc_batches,
                       gamma=args.scheduler_gamma)
    ce_crit = nn.CrossEntropyLoss()
    dist_match_crit = KLD_Loss()

    # endregion Model & Optimization

    # region Distribution Matching Loss
    if args.assumed_dist_params is None:
        prior_cls_probs = pd.value_counts(train_cls_targets).sort_index().values / len(train_cls_targets)
        prior_cls_probs = torch.tensor(prior_cls_probs).to(device)
    elif args.assumed_dist_params['dist_type'] == 'costum':
        summed_vals = np.sum(list(args.assumed_dist_params['prior_dist'].values()))
        prior_cls_probs = torch.tensor([args.assumed_dist_params['prior_dist'][x] / summed_vals for x in classes])
        prior_cls_probs = prior_cls_probs.to(device)
    elif args.assumed_dist_params['dist_type'] == 'uniform':
        prior_cls_probs = torch.ones(len(classes)) / len(classes)
        prior_cls_probs = prior_cls_probs.to(device)
    else:
        dist_loss_type = args.assumed_dist_params['dist_type']
        raise ValueError(f'No such supported value of dist_loss as: {dist_loss_type}')

    # endregion Distribution Matching Loss

    # region Prepare Logging

    exp_dir = os.path.join(save_dir, args.dataset, args.exp_name)
    if os.path.exists(exp_dir) and 'debug' not in args.exp_name:
        raise ValueError(
            f'Preventing deletion of previous experiment! To rerun this experiment first delete the folder {exp_dir}')
    os.makedirs(exp_dir, exist_ok=True)
    save_experiment_hyper_params(args, exp_dir)
    tens_dir = join(exp_dir, 'tensorboard')
    os.makedirs(tens_dir, exist_ok=True)
    writer = SummaryWriter(tens_dir)

    # endregion Prepare Logging

    for epoch in range(args.epochs):
        model.train()
        correct, total_el = 0.0, 0.0
        total_loss = 0.0
        b_acc_pred_losses, b_acc_preds_dist = [], []
        for batch_idx, (data, cls_target) in enumerate(tqdm(train_loader)):
            cls_target = cls_target.detach().cpu()
            data = data.to(device)
            with torch.no_grad():
                _, _, clip_cls_pred = labels_model.predict__eval(data)
                clip_cls_pred = clip_cls_pred.flatten().to(device)
            output = model(data)
            preds_loss = ce_crit(output, clip_cls_pred)
            b_acc_pred_losses.append(preds_loss.unsqueeze(0))

            preds_dist = F.softmax(output, dim=-1).mean(dim=0).unsqueeze(0)
            b_acc_preds_dist.append(preds_dist)

            writer.add_scalar('train/Batch_CE_Loss', preds_loss.item(),
                              global_step=epoch * len(train_loader) + batch_idx)

            if batch_idx % args.acc_batches == 0 or batch_idx == len(train_loader) - 1:
                acc_batches_loss = torch.cat(b_acc_pred_losses).mean()

                if args.alpha != 0.:
                    b_acc_preds_dist = torch.cat(b_acc_preds_dist, dim=0).mean(dim=0)
                    dist_match_loss = dist_match_crit(b_acc_preds_dist, prior_cls_probs)
                    dist_match_loss *= args.alpha
                    writer.add_scalar('train/Batch_Prior_Loss', dist_match_loss.item(),
                                      global_step=epoch * len(train_loader) + batch_idx)
                    acc_batches_loss += dist_match_loss

                acc_batches_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += acc_batches_loss.item()
                writer.add_scalar('train/Batch_Loss', acc_batches_loss.item(),
                                  global_step=epoch * len(train_loader) + batch_idx)

                b_acc_pred_losses, b_acc_preds_dist = [], []
                acc_batches_loss = 0.

            cls_pred = output.argmax(dim=1, keepdim=True)
            cls_pred = cls_pred.detach().cpu()
            correct += cls_pred.eq(cls_target.view_as(cls_pred)).sum().item()
            total_el += data.shape[0]

        # region Log Epoch

        train_acc = correct / total_el
        print_str = f'Epoch {epoch} || Train Acc. {train_acc}\n'
        writer.add_scalar('train/Epoch_MAE', train_acc, global_step=epoch)

        train_loss = total_loss * args.acc_batches / len(train_loader)
        writer.add_scalar('train/Epoch_Loss', train_loss, global_step=epoch)

        for param_group in optimizer.param_groups:
            writer.add_scalar('train/LR', param_group['lr'], global_step=epoch)
            break

        print(print_str)

        # endregion Log Epoch

        # region Evaluate Epoch

        if epoch % args.eval_every == 0:
            model.eval()
            correct, total_el = 0.0, 0.0
            test_total_loss, test_prior_loss, test_ce_loss = 0.0, 0.0, 0.0
            b_accs = []
            b_acc_pred_losses, b_acc_preds_dist = [], []
            for batch_idx, (data, cls_target) in enumerate(tqdm(test_loader)):
                cls_target = cls_target.detach().cpu()
                data = data.to(device)

                with torch.no_grad():
                    _, _, clip_cls_pred = labels_model.predict__eval(data)
                    clip_cls_pred = clip_cls_pred.flatten().to(device)
                    output = model(data).detach()
                    preds_loss = ce_crit(output, clip_cls_pred)
                    b_acc_pred_losses.append(preds_loss.unsqueeze(0))

                    preds_dist = F.softmax(output, dim=-1).mean(dim=0).unsqueeze(0)
                    b_acc_preds_dist.append(preds_dist)

                    if batch_idx % args.acc_batches == 0 or batch_idx == len(test_loader) - 1:
                        acc_batches_loss = torch.cat(b_acc_pred_losses).mean()

                        if args.alpha != 0.:
                            b_acc_preds_dist = torch.cat(b_acc_preds_dist, dim=0).mean(dim=0)
                            dist_match_loss = dist_match_crit(b_acc_preds_dist, prior_cls_probs)
                            dist_match_loss *= args.alpha
                            writer.add_scalar('train/Batch_Prior_Loss', dist_match_loss.item(),
                                              global_step=epoch * len(test_loader) + batch_idx)
                            test_prior_loss += dist_match_loss.item()
                            acc_batches_loss += dist_match_loss

                        writer.add_scalar('train/Batch_Loss', acc_batches_loss.item(),
                                          global_step=epoch * len(train_loader) + batch_idx)
                        test_total_loss += acc_batches_loss.item()
                        b_acc_pred_losses, b_acc_preds_dist = [], []
                        acc_batches_loss = 0.

                    cls_pred = output.argmax(dim=1, keepdim=True)
                    cls_pred = cls_pred.detach().cpu()
                    correct += cls_pred.eq(cls_target.view_as(cls_pred)).sum().item()
                    total_el += data.shape[0]

                    test_ce_loss += preds_loss.item()

            # region Log Evaluation + Save Best Weights

            test_loss = test_total_loss * args.acc_batches / len(test_loader)
            test_ce_loss = test_ce_loss / len(test_loader)
            test_acc = correct / total_el
            print_str = f'Epoch {epoch} || Test ACC {test_acc}\n'
            writer.add_scalar('test/Epoch_Loss', test_loss, global_step=epoch)
            test_prior_loss = test_prior_loss * args.acc_batches / len(test_loader)
            writer.add_scalar('test/Epoch_Prior_Loss', test_prior_loss, global_step=epoch)
            writer.add_scalar('test/Epoch_CE_Loss', test_ce_loss, global_step=epoch)
            writer.add_scalar('test/Epoch_Accuracy', test_acc, global_step=epoch)
            print(print_str)
            if best_test_loss > test_loss:
                writer.add_scalar('test/best_epoch', epoch, global_step=epoch)
                torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_best.pth'))
                best_test_loss = test_loss

            # endregion Log Evaluation + Save Best Weights

        # endregion Evaluate Epoch

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_last.pth'))
