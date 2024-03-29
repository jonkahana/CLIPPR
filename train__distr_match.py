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


def sample_assumed_distribution(dist_parameters, num_samples):
    dist_type = dist_parameters['dist_type']
    if dist_type == 'gaussian':
        distribution = torch.distributions.Normal(loc=dist_parameters['mean'], scale=dist_parameters['std'])
        sample = distribution.sample([num_samples])
        sample = torch.clip(sample, min=dist_parameters['min'], max=dist_parameters['max'])
        return sample
    elif dist_type == 'costum':
        sample = np.random.choice(dist_parameters['example'], size=num_samples, replace=True)
        return torch.tensor(sample)
    else:
        raise ValueError(f'No such supported assumed distribution type as {dist_type}')


if __name__ == '__main__':

    # region args
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        help="choices are ['utk', 'stanford_cars', 'adience', 'cifar10']")
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--assumed-dist-params', type=str, default=None)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--swd-sampled-batch', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--weight-decay', type=float, default=0.0003)
    parser.add_argument('--stage1-length', type=int, default=None)
    parser.add_argument('--stage2-length', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args = DictX(vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.assumed_dist_params is not None:
        args.assumed_dist_params = eval(args.assumed_dist_params)
    args.swd_sampled_batch = args.batch_size if args.swd_sampled_batch is None else args.swd_sampled_batch

    # endregion args

    # region Load Data

    transform = None
    if args.dataset == 'utk':
        train_set = UTK_Faces(target='age', split='train')
        test_set = UTK_Faces(target='age', split='test')
        prompt = PROMPTS['utk']
        if args.prompt is not None:
            prompt = args.prompt
    elif args.dataset == 'adience':
        train_set = Adience(split='train')
        test_set = Adience(split='test')
        prompt = PROMPTS['adience']
    elif args.dataset == 'stanford_cars':
        train_set = Stanford_Cars(data_name='stanford_cars', label_name='year', split='train')
        test_set = Stanford_Cars(data_name='stanford_cars', label_name='year', split='test')
        prompt = PROMPTS['stanford_cars']
    else:
        raise ValueError(f'dataset = {args.dataset}, is not supported at the moment')

    classes = train_set.all_labels_names
    train_regr_labels = train_set.regr_targets
    cls2regr = train_set.cls2regr

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # endregion data

    # region Model & Optimization

    model = CLIP_Visual(classes=None, device=device).to(device)
    model_parameters = model.classifier.parameters()
    labels_model = CLIP_Zero_Shot(classes=classes, prompt=prompt, device=device).to(device)
    labels_model.eval()

    optimizer = optim.Adam(model_parameters, lr=0.001, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=len(train_loader) * args.scheduler_epochs,
                       gamma=args.scheduler_gamma)
    l1_crit = nn.L1Loss()

    # endregion Model & Optimization

    # region Distribution Matching Loss

    dist_match_crit = SWD_Loss(num_proj=0, device=device)

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
        print(f'\n')
        if args.stage1_length is not None and args.stage1_length == epoch:
            print('--------------------- Start Stage 2 ---------------------')
            model.freeze_backbone = False
            optimizer = optim.Adam(model.model.parameters(), lr=0.0000001, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999))
            scheduler = StepLR(optimizer=optimizer,
                               step_size=len(train_loader) * args.scheduler_epochs,
                               gamma=args.scheduler_gamma)
        if args.stage1_length is not None and args.stage1_length + args.stage2_length == epoch:
            print('--------------------- Start Stage 3 ---------------------')
            model.freeze_backbone = True
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999))
            scheduler = StepLR(optimizer=optimizer,
                               step_size=len(train_loader) * args.scheduler_epochs, gamma=args.scheduler_gamma)

        model.train()
        b_maes = []
        total_loss = 0.0
        for batch_idx, (data, regr_target) in enumerate(tqdm(train_loader)):
            regr_target = regr_target.detach().cpu()
            data = data.to(device)
            with torch.no_grad():
                _, _, cls_pred = labels_model.predict__eval(data)
                regr_pred = torch.tensor([cls2regr[x.item()] for x in cls_pred]).to(device)
            optimizer.zero_grad()
            output = model(data)

            l1_loss = l1_crit(output, regr_pred)

            sampled_regr_labels = sample_assumed_distribution(args.assumed_dist_params,
                                                              args.swd_sampled_batch
                                                              ).to(device).float()
            dist_match_loss = dist_match_crit(output, sampled_regr_labels)
            dist_match_loss *= args.alpha

            loss = l1_loss + dist_match_loss
            writer.add_scalar('train/Batch_Lasso_Loss', l1_loss.item(),
                              global_step=epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/Batch_MLE_Loss', dist_match_loss.item(),
                              global_step=epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/Batch_Loss', loss.item(), global_step=epoch * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()
            scheduler.step()
            b_maes.append(torch.mean(torch.abs(output.detach().cpu() - regr_target)).item())
            total_loss += loss.item()

        # region Log Epoch

        train_mae = np.mean(b_maes)
        print_str = f'Epoch {epoch} || Train MAE {train_mae}\n'
        writer.add_scalar('train/Epoch_MAE', train_mae, global_step=epoch)

        train_loss = total_loss / len(train_loader)
        writer.add_scalar('train/Epoch_Loss', train_loss, global_step=epoch)

        for param_group in optimizer.param_groups:
            writer.add_scalar('train/LR', param_group['lr'], global_step=epoch)
            break

        print(print_str)

        # endregion Log Epoch

        # region Evaluate Epoch

        if epoch % args.eval_every == 0:
            model.eval()
            test_total_loss, test_nll_loss, test_l1_loss = 0.0, 0.0, 0.0
            b_maes = []
            for batch_idx, (data, regr_target) in enumerate(tqdm(test_loader)):
                regr_target = regr_target.detach().cpu()
                data = data.to(device)

                with torch.no_grad():
                    _, _, cls_pred = labels_model.predict__eval(data)
                    regr_pred = torch.tensor([cls2regr[x.item()] for x in cls_pred]).to(device)
                    output = model(data).detach()
                    l1_loss = l1_crit(output, regr_pred)

                    sampled_regr_labels = sample_assumed_distribution(args.assumed_dist_params,
                                                                      args.swd_sampled_batch
                                                                      ).to(device).float()
                    dist_match_loss = dist_match_crit(output, sampled_regr_labels)

                    loss = l1_loss + dist_match_loss
                    dist_match_loss *= args.alpha

                    test_total_loss += loss.item()
                    test_nll_loss += dist_match_loss.item()
                    test_l1_loss += l1_loss.item()
                    b_maes.append(torch.mean(torch.abs(output.detach().cpu() - regr_target)).item())

            # region Log Evaluation + Save Best Weights

            test_loss = test_total_loss / len(test_loader)
            test_nll_loss = test_nll_loss / len(test_loader)
            test_l1_loss = test_l1_loss / len(test_loader)
            test_mae = np.mean(b_maes)
            print_str = f'Epoch {epoch} || Test MAE {test_mae}\n'
            writer.add_scalar('test/Epoch_Loss', test_loss, global_step=epoch)
            writer.add_scalar('test/Epoch_MLE_Loss', test_nll_loss, global_step=epoch)
            writer.add_scalar('test/Epoch_Lasso_Loss', test_l1_loss, global_step=epoch)
            writer.add_scalar('test/Epoch_MAE', test_mae, global_step=epoch)
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
