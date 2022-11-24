import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

import argparse

from dataset import *
from model import *
from utils import *

save_dir = 'weights'



def load_model(args):
    if args.model == 'clip_vis':
        model = CLIP_Visual(classes=classes, device=device).to(device)
    elif args.model == 'clip_zero':
        model = CLIP_Zero_Shot(classes=classes, prompt=prompt, device=device).to(device)
    else:
        raise ValueError(f'model = {args.model}, is not supported at the moment')

    if args.model != 'clip_zero':
        model.load_state_dict(
            torch.load(os.path.join(save_dir, args.dataset, args.exp_name, f'epoch_{args.epoch}.pth')))
    else:
        os.makedirs(os.path.join(save_dir, args.dataset, args.exp_name), exist_ok=True)
    model.eval()
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, help="choices are ['utk', 'np', 'carpk', 'mobile_phones']")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--inet-pretrain', type=bool, default=False)
    parser.add_argument('--regression', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epoch', type=str, default='best')
    args = parser.parse_args()
    args = DictX(vars(args))

    reg_str = 'classification' if not args.regression else f'regression'
    args.exp_name = f'{args.exp_name}__{reg_str}'
    print(f'Testing pre-training of {args.exp_name} Over {args.dataset} By {reg_str}')
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

    is_classes = not args.regression or args.model == 'clip_zero'
    classes = train_set.all_labels_names if is_classes else None
    cls2regr = train_set.cls2regr if is_classes else None

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers)

    model = load_model(args)

    crit = nn.L1Loss() if args.regression else nn.CrossEntropyLoss()

    test_total_loss, test_total_mae = 0.0, 0.0
    test_correct, test_total_el = 0.0, 0.0
    all_cls_targets, all_cls_preds, all_rgr_targets, all_rgr_preds = [], [], [], []
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
            if len(output.shape) == 1 and not args.regression:
                output = output.view(-1, 1)
        if args.regression:
            if args.model == 'clip_zero':
                cls_pred = output.argmax(dim=1, keepdim=True).detach().cpu().numpy().flatten()
                output = torch.tensor([cls2regr[x] for x in cls_pred]).float().to(device)
            loss = crit(output, target)
            test_total_loss += loss.item()
        else:
            cls_pred = output.argmax(dim=1, keepdim=True)
            np_cls_pred = cls_pred.detach().cpu().numpy().flatten()
            all_cls_preds.extend(list(np_cls_pred))
            all_cls_targets.extend(list(target.detach().cpu().numpy()))
            test_correct += cls_pred.eq(target.view_as(cls_pred)).sum().item()
            test_total_el += data.shape[0]


    if args.regression:
        test_loss = test_total_loss / len(test_loader)
        print_str = f'Test MAE (Loss): {test_loss}'
    else:
        test_acc = 100. * test_correct / test_total_el
        test_mae = test_total_mae / len(test_loader)
        print_str = f'Test Accuracy: {test_acc}'

    print(print_str)
    if args.epoch == 'best':
        with open(os.path.join(save_dir, args.dataset, args.exp_name, f'{args.exp_name}__Results.txt'), 'a+') as f:
            for line in print_str.split('||'):
                f.write(line + '\n\n')
    else:
        with open(os.path.join(save_dir, args.dataset, args.exp_name, f'{args.exp_name}__epoch_{args.epoch}__Results.txt'), 'a+') as f:
            for line in print_str.split('||'):
                f.write(line + '\n\n')

    for line in print_str.split('||'):
        key = line.split(': ')[0]
        val = line.split(': ')[1]
