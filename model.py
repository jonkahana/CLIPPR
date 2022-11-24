import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torch.nn import init

import torchvision.models
import clip

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class CLIP_Zero_Shot(nn.Module):

    def __init__(self, classes, prompt=None, device='cuda'):
        super(CLIP_Zero_Shot, self).__init__()
        self.device = device
        self.classes = classes
        self.prompt = prompt
        model, preprocess = clip.load('ViT-B/32', device)
        self.model, self.preprocess = model, preprocess

    def clip_feature_extractor(self, x, prompt):
        text_inputs = torch.cat([clip.tokenize(eval(prompt)) for c in self.classes]).to(self.device)
        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

    def forward(self, x, prompt=None):
        prompt = prompt if prompt is not None else self.prompt
        with torch.no_grad():
            image_features, text_features = self.clip_feature_extractor(x, prompt=prompt)
            image_features = image_features.detach()
            text_features = text_features.detach()

        logits = image_features @ text_features.T

        return logits

    def predict__eval(self, x, prompt=None, cls_func=lambda x: x, return_features=False):

        prompt = prompt if prompt is not None else self.prompt

        with torch.no_grad():
            text_inputs = []
            for orig_cls in self.classes:
                c = cls_func(orig_cls)
                text_inputs.append(clip.tokenize(eval(prompt)))

            text_inputs = torch.cat(text_inputs).to(self.device)
            image_features = self.model.encode_image(x)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()
            text_features = text_features.detach()

            logits = image_features @ text_features.T

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1, keepdim=True)
            if return_features:
                return logits, probs, preds, image_features, text_features
            return logits, probs, preds


class CLIP_Visual(nn.Module):

    def __init__(self, classes, device='cuda'):
        super(CLIP_Visual, self).__init__()
        self.device = device
        self.backbone = 'ViT-B/16'
        self.classes = classes
        self.out_size = 1 if classes is None else len(classes)
        model, preprocess = clip.load(self.backbone, device)
        self.model, self.preprocess = model.visual.float(), preprocess
        backbone_out_size = 512
        self.classifier = nn.Sequential(nn.Linear(backbone_out_size, 128), nn.ReLU(),
                                        nn.Linear(128, 128), nn.ReLU(),
                                        nn.Linear(128, self.out_size, bias=False)).to(device)
        # self.classifier = nn.Linear(backbone_out_size, self.out_size, bias=True).to(device)
        # torch.nn.init.xavier_uniform(self.classifier.weight)
        # self.classifier.bias.data.fill_(0.01)

    def clip_feature_extractor(self, x):
        img_repr = self.model(x)
        img_repr = F.normalize(img_repr, dim=-1)
        return img_repr

    def forward(self, x):
        with torch.no_grad():
            clip_repr = self.clip_feature_extractor(x)
            clip_repr = clip_repr.float().detach()
        output = self.classifier(clip_repr)
        if self.out_size == 1:
            output = output.squeeze(-1)
        return output

    def predict__eval(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1, keepdim=True)
            return logits, probs, preds


class SWD_Loss(nn.Module):

    def __init__(self, num_proj=20, device='cuda'):
        super(SWD_Loss, self).__init__()
        self.device = device
        self.num_proj = num_proj

    def forward(self, pred_codes, target_codes):

        # pred_codes, target_codes = pred_codes.to(device), target_codes.to(device)

        if len(pred_codes.shape) == 1:
            assert self.num_proj <= 0
            pred_projs = pred_codes.view(1, -1)
            target_projs = target_codes.view(1, -1)
        elif pred_codes.shape[1] == 1:
            assert self.num_proj <= 0
            pred_projs = pred_codes.T
            target_projs = target_codes.T
        else:
            projs_matrix = torch.rand((self.num_proj, pred_codes.shape[1]), requires_grad=False).to(self.device)

            pred_projs = projs_matrix @ pred_codes.T  # num_proj x num_pred
            target_projs = projs_matrix @ target_codes.T  # num_proj x num_target

        swd_pred_projs, _ = torch.sort(pred_projs, dim=-1)  # num_proj x num_pred
        swd_target_projs, _ = torch.sort(target_projs, dim=-1)  # num_proj x num_target

        if swd_pred_projs.shape[1] != swd_target_projs.shape[1]:
            # if num_target != num_pred sample one of them (after sorting!!)
            if swd_pred_projs.shape[1] > swd_target_projs.shape[1]:
                taken_indxs = np.linspace(start=0, stop=swd_pred_projs.shape[1] - 1,
                                          num=swd_target_projs.shape[1]).astype(int)
                swd_pred_projs = swd_pred_projs[:, taken_indxs]
            elif swd_pred_projs.shape[1] < swd_target_projs.shape[1]:
                taken_indxs = np.linspace(start=0, stop=swd_target_projs.shape[1] - 1,
                                          num=swd_pred_projs.shape[1]).astype(int)
                swd_target_projs = swd_target_projs[:, taken_indxs]

        swd = torch.mean(torch.mean(torch.abs(swd_pred_projs - swd_target_projs), dim=1), dim=0)

        return swd


class KLD_Loss(nn.Module):

    def __init__(self):
        super(KLD_Loss, self).__init__()

    def forward(self, pred, target):
        kld_pointwise = target * torch.log(target / pred)
        return kld_pointwise.sum()
