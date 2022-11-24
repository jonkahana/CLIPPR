import glob
from copy import deepcopy
import os
from os.path import join
import PIL
from PIL import Image
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import resize as vis_resize
from tqdm import tqdm

UTK_ATTRS = ['age', 'gender', 'race']

PROMPTS = {'utk': "f'a person of age {c}.'",
           'adience': "f'age {c}'",
           'smallnorb':
               {'azimuth': "f'an object facing azimuth {c}'"},
           'stanford_cars': "f'a car from {c}'",
           'cifar10': "f'a photo of a {c}'",
           }

CIFAR10_CLASS_LABELS = ["airplane",
                        "automobile",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck"]

DATA_PATH = None  # TODO: Replace with Your Datasets Folder Path

if DATA_PATH is None:
    raise ValueError('Please update your DATA_PATH variable!')


def default_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def arr2arr(arr1, arr2):
    a2a = {}
    for v1, v2 in zip(arr1, arr2):
        if v1 in a2a.keys():
            if a2a[v1] != v2:
                raise ValueError(f'Prompt targets and Regression targets are not synced')
        else:
            a2a[v1] = v2
    return a2a


class UTK_Faces(Dataset):

    def __init__(self, target, split='all',
                 data_path=join(DATA_PATH, 'UTKFace'),
                 ):
        super(UTK_Faces, self).__init__()

        self.split = split
        self.img_paths = np.array(glob.glob(join(data_path, '*')))
        self.filenames = np.array([x.split('/')[-1] for x in self.img_paths])

        target_place = UTK_ATTRS.index(target)
        self.regr_targets = np.array([int(x.split('_')[target_place]) for x in self.filenames]).astype(int)
        self.regr_targets[self.regr_targets >= 100] = 100
        self.prompt_targets = self.regr_targets.astype(str)
        self.prompt_targets[self.prompt_targets == '100'] = '100+'

        self.prompt2regr = arr2arr(self.prompt_targets, self.regr_targets)
        self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}
        for i, x in enumerate(sorted(np.unique(self.prompt_targets))):
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = self.prompt2regr[x]
        self.all_labels_names = np.array(list(self.prompt2cls.keys()))
        self.cls_targets = np.array([self.prompt2cls[x] for x in self.prompt_targets])

        # filter by split
        self.split_indices = self._get_split_indices()
        self.img_paths = self.img_paths[self.split_indices]
        self.filenames = self.filenames[self.split_indices]
        self.regr_targets = self.regr_targets[self.split_indices]
        self.prompt_targets = self.prompt_targets[self.split_indices]
        self.cls_targets = self.cls_targets[self.split_indices]

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = self.transform(default_loader(self.img_paths[i]))
        regr_tgt = self.regr_targets[i]
        return img, regr_tgt

    def _get_split_indices(self):
        np.random.seed(0)
        all_indxs = list(np.arange(len(self.img_paths)))
        if self.split == 'all':
            return all_indxs
        train_indxs, test_indxs = train_test_split(all_indxs, test_size=0.25, stratify=self.regr_targets)
        if self.split == 'train':
            return train_indxs
        elif self.split == 'test':
            return test_indxs
        else:
            raise ValueError(f'No such split value as {self.split}')


class transform_NumpytoPIL(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, img: torch.Tensor):
        """
        Args:
            img (torch.Tensor): Tensor image to be converted to numpy.array

        Returns:
            img (numpy.array): numpy image.
        """
        if np.max(img) <= 1:
            img = (img * 255.).astype(np.uint8)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = np.tile(img, (1, 1, 3))
        return PIL.Image.fromarray(img)


class Stanford_Cars(Dataset):

    def __init__(self, data_name, label_name, split='train',
                 data_path=join(DATA_PATH, 'npz')
                 ):
        super(Stanford_Cars, self).__init__()

        self.data_name = data_name
        if split == 'train':
            self.data_name = self.data_name.replace('test', 'train')
        elif split == 'test':
            self.data_name = self.data_name.replace('train', 'test')
        else:
            raise ValueError(f'Unknown split request: {split}')
        self.label_name = label_name

        np_path = join(data_path, 'stanford_cars_balanced__x256__train' + '.npz')
        self.imgs, self.regr_targets, self.prompt_targets = self.load_np(np_path)

        self.prompt2regr = arr2arr(self.prompt_targets, self.regr_targets)
        self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}
        for i, x in enumerate(sorted(np.unique(self.prompt_targets))):
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = self.prompt2regr[x]
        self.all_labels_names = np.array(list(self.prompt2cls.keys()))
        self.cls_targets = np.array([self.prompt2cls[x] for x in self.prompt_targets])

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transform_NumpytoPIL(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize_transform
        ])

    def load_np(self, np_path):
        data = dict(np.load(np_path, allow_pickle=True))
        imgs = data['imgs']
        target_values = data['contents'][:, np.where(data['colnames'] == self.label_name)[0][0]]
        regr_target_values, prompt_target_values = self.transform_raw_label(target_values)
        return imgs, regr_target_values, prompt_target_values

    def transform_raw_label(self, target_labels):
        if 'smallnorb' in self.data_name:
            if self.label_name == 'azimuth':
                # starts from a label of 0-18
                degs_per_part = 360. / 18.
                radian_targets = target_labels.astype(float) * degs_per_part * (np.pi / 180.)
                prompt_targets = radian_targets * (180. / np.pi)
                prompt_targets = np.round(prompt_targets, 0)
                prompt_targets = prompt_targets.astype(str)
                regr_targets = prompt_targets.astype(float)  # radian_targets
            else:
                raise ValueError(f'No supported attribute as {self.label_name} for dataset {self.data_name}')
        elif 'cars3d' in self.data_name:
            if self.label_name == 'azimuth':
                # starts from a label of 0-24
                degs_per_part = 360. / 24.
                radian_targets = target_labels.astype(float) * degs_per_part * (np.pi / 180.)
                prompt_targets = radian_targets * (180. / np.pi)
                prompt_targets = np.round(prompt_targets, 0)
                prompt_targets = prompt_targets.astype(str)
                regr_targets = prompt_targets.astype(float)  # radian_targets
        elif 'stanford_cars' in self.data_name:
            if self.label_name == 'year':
                prompt_targets = target_labels.astype(str)
                # make regression labels starts from 1
                regr_targets = (target_labels - 1990).astype(int)
            else:
                raise ValueError(f'No supported attribute as {self.label_name} for dataset {self.data_name}')
        else:
            raise ValueError(f'No supported dataset like {self.data_name}')
        return regr_targets, prompt_targets

    def __len__(self):
        return len(self.regr_targets)

    def __getitem__(self, i):
        img = self.transform(self.imgs[i])
        regr_tgt = self.regr_targets[i]
        return img, regr_tgt



class CIFAR10(Dataset):

    def __init__(self, split='train'):
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        ds = torchvision.datasets.CIFAR10
        coarse = {}
        self.set = ds(root='data', train=split, download=True, transform=None, **coarse)

        self.all_labels_names = np.array(CIFAR10_CLASS_LABELS).astype(str)
        self.cls_targets = self.set.targets
        self.prompt2regr, self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}, {}

        for i, x in enumerate(CIFAR10_CLASS_LABELS):
            self.prompt2regr[x] = 1.
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = 1.

    def __getitem__(self, i):
        img = Image.fromarray(self.set.data[i])
        img = self.transform(img)
        cls_tgt = self.set.targets[i]
        return img, cls_tgt

    def __len__(self):
        return len(self.set.data)


class Adience(Dataset):

    def __init__(self, split='all',
                 data_path=join(DATA_PATH, 'Adience')
                 ):
        super(Adience, self).__init__()

        self.split = split
        self.full_metadata = pd.read_csv(join(data_path, 'metadata.csv'))
        if split == 'train':
            selected_folds = [0, 1, 2, 3]
        else:
            selected_folds = [4]
        self.metadata = self.full_metadata[self.full_metadata['fold'].isin(selected_folds)]
        # self.img_paths = np.array(sorted(self.img_paths))
        self.img_paths = np.array([join(data_path, x) for x in self.metadata['img_path'].values])
        self.regr_targets = self.metadata['age'].values
        self.prompt_targets = self.regr_targets.astype(str)

        self.prompt2cls, self.cls2prompt, self.cls2regr, self.prompt2regr = {}, {}, {}, {}
        for i, x in enumerate(sorted(np.unique(self.full_metadata['age']))):
            self.prompt2cls[str(x)] = i
            self.cls2prompt[i] = str(x)
            self.cls2regr[i] = x
            self.prompt2regr[str(x)] = x
        self.all_labels_names = np.array(list(self.prompt2cls.keys()))
        self.cls_targets = np.array([self.prompt2cls[x] for x in self.prompt_targets])

        # filter by split

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = self.transform(default_loader(self.img_paths[i]))
        regr_tgt = self.regr_targets[i]
        return img, regr_tgt
