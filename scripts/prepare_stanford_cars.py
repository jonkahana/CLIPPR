from os.path import join as join
import scipy.io

import cv2
from tqdm import tqdm
# 
from PIL import Image
import glob
from copy import deepcopy
# 
import numpy as np
import pandas as pd
#
from sklearn.model_selection import train_test_split

DATA_PATH = None  # TODO: Replace with Your Datasets Folder Path
if DATA_PATH is None:
    raise ValueError('Please update your DATA_PATH variable!')


dataset_folder = join(DATA_PATH, 'stanford_cars')
save_folder = join(DATA_PATH, 'npz')

np.random.seed(2)


def get_filename(path):
    return path.split('/')[-1]


def get_img(path):
    img = np.array(Image.open(path))
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 2:
        img = np.expand_dims(np.array(img), -1)
        img = np.concatenate([img] * 3, axis=-1)
    return img


def prepare_stanford_cars():

    # region Load Data
    mat = scipy.io.loadmat(join(dataset_folder, 'cars_annos.mat'))
    meta_mat = scipy.io.loadmat(join(dataset_folder, 'devkit', 'cars_meta.mat'))

    class_names = meta_mat['class_names'][0]
    class_names = [x[0] for x in class_names]

    id2years = [int(x.split(' ')[-1]) for x in class_names]
    id2firm = [x.split(' ')[0] for x in class_names]
    id2model = [' '.join(x.split(' ')[1:-1]) for x in class_names]

    class_ids = np.array([mat['annotations'][0][i][-2][0][0] for i in range(len(mat['annotations'][0]))]) - 1
    colnames = ['year', 'firm', 'model', 'bb_0', 'bb_1', 'bb_2', 'bb_3']
    bbox_contents = np.array([[x[0][0] for i, x in enumerate(mat['annotations'][0][i]) if (i >= 1 and i <= 4)] for i in
                              range(len(mat['annotations'][0]))])
    contents = pd.DataFrame(bbox_contents, columns=colnames[-4:])
    contents['year'] = [id2years[x] for x in class_ids]
    contents['firm'] = [id2firm[x] for x in class_ids]
    contents['model'] = [id2model[x] for x in class_ids]
    contents = contents[colnames].values

    paths = [mat['annotations'][0][i][0][0] for i in range(len(mat['annotations'][0]))]
    paths = np.array([join(dataset_folder, x) for x in paths])
    splits = np.array([mat['annotations'][0][i][-1][0][0] for i in range(len(mat['annotations'][0]))])

    imgs = [get_img(path)[np.newaxis] for path in tqdm(paths)]
    imgs = np.concatenate(imgs)
    train_indxs = np.where(splits == 0)[0]
    test_indxs = np.where(splits == 1)[0]
    # endregion Load Data

    # region Save Train Dataset

    train_imgs = imgs[train_indxs]
    train_contents = contents[train_indxs]
    np.savez(join(save_folder, 'stanford_cars_balanced__x256__train.npz'),
             imgs=train_imgs,
             contents=train_contents,
             colnames=colnames
             )

    # endregion Save Train Dataset

    # region Save Test Dataset

    test_imgs = imgs[test_indxs]
    test_contents = contents[test_indxs]
    np.savez(join(save_folder, 'stanford_cars_balanced__x256__test.npz'),
             imgs=test_imgs,
             contents=test_contents,
             colnames=colnames
             )
    # endregion Save Test Dataset


def reduce_dataset(train_test):
    data = dict(np.load(join(save_folder, f'stanford_cars__x256__{train_test}.npz'), allow_pickle=True))
    year_index = np.where(data['colnames'] == 'year')[0][0]
    years = data['contents'][:, year_index]
    reduced_classes = [2012]
    keep_percs = [0.35]
    removed_indxs = []
    for cls, keep_p in zip(reduced_classes, keep_percs):
        removed_perc = 1 - keep_p
        num_removed = np.round(cls * removed_perc).astype(int)
        print(f'Removing {removed_perc * 100}% from Class {cls}. Number of Samples Removed {num_removed}')
        cls_indxs = np.where(years == cls)[0]
        cls_removed_indxs = np.random.choice(cls_indxs, size=int(removed_perc * len(cls_indxs)), replace=False)
        removed_indxs.extend(cls_removed_indxs)

    keep_indxs = list(set(np.arange(len(years))) - set(removed_indxs))
    data_kept_percentage = np.round(100*len(keep_indxs)/len(years), 2)
    print(f'Keeping {data_kept_percentage}% out of the Original Dataset. {len(keep_indxs)} Samples.')

    balanced_imgs = data['imgs'][keep_indxs]
    balanced_contents = data['contents'][keep_indxs]
    balanced_colnames = data['colnames']

    save_name = f'stanford_cars_balanced__x256__{train_test}.npz'

    np.savez(join(save_folder, save_name),
         imgs=balanced_imgs,
         contents=balanced_contents,
         colnames=balanced_colnames
        )


if __name__ == '__main__':
    prepare_stanford_cars()
    reduce_dataset('train')
    reduce_dataset('test')