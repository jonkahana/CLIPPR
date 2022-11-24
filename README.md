

# Improving Zero-Shot Models with Label Distribution Priors

> [Project Page](https://www.vision.huji.ac.il/clippr)

### CLIPPR

> [Improving Zero-Shot Models with Label Distribution Priors](https://arxiv.org/abs/PAPER_ID) \
> Joanthan Kahana, Niv Cohen, Yedid Hoshen \
> Official PyTorch Implementation

> **Abstract:**  Labeling large image datasets with attributes such as fa-
cial age or object type is tedious and sometimes infeasible.
Supervised machine learning methods provide a highly ac-
curate solution, but require manual labels which are often
unavailable. Zero-shot models (e.g., CLIP) do not require
manual labels but are not as accurate as supervised ones,
particularly when the attribute is numeric. We propose a
new approach, CLIPPR (CLIP with Priors), which adapts
zero-shot models for regression and classification on unla-
belled datasets. Our method does not use any annotated
images. Instead, we assume a prior over the label distri-
bution in the dataset. We then train an adapter network
on top of CLIP under two competing objectives: i) mini-
mal change of predictions from the original CLIP model ii)
minimal distance between predicted and prior distribution
of labels. Additionally, we present a novel approach for se-
lecting prompts for Vision & Language models using a dis-
tributional prior. Our method is effective and presents a sig-
nificant improvement over the original model. We demon-
strate an improvement of 28% in mean absolute error on the
UTK age regression task. We also present promising results
for classification benchmarks, improving the classification
accuracy on the ImageNet dataset by 2.83%, without using
any labels.

This repository is the official PyTorch implementation of [Improving Zero-Shot Models with Label Distribution Priors](https://arxiv.org/abs/PAPER_ID)

<a href="https://arxiv.org/abs/PAPER_ID" target="_blank"><img src="https://img.shields.io/badge/arXiv-PAPER_ID-b31b1b.svg"></a>

![alt text](https://github.com/jonkahana/CLIPPR/blob/main/imgs/CLIPPR_block_diagram.png?raw=true)

## Usage

### Requirements
![python >= 3.7.3](https://img.shields.io/badge/python->=3.7.3-blue.svg) 
![cuda >= 11.4](https://img.shields.io/badge/CUDA->=11.4-bluegreen.svg) 
![pytorch >= 1.9.0](https://img.shields.io/badge/pytorch->=1.9.0-orange.svg)

* ![numpy >= 1.16.2](https://img.shields.io/badge/numpy->=1.16.2-purple.svg)
* ![pandas >= 1.3.4](https://img.shields.io/badge/pandas->=1.3.4-darkblue.svg)
* ![clip >= 1.0](https://img.shields.io/badge/clip->=1.0-darkgreen.svg)

### Downloading the Datasets

You need to download the datasets first. download each to a separate directory under the same father directory.

**NOTE:** Please update the `DATA_PATH` parameter in `dataset.py` and `scripts/prepare_stanford_cars.py` to the father directory of the dataset.

For the ImageNet dataset please perform the pre-processing script found [here](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).
(ImageNet dataset Coming soon...)

For the Stanford Cars dataset please perform our pre-processing script: `scripts/prepare_stanford_cars.py`.

### Training

We provide training & evaluation scripts for: CLIPPR, Zero-Shot CLIP (evaluation only) and a Supervised adapter on top of CLIP, for each one of the evaluated datasets.  
The scripts can be found in the `bash_scripts` folder sorted by dataset.

**NOTE:** Inside the `bash_scripts\utk` folder you can also find code for our ablation studies.

### Trained Checkpoints and More Coming Soon! 


## Citation
If you find this useful, please cite our paper:
```

```