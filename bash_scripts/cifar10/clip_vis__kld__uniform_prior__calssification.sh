#!/bin/bash

False=''
True='True'


python -u train__distr_match__classification.py \
--exp-name=uniform_prior__clip_vis__0_labels \
--dataset=cifar10 \
--device=cuda \
--assumed-dist-params="{'dist_type':'uniform'}" \
--acc-batches=4 \
--epochs=70 \
--alpha=1 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=distr_match__uniform_prior__clip_vis__0_labels \
--dataset=cifar10 \
--model=clip_vis \
--regression=$False \
--device=cuda \
--batch-size=256 \
--epoch=best \



