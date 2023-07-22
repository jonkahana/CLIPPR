#!/bin/bash

False=''
True='True'


python -u train__distr_match__classification.py \
--exp-name=clippr__uniform \
--dataset=cifar10 \
--device=cuda \
--assumed-dist-params="{'dist_type':'uniform'}" \
--acc-batches=4 \
--acc-batches-over-time=$False \
--epochs=70 \
--alpha=100 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--weight-decay=0.0003 \
--stage1-length=10 \
--stage2-length=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=clippr__uniform \
--dataset=cifar10 \
--model=clip_vis \
--regression=$False \
--device=cuda \
--batch-size=256 \
--epoch=best \



