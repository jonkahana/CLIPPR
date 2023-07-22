#!/bin/bash

False=''
True='True'


python -u train__distr_match__classification.py \
--exp-name=clippr__uniform \
--dataset=imagenet \
--inet-pretrain=$True \
--device=cuda \
--assumed-dist-params="{'dist_type':'uniform'}" \
--acc-batches=300 \
--acc-batches-over-time=$True \
--epochs=14 \
--alpha=100 \
--lr=0.001 \
--scheduler-gamma=0.3 \
--scheduler-epochs=2 \
--weight-decay=0.0003 \
--stage1-length=2 \
--stage2-length=2 \
--batch-size=128 \


python -u eval.py \
--exp-name=clippr__uniform \
--dataset=imagenet \
--model=clip_vis \
--regression=$False \
--device=cuda \
--batch-size=256 \
--epoch=last \



