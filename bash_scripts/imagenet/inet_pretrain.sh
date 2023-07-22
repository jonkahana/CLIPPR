#!/bin/bash

False=''
True='True'



python -u train__distr_match__classification.py \
--exp-name=inet_pretrain \
--dataset=imagenet \
--device=cuda \
--inet-pretrain=$False \
--epochs=3 \
--scheduler-gamma=0.3 \
--weight-decay=0.0003 \
--alpha=0. \
--scheduler-epochs=100 \
--batch-size=128 \


python -u eval.py \
--exp-name=inet_pretrain \
--dataset=imagenet \
--model=clip_vis \
--device=cuda \
--regression=$False \
--batch-size=256 \
--epoch=last \



