#!/bin/bash

False=''
True='True'



python -u train.py \
--exp-name=clip_supervised \
--dataset=cifar10 \
--device=cuda \
--regression=$False \
--epochs=70 \
--scheduler-gamma=0.3 \
--weight-decay=0.0003 \
--stage1-length=10 \
--stage2-length=10 \
--scheduler-epochs=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=clip_supervised \
--dataset=cifar10 \
--model=clip_vis \
--device=cuda \
--regression=$False \
--batch-size=256 \
--epoch=best \



