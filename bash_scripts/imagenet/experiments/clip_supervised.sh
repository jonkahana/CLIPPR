#!/bin/bash

False=''
True='True'



python -u train.py \
--exp-name=clip_supervised \
--dataset=imagenet \
--device=cuda \
--regression=$False \
--epochs=14 \
--scheduler-gamma=0.3 \
--weight-decay=0.0003 \
--stage1-length=2 \
--stage2-length=2 \
--scheduler-epochs=2 \
--batch-size=128 \


python -u eval.py \
--exp-name=clip_supervised \
--dataset=imagenet \
--model=clip_vis \
--device=cuda \
--regression=$False \
--batch-size=256 \
--epoch=best \



