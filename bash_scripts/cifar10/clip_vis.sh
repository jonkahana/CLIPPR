#!/bin/bash

False=''
True='True'



python -u train.py \
--exp-name=clip_vis \
--dataset=cifar10 \
--device=cuda \
--regression=$False \
--epochs=70 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=clip_vis \
--dataset=cifar10 \
--model=clip_vis \
--device=cuda \
--regression=$False \
--batch-size=256 \
--epoch=best \



