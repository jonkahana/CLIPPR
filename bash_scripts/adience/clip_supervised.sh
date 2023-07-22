#!/bin/bash

False=''
True='True'



python -u train.py \
--exp-name=clip_supervised \
--dataset=adience \
--device=cuda \
--epochs=70 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--weight-decay=0.0003 \
--stage1-length=10 \
--stage2-length=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=clip_supervised \
--dataset=adience \
--model=clip_vis \
--device=cuda \
--batch-size=256 \
--epoch=best \



