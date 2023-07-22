#!/bin/bash

False=''
True='True'



python -u eval.py \
--exp-name=clip_zero \
--dataset=imagenet \
--model=clip_zero \
--device=cuda \
--regression=$False \
--batch-size=256 \
--epoch=best \



