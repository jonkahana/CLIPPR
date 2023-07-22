#!/bin/bash

False=''
True='True'


python -u train__distr_match.py \
--exp-name=clippr__costum_prior \
--dataset=stanford_cars \
--device=cuda \
--assumed-dist-params="{'dist_type':'costum', 'example': [22]*30 + [17]*20 + [19]*10 + [20]*10 + [21]*5 + [18]*5 + [3]*3 + [4]*3 + [8]*2 + [11]*2 + [1]*1 + [12]*1 + [16]*1 + [10]*1 + [9]*1 + [7]*1}" \
--epochs=70 \
--alpha=1 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--weight-decay=0.0003 \
--stage1-length=10 \
--stage2-length=10 \
--batch-size=128

python -u eval.py \
--exp-name=clippr__costum_prior \
--dataset=stanford_cars \
--model=clip_vis \
--device=cuda \
--batch-size=256 \
--epoch=best \
