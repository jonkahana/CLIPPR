#!/bin/bash

False=''
True='True'


python -u train__distr_match.py \
--exp-name=clippr__gaussian__mean_33_scale_10 \
--dataset=utk \
--device=cuda \
--assumed-dist-params="{'dist_type':'gaussian', 'mean':33, 'std':10, 'min':0, 'max':100}" \
--epochs=70 \
--alpha=1 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--weight-decay=0.0003 \
--stage1-length=10 \
--stage2-length=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=clippr__gaussian__mean_33_scale_10 \
--dataset=utk \
--model=clip_vis \
--device=cuda \
--batch-size=256 \
--epoch=best \



