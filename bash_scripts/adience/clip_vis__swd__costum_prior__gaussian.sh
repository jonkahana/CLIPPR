#!/bin/bash

False=''
True='True'


python -u train__distr_match.py \
--exp-name=swd__clip_vis__0_labels__gaussian \
--dataset=adience \
--device=cuda \
--assumed-dist-params="{'dist_type':'costum', 'example': [28]*24 + [1]*12 + [40]*11 + [5]*10 + [10]*10 + [18]*8 + [80]*4 + [50]*4 + [35]*1 + [13]*1 + [22]*1 + [34]*1 + [23]*1 + [45]*1 + [30]*1 + [55]*1 + [36]*1 + [57]*1 + [3]*1 + [29]*1 + [43]*1 + [58]*1 + [2]*1 + [32]*1 + [56]*1 + [16]*1 + [42]*1 + [46]*1}" \
--epochs=70 \
--alpha=1 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=distr_match__swd__clip_vis__0_labels__gaussian \
--dataset=adience \
--model=clip_vis \
--device=cuda \
--batch-size=256 \
--epoch=best \



