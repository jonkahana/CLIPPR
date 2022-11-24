#!/bin/bash

False=''
True='True'


python -u train__distr_match.py \
--exp-name=swd__clip_vis__0_labels__gaussian__prompt_3 \
--dataset=utk \
--device=cuda \
--assumed-dist-params="{'dist_type':'gaussian', 'mean':33, 'std':20, 'min':0, 'max':100}" \
--epochs=70 \
--prompt="f'this person was born {c} years ago.'" \
--alpha=1 \
--scheduler-gamma=0.3 \
--scheduler-epochs=10 \
--batch-size=128 \


python -u eval.py \
--exp-name=distr_match__swd__clip_vis__0_labels__gaussian__prompt_3 \
--dataset=utk \
--model=clip_vis \
--device=cuda \
--batch-size=256 \
--epoch=best \



