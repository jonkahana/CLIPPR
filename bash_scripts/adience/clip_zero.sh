#!/bin/bash

False=''
True='True'



python -u eval.py \
--exp-name=clip_zero \
--dataset=adience \
--model=clip_zero \
--device=cuda \
--regression=$True \
--batch-size=256 \
--epoch=best \



