#!/bin/bash

python -u model.py \
 --epochs 200 \
 --lr 0.01 \
 --window-size 4 \
 --embedding-size 100 \
 --num-tokens 50000 \
 --negative-sampling \
 --k 3