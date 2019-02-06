#!/usr/bin/env bash

python ./lingofunc_generate/fit.py \
    --debug --nrows 100 \
    --data-path './data/review.csv' \
    --text-col 'text' \
    --label-col 'stars' \
    --labels-positive 5 --labels-positive 4 \
    --labels-negative 1 --labels-negative 2 \
    --labels-neutral 3 \
#    --word-level \
    --new-model \
    --num-epochs 500
