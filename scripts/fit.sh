#!/usr/bin/env bash

python ./lingofunk_generate/fit.py \
    --data-path './data/reviews.csv' \
    --text-col 'text' \
    --label-col 'stars' \
    --labels-positive 5 --labels-positive 4 \
    --labels-negative 1 --labels-negative 2 \
    --labels-neutral 3 \
    --num-epochs 1 \
    --debug --nrows 20000 \
    --gen-epochs 1
# --word-level
