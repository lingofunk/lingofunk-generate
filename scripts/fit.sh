#!/usr/bin/env bash

cd lingofunc_generate && \
python fit.py \
    --debug \
    --data-path '../data/review.csv' \
    --text-col 'text' \
    --label-col 'stars' \
    --labels-positive 5 --labels-positive 4 \
    --labels-negative 1 --labels-negative 2 \
    --labels-neutral 3 \
    --num-epochs 1 && \
cd ..
