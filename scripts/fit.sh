#!/usr/bin/env bash

python ./lingofunk_generate/fit.py \
    --nrows 50000 \
    --max-texts-per-label 3000 \
    --data-path './data/reviews.csv' \
    --text-col 'text' \
    --label-col 'stars' \
    --labels-positive 5 --labels-positive 4 \
    --labels-negative 1 --labels-negative 2 \
    --labels-neutral 3 \
    --gen-epochs 1 \
    --num-epochs 500

# --new-model
# --word-level
# --max-length 10
# --nrows 50000 \
# --max-texts-per-label 2000 \
