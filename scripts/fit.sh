#!/usr/bin/env bash

python ./lingofunk_generate/fit.py \
    --nrows 500 \
    --max-texts-per-label 2 \
    --data-path './data/reviews.csv' \
    --text-col 'text' \
    --label-col 'stars' \
    --style-positive --labels-positive 5 --labels-positive 4 \
    --style-negative --labels-negative 1 --labels-negative 2 \
    --style-neutral --labels-neutral 3 \
    --gen-epochs 100 \
    --num-epochs 2 \
    --load-models --models-folder-source './models_char' \
    --models-folder-target './models' \
    --batch-size 200

# --new-model
# --word-level
# --max-length 10

# --nrows 50000
# --max-texts-per-label 2000

# --load-models --models-folder-source './models_char'
# --models-folder-target
# --batch-size
