#!/usr/bin/env bash

PYTHONPATH=. python -m lingofunk_generate.server \
    --temperature 0.5 \
    --models 'models_best' \
    --max-gen-length 450
