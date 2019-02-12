#!/usr/bin/env bash

PYTHONPATH=. python -m lingofunk_generate.server \
    --temperature 0.5 \
    --models 'models_char_1epochs_20000texts'
