import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

from lingofunk_generate.constants import (
    DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND,
    MODELS_FOLDER_PATH,
    NUM_MODELS_TO_CHOOSE_FROM_WHEN_SYNTHESIZE,
    NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING,
    PORT_DEFAULT,
    TEMPERATURE_DEFAULT,
    TEXT_STYLES,
    VALUE_TO_TEXT_STYLE,
)
from lingofunk_generate.model_restore_utils import load_models as _load_models
from lingofunk_generate.utils import log as _log
from textgenrnn.utils import synthesize

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

models = dict.fromkeys(TEXT_STYLES, None)
temperature = None
graph = tf.get_default_graph()


def log(text):
    _log(text, prefix="Server: ")


def get_text_of_style(text_style, temperature):
    global graph

    with graph.as_default():
        if models[text_style] is not None:
            text = models[text_style].generate(
                1, temperature=temperature, return_as_list=True
            )[0]
        else:
            text = DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND

    return text


@app.route("/hello", methods=["GET"])
def hello_world():
    return "Hello World!"


@app.route("/generate/discrete", methods=["POST"])
def generate_discrete():
    logger.debug("request: {}".format(request.get_json()))

    data = request.get_json()
    text_style = data.get("style-name")

    global temperature  # TODO: remove global

    text = get_text_of_style(text_style, temperature)

    return jsonify(text=text)


@app.route("/generate/continuous", methods=["POST"])
def generate_continuous():
    logger.debug("request: {}".format(request.get_json()))

    global temperature

    data = request.get_json()
    text_style_value = data.get("style-value")

    if text_style_value in [-1.0, 0.0, +1.0]:
        text_style = VALUE_TO_TEXT_STYLE[text_style_value]
        text = get_text_of_style(text_style, temperature)

        return jsonify(text=text)

    text_style_value_extreme = (
        +1.0 if text_style_value > 0 else -1.0
    )  # I really don't know how to name it
    text_style_value_neutral = 0.0

    model_extreme = models[VALUE_TO_TEXT_STYLE[text_style_value_extreme]]
    model_neutral = models[VALUE_TO_TEXT_STYLE[text_style_value_neutral]]

    fraction_of_model_extreme = abs(text_style_value)
    num_models_extreme = int(
        np.round(fraction_of_model_extreme * NUM_MODELS_TO_CHOOSE_FROM_WHEN_SYNTHESIZE)
    )
    num_models_neutral = NUM_MODELS_TO_CHOOSE_FROM_WHEN_SYNTHESIZE - num_models_extreme

    assert (
        num_models_extreme > 0
    ), 'Number of "{}" models is less than zero: "{}"'.format(
        VALUE_TO_TEXT_STYLE[text_style_value_extreme], num_models_extreme
    )
    assert (
        num_models_neutral > 0
    ), 'Number of "{}" models is less than zero: "{}"'.format(
        VALUE_TO_TEXT_STYLE[text_style_value_neutral], num_models_neutral
    )

    models_to_generate_text = num_models_extreme * [
        model_extreme
    ] + num_models_neutral * [model_neutral]
    np.random.shuffle(models_to_generate_text)

    with graph.as_default():
        text = synthesize(
            models_to_generate_text, temperature=temperature, return_as_list=True
        )[0]

    return jsonify(text=text)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=PORT_DEFAULT,
        help="The port to listen on",
    )

    parser.add_argument(
        "--seed", type=int, required=False, default=int(time.time()), help="Random seed"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=TEMPERATURE_DEFAULT,
        help="Low temperature (eg 0.2) makes the model more confident but also more conservative "
        + "when generating response. "
        + "High temperatures (eg 0.9, values higher than 1.0 also possible) make responses diverse, "
        + "but mistakes are also more likely to take place",
    )

    parser.add_argument(
        "--models",
        type=str,
        required=False,
        default=None,
        help="Subfolder of folder models to load models (weights, vocabs, configs) from",
    )

    return parser.parse_args()


def _set_seed(seed):
    np.random.seed(seed)


def _set_temperature(t):
    global temperature
    temperature = t


def _test_models():
    log("test models")

    global temperature

    for text_style, model in models.items():
        if model is None:
            continue

        try:
            for _ in range(NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING):
                text = model.generate(1, temperature=temperature, return_as_list=True)[
                    0
                ]
                log('generate sample of style "{}" model\n{}'.format(text_style, text))
        except:
            raise ValueError('Model for style "{}" is invalid'.format(text_style))


def _main():
    args = _parse_args()

    _set_seed(args.seed)
    _set_temperature(args.temperature)
    _load_models(models, args.models, graph)
    _test_models()

    app.run(host="0.0.0.0", port=args.port, debug=True, threaded=True)


if __name__ == "__main__":
    _main()
