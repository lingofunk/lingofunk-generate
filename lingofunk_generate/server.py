import sys
import os
import argparse
import logging
import time
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

project_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_folder)

from lingofunk_generate.model_restore_utils import restore_model_from_data
from lingofunk_generate.utils import get_model_name
from lingofunk_generate.utils import log as _log
from lingofunk_generate.constants import TEXT_STYLES
from lingofunk_generate.constants import PORT_DEFAULT
from lingofunk_generate.constants import TEMPERATURE_DEFAULT
from lingofunk_generate.constants import DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND
from lingofunk_generate.constants import NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

models = dict.fromkeys(TEXT_STYLES, None)
temperature = None
graph = None


def log(text):
    _log(text, prefix='Server: ')


@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/generate/discrete', methods=['POST'])
def generate_discrete():
    logger.debug('request: {}'.format(request.get_json()))

    data = request.get_json()
    text_style = data.get('style')

    global graph
    global temperature

    with graph.as_default():
        if models[text_style] is not None:
            text = models[text_style].generate(1, temperature=temperature, return_as_list=True)[0]
        else:
            text = DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND

    return jsonify(text=text)


@app.route('/generate/continuous', methods=['POST'])
def generate_continuous():
    logger.debug('request: {}'.format(request.get_json()))

    data = request.get_json()
    text_style_value = data.get('value')

    global graph
    global temperature

    models

    with graph.as_default():
        if models[text_style] is not None:
            text = models[text_style].generate(1, temperature=temperature, return_as_list=True)[0]
        else:
            text = DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND

    return jsonify(text=text)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--port', type=int, required=False, default=PORT_DEFAULT,
        help='The port to listen on')

    parser.add_argument(
        '--seed', type=int, required=False, default=int(time.time()),
        help='Random seed')

    parser.add_argument(
        '--temperature', type=float, required=False, default=TEMPERATURE_DEFAULT,
        help='Low temperature (eg 0.2) makes the model more confident but also more conservative ' + \
             'when generating response. ' + \
             'High temperatures (eg 0.9, values higher than 1.0 also possible) make responses diverse, ' + \
             'but mistakes are also more likely to take place')

    return parser.parse_args()


def _set_seed(seed):
    np.random.seed(seed)


def _set_temperature(t):
    global temperature
    temperature = t


def _load_models():
    log('load models')

    for text_style in TEXT_STYLES:
        model_name = get_model_name(text_style)

        try:
            model = restore_model_from_data(model_name)
        except (IOError, ValueError):
            log('model "{}" not found'.format(model_name))
            model = None

        models[text_style] = model

    global graph
    graph = tf.get_default_graph()


def _test_models():
    log('test models')

    global temperature

    for text_style, model in models.items():
        if model is None:
            continue

        try:
            for _ in range(NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING):
                text = model.generate(1, temperature=temperature, return_as_list=True)[0]
                log('generate sample of style "{}" model\n{}'.format(text_style, text))
        except:
            raise ValueError('Model for style "{}" is invalid'.format(text_style))


def _main():
    args = _parse_args()

    _set_seed(args.seed)
    _set_temperature(args.temperature)
    _load_models()
    _test_models()

    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=True)


if __name__ == '__main__':
    _main()