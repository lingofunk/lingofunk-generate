import sys
import argparse
from flask import Flask, request
import tensorflow as tf
import numpy as np

sys.path.insert(0, '..')

from lingofunc_generate.model_restore_utils import restore_model_from_data
from lingofunc_generate.utils import get_model_name
from lingofunc_generate.utils import log as _log
from lingofunc_generate.constants import TEXT_STYLES


DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND = 'Required model not found. Sorry'
NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING = 2


app = Flask(__name__)

models = dict.fromkeys(TEXT_STYLES, None)
temperature = None
graph = None


def log(text):
    _log(text, prefix='Server: ')


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/generate', methods=['GET'])
def generate_text():
    log('generate text for style "{}"'.format(request.args['style']))

    text_style = request.args['style']

    global graph
    global temperature

    with graph.as_default():
        if models[text_style] is not None:
            text = models[text_style].generate(1, temperature=temperature, return_as_list=True)[0]
        else:
            text = DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND

    return text


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--port', type=int, required=False, default=8001,
        help='The port to listen on (default is 8001).')

    parser.add_argument(
        '--seed', type=int, required=False, default=11221963)

    parser.add_argument(
        '--temperature', type=float, required=False, default=0.5)

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
