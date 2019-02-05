import sys
import argparse
from flask import Flask, request
import tensorflow as tf
import numpy as np

sys.path.insert(0, '..')

from lingofunc_generate.model_restore_utils import restore_model_from_data
from lingofunc_generate.utils import get_model_name
from lingofunc_generate.utils import log as _log
from lingofunc_generate.constants import TEXT_CLASS_LABELS


DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND = 'Required model not found. Sorry'
NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING = 2


app = Flask(__name__)

models = dict.fromkeys(TEXT_CLASS_LABELS, None)
temperature = None
graph = None


def log(text):
    _log(text, prefix='Server: ')


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/generate', methods=['GET'])
def generate_text():
    log('generate text for class "{}"'.format(request.args['class']))

    class_label = request.args['class']

    global graph
    global temperature

    if models[class_label] is not None:
        with graph.as_default():
            text = models[class_label].generate(1, temperature=temperature, return_as_list=True)[0]
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

    for class_label in TEXT_CLASS_LABELS:
        model_name = get_model_name(class_label)

        try:
            model = restore_model_from_data(model_name)
        except (IOError, ValueError):
            log('model "{}" not found'.format(model_name))
            model = None

        models[class_label] = model

    global graph
    graph = tf.get_default_graph()


def _test_models():
    log('test models')

    global temperature

    for class_label, model in models.items():
        if model is None:
            continue

        try:
            for _ in range(NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING):
                text = model.generate(1, temperature=temperature, return_as_list=True)[0]
                log('generate sample of class "{}" model\n{}'.format(class_label, text))
        except:
            raise ValueError('Model for class "{}" is invalid'.format(class_label))


def _main():
    args = _parse_args()

    _set_seed(args.seed)
    _set_temperature(args.temperature)
    _load_models()
    _test_models()

    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=True)


if __name__ == '__main__':
    _main()
