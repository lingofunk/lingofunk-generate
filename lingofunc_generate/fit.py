import sys
import os
import argparse
import pandas as pd
from textgenrnn.textgenrnn import textgenrnn

sys.path.insert(0, '..')

from lingofunc_generate.model_restore_utils import move_model_data
from lingofunc_generate.constants import MODELS_FOLDER_PATH, TEXT_CLASS_LABELS
from lingofunc_generate.utils import get_model_name


NROWS_DEBUG = 10


def log(text, prefix='Train: '):
    print(prefix + text)


def _create_models_folder_if_not_exists():
    if not os.path.exists(MODELS_FOLDER_PATH):
        os.makedirs(MODELS_FOLDER_PATH)


def _parse_args():
    log('parse args')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-path',type=str, required=True)
    parser.add_argument(
        '--text-col', type=str, required=True)
    parser.add_argument(
        '--label-col', type=str, required=True)

    parser.add_argument(
        '--word-level', type=bool, required=False, default=True)
    parser.add_argument(
        '--new-model', type=bool, required=False, default=False)

    parser.add_argument(
        '--train-size', type=float, required=False, default=0.9)
    parser.add_argument(
        '--num-epochs', type=int, required=False, default=10)
    parser.add_argument(
        '--max-length', type=int, required=False, default=10)
    parser.add_argument(
        '--max-gen-length', type=int, required=False, default=50)

    for class_label in TEXT_CLASS_LABELS:
        parser.add_argument(
            '--label-values-{}'.format(class_label), action='append', type=int, required=False)

    return parser.parse_args()


def _read_data_csv(data_path, names=None):
    log('read data')

    return pd.read_csv(data_path, names=names, nrows=NROWS_DEBUG)


def _fit_and_save_model(model_name, data, label_values, text_col, label_col, word_level, new_model, train_size, num_epochs, max_length, max_gen_length):
    log('train model ' + model_name)

    texts = data[data[label_col].isin(label_values)][text_col].values
    model = textgenrnn(name=model_name)

    model.train_on_texts(
        texts,
        new_model=new_model,
        word_level=word_level,
        train_size=train_size,
        num_epochs=num_epochs,
        gen_epochs=num_epochs // 3,  # TODO: remove hardcode?
        max_length=max_length,
        max_gen_length=max_gen_length)

    log('save model ' + model_name)

    try:
        move_model_data(
            model_name=model_name,
            source_folder_path=os.path.dirname(os.path.abspath(__file__)))
    except IOError:
        log('error, fail to move model data files to special folder')


def _main():
    _create_models_folder_if_not_exists()

    args = _parse_args()

    data = _read_data_csv(
        args.data_path)
        # names=[args.text_col, args.label_col])

    for class_label in TEXT_CLASS_LABELS:
        label_values = getattr(args, 'label_values_' + class_label)

        if not label_values:
            continue

        model_name = get_model_name(class_label)

        _fit_and_save_model(
            model_name,
            data,
            label_values,
            args.text_col,
            args.label_col,
            args.word_level,
            args.new_model,
            args.train_size,
            args.num_epochs,
            args.max_length,
            args.max_gen_length)

if __name__ == '__main__':
    _main()
