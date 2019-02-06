import sys
import os
import shutil
import argparse
import pandas as pd
from textgenrnn.textgenrnn import textgenrnn

project_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_folder)

from lingofunk_generate.model_restore_utils import move_model_data
from lingofunk_generate.constants import MODELS_FOLDER_PATH
from lingofunk_generate.constants import TEXT_STYLES
from lingofunk_generate.constants import DATA_FILE_PATH_DEFAULT
from lingofunk_generate.constants import MAX_LENGTH_DEFAULT_CHAR_LEVEL, MAX_LENGTH_DEFAULT_WORD_LEVEL
from lingofunk_generate.constants import MAX_GEN_LENGTH_DEFAULT_CHAR_LEVEL, MAX_GEN_LENGTH_DEFAULT_WORD_LEVEL
from lingofunk_generate.constants import GEN_EPOCHS_DIVIDER_DEFAULT, GEN_EPOCHS_DIVIDER_DEFAULT_IF_DEGUG
from lingofunk_generate.constants import NUM_EPOCHS_DEFAULT
from lingofunk_generate.constants import TRAIN_SIZE_DEFAULT
from lingofunk_generate.constants import DROPOUT_DEFAULT
from lingofunk_generate.constants import NROWS_TO_READ_IF_DEBUG
from lingofunk_generate.utils import get_model_name
from lingofunk_generate.utils import log as _log


def log(text):
    _log(text, prefix='Fit: ')


def _remove_folder_for_models_if_exists():
    if os.path.exists(MODELS_FOLDER_PATH):
        log('remove folder for models (as it may be not empty)')
        shutil.rmtree(MODELS_FOLDER_PATH)


def _create_folder_for_models_if_not_exists():
    if not os.path.exists(MODELS_FOLDER_PATH):
        log('create folder for models')
        os.makedirs(MODELS_FOLDER_PATH)


def _read_data_csv(data_path, index_col=0, cols=None, nrows=None):
    log('read data')

    df = pd.read_csv(data_path, usecols=cols, index_col=index_col, nrows=nrows)
    df.reset_index(inplace=True)

    return df


def _fit_and_save_model(model_name,
                        data,
                        target_texts_labels,
                        text_col,
                        label_col,
                        word_level,
                        new_model,
                        train_size,
                        dropout,
                        num_epochs,
                        gen_epochs,
                        max_length,
                        max_gen_length,
                        debug=False):
    log('train model ' + model_name)

    texts = data[data[label_col].isin(target_texts_labels)][text_col].values
    model = textgenrnn(name=model_name)

    if max_length is None:
        max_length = MAX_LENGTH_DEFAULT_WORD_LEVEL if word_level else MAX_LENGTH_DEFAULT_CHAR_LEVEL
    if max_gen_length is None:
        max_gen_length = MAX_GEN_LENGTH_DEFAULT_WORD_LEVEL if word_level else MAX_GEN_LENGTH_DEFAULT_CHAR_LEVEL
    if gen_epochs is None:
        gen_epochs = num_epochs // GEN_EPOCHS_DIVIDER_DEFAULT_IF_DEGUG if debug else num_epochs // GEN_EPOCHS_DIVIDER_DEFAULT

    model.train_on_texts(
        texts,
        new_model=new_model,
        word_level=word_level,
        train_size=train_size,
        dropout=dropout,
        num_epochs=num_epochs,
        gen_epochs=gen_epochs,
        max_length=max_length,
        max_gen_length=max_gen_length)

    log('save model ' + model_name)

    try:
        move_model_data(model_name=model_name,
                        source_folder_path=os.getcwd())
    except IOError:
        log('error, fail to move model data files to special folder')


def _parse_args():
    log('parse args')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', action='store_true', required=False,
        help='Specify, if want to run fitting in debug mode. ' +\
             'It means that if parameter nrows is also specified, only first nrows will be read from .csv file')
    parser.add_argument(
        '--nrows', type=int, required=False, default=NROWS_TO_READ_IF_DEBUG,
        help='How many rows should be read in debug mode')

    parser.add_argument(
        '--data-path',type=str, required=False, default=DATA_FILE_PATH_DEFAULT,
        help='Path to data .csv file')
    parser.add_argument(
        '--text-col', type=str, required=True,
        help='Text column name in data file')
    parser.add_argument(
        '--label-col', type=str, required=True,
        help='Style label column name in data file')

    parser.add_argument(
        '--word-level', action='store_true', required=False,
        help='Specify, if want to build word-level models (instead of default char-level models)')
    parser.add_argument(
        '--new-model', action='store_true', required=False,
        help='Specify, if want to get new textgenrnn model, not pretrained one')
    parser.add_argument(
        '--train-size', type=float, required=False, default=TRAIN_SIZE_DEFAULT,
        help='Train size (validation size = 1.0 - train size)')
    parser.add_argument(
        '--dropout', type=float, required=False, default=DROPOUT_DEFAULT,
        help='Dropout (the proportion of tokens to be thrown away on each epoch)')

    parser.add_argument(
        '--num-epochs', type=int, required=False, default=NUM_EPOCHS_DEFAULT,
        help='Number of epochs to train the model')
    parser.add_argument(
        '--gen-epochs', type=int, required=False, default=None,
        help='Number of epochs, after each of which sample text generations ny the model will be displayed in console')
    parser.add_argument(
        '--max-length', type=int, required=False, default=None,
        help='Maximum number of previous tokens (words or chars) to take into account while predicting the next one')
    parser.add_argument(
        '--max-gen-length', type=int, required=False, default=None,
        help='Maximum number of tokens to generate as sample after gen_epochs')

    for text_style in TEXT_STYLES:
        parser.add_argument(
            '--labels-{}'.format(text_style), action='append', type=int, required=False,
            help='Texts of which labels should be treated as ones of style "{}"'.format(text_style))

    return parser.parse_args()


def _main():
    _remove_folder_for_models_if_exists()
    _create_folder_for_models_if_not_exists()

    args = _parse_args()

    data = _read_data_csv(
        args.data_path,
        cols=[args.text_col, args.label_col],
        nrows=args.nrows if args.debug else None)

    for text_style in TEXT_STYLES:
        text_labels = getattr(args, 'labels_' + text_style)

        if not text_labels:
            continue

        model_name = get_model_name(text_style)

        _fit_and_save_model(
            model_name,
            data,
            text_labels,
            args.text_col,
            args.label_col,
            args.word_level,
            args.new_model,
            args.train_size,
            args.dropout,
            args.num_epochs,
            args.gen_epochs,
            args.max_length,
            args.max_gen_length,
            args.debug)

if __name__ == '__main__':
    _main()
