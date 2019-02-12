import sys
import os
import shutil
import argparse
import pandas as pd
import time
import numpy as np
import tensorflow as tf
from textgenrnn.textgenrnn import textgenrnn

project_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_folder)

from lingofunk_generate.model_restore_utils import move_model_data
from lingofunk_generate.model_restore_utils import load_model
from lingofunk_generate.constants import MODELS_FOLDER_PATH
from lingofunk_generate.constants import TEXT_STYLES
from lingofunk_generate.constants import DATA_FILE_PATH_DEFAULT
from lingofunk_generate.constants import MAX_LENGTH_DEFAULT_CHAR_LEVEL, MAX_LENGTH_DEFAULT_WORD_LEVEL
from lingofunk_generate.constants import MAX_GEN_LENGTH_DEFAULT_CHAR_LEVEL, MAX_GEN_LENGTH_DEFAULT_WORD_LEVEL
from lingofunk_generate.constants import NUM_EPOCHS_DEFAULT
from lingofunk_generate.constants import TRAIN_SIZE_DEFAULT
from lingofunk_generate.constants import DROPOUT_DEFAULT
from lingofunk_generate.constants import BATCH_SIZE_DEFAULT
from lingofunk_generate.utils import get_model_name
from lingofunk_generate.utils import log as _log


graph = tf.get_default_graph()
np.random.seed(int(time.time()))


def log(text):
    _log(text, prefix='Fit: ')


def _remove_folder_for_models_if_exists(models_folder):
    if os.path.exists(models_folder):
        log('remove folder for models "{}" (as it may be not empty)'.format(models_folder))
        shutil.rmtree(models_folder)


def _create_folder_for_models_if_not_exists(models_folder):
    if not os.path.exists(models_folder):
        log('create folder for models "{}"'.format(models_folder))
        os.makedirs(models_folder)


def _read_data_csv(data_path, index_col=0, cols=None, nrows=None):
    log('read data')

    df = pd.read_csv(data_path, usecols=cols, sep=',', index_col=index_col, nrows=nrows)
    df.reset_index(inplace=True)

    return df


def _stratify_data(df, text_col, label_col, max_texts_per_label=None):
    if max_texts_per_label is None:
        return df

    df = df.sample(frac=1).reset_index(drop=True)
    labels = df[label_col].unique().tolist()

    # TODO: optimize

    texts_labels = []

    for label in labels:
        reviews_current = df[df[label_col] == label]

        texts_labels_current = list(
            zip(reviews_current[text_col].values,
                reviews_current[label_col].values)
        )

        texts_labels_current = texts_labels_current[:max_texts_per_label]  # None also OK
        texts_labels += texts_labels_current

        log('For label "{}" keep "{}" texts'.format(label, len(texts_labels_current)))

    np.random.shuffle(texts_labels)

    texts = [p[0] for p in texts_labels]
    labels = [p[1] for p in texts_labels]

    df_result = pd.DataFrame()
    df_result[text_col] = texts
    df_result[label_col] = labels

    return df_result


def _fit_and_save_model(text_style,
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
                        batch_size,
                        load_models=False,
                        models_folder_source=None,
                        models_folder_target=MODELS_FOLDER_PATH):
    model_name = get_model_name(text_style)
    log('train model ' + model_name)

    if len(target_texts_labels) == 0:
        log('no labels, finish training')
        return

    model_name = get_model_name(text_style)

    if not load_models:
        model = textgenrnn(name=model_name)
    else:
        log('restore model "{}"'.format(model_name))
        model = load_model(text_style, models_folder_source, graph)

    texts = data[data[label_col].isin(target_texts_labels)][text_col].values

    if max_length is None:
        max_length = MAX_LENGTH_DEFAULT_WORD_LEVEL if word_level else MAX_LENGTH_DEFAULT_CHAR_LEVEL
    if max_gen_length is None:
        max_gen_length = MAX_GEN_LENGTH_DEFAULT_WORD_LEVEL if word_level else MAX_GEN_LENGTH_DEFAULT_CHAR_LEVEL

    model.train_on_texts(
        texts,
        new_model=new_model,
        word_level=word_level,
        train_size=train_size,
        dropout=dropout,
        num_epochs=num_epochs,
        gen_epochs=gen_epochs,
        max_length=max_length,
        max_gen_length=max_gen_length,
        batch_size=batch_size)

    log('save model ' + model_name)

    try:
        move_model_data(model_name=model_name,
                        source_folder_path=os.getcwd(),
                        target_folder_path=models_folder_target)
    except IOError:
        log('error, fail to move model data files to special folder')


def _parse_args():
    log('parse args')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--nrows', type=int, required=False, default=None,
        help='How many data rows should be read')

    parser.add_argument(
        '--load-models', action='store_true', required=False,
        help='Try to load models and to continue fitting them')
    parser.add_argument(
        '--models-folder-source', type=str, required=False, default=MODELS_FOLDER_PATH,
        help='Folder to load models (weights, vocabs, configs) from')
    parser.add_argument(
        '--models-folder-target', type=str, required=False, default=MODELS_FOLDER_PATH,
        help='Folder to save models (weights, vocabs, configs) to')

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
        '--gen-epochs', type=int, required=False, default=0,
        help='Number of epochs, after each of which sample text generations ny the model will be displayed in console')
    parser.add_argument(
        '--max-length', type=int, required=False, default=None,
        help='Maximum number of previous tokens (words or chars) to take into account while predicting the next one')
    parser.add_argument(
        '--max-gen-length', type=int, required=False, default=None,
        help='Maximum number of tokens to generate as sample after gen_epochs')

    parser.add_argument(
        '--batch-size', type=int, required=False, default=BATCH_SIZE_DEFAULT,
        help='Batch size')

    parser.add_argument(
        '--max-texts-per-label', type=int, required=False, default=None,
        help='Maximum number of texts to keep per each label')

    for text_style in TEXT_STYLES:
        parser.add_argument(
            '--style-{}'.format(text_style), action='store_true', required=False,
            help='Train model for style "{}"'.format(text_style))

    for text_style in TEXT_STYLES:
        parser.add_argument(
            '--labels-{}'.format(text_style), action='append', type=int, required=False,
            help='Texts of which labels should be treated as ones of style "{}"'.format(text_style))

    return parser.parse_args()


def _main():
    args = _parse_args()

    text_styles = [
        text_style for text_style in TEXT_STYLES if getattr(args, 'style_' + text_style)
    ]

    if not args.load_models:
        _remove_folder_for_models_if_exists(args.models_folder_target)
        _create_folder_for_models_if_not_exists(args.models_folder_target)
    else:
        if not os.path.isdir(args.models_folder_source):
            raise IOError('Can\'t load models from folder "{}": folder not exists'.format(args.models_folder_source))

    if not os.path.exists(args.models_folder_target):
        log('create target folder for models "{}"'.format(args.models_folder_target))
        os.makedirs(args.models_folder_target)

    data = _read_data_csv(
        args.data_path,
        cols=[args.text_col, args.label_col],
        nrows=args.nrows)

    data = _stratify_data(data, args.text_col, args.label_col, args.max_texts_per_label)

    for text_style in text_styles:
        assert text_style in TEXT_STYLES, 'No such text style "{}" in known styles "{}". Can\'t fit'.format(
            text_style, TEXT_STYLES)

        text_labels = getattr(args, 'labels_' + text_style)

        _fit_and_save_model(
            text_style,
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
            args.batch_size,
            args.load_models,
            args.models_folder_source,
            args.models_folder_target)

if __name__ == '__main__':
    _main()
