import os
import shutil
from textgenrnn.textgenrnn import textgenrnn


MODELS_FOLDER_PATH = os.path.join(
    '..', 'models'
)

MODEL_DATA_FILES_ENDINGS = {
    'weights': '_weights.hdf5',
    'vocab': '_vocab.json',
    'config': '_config.json'
}

MODEL_DATA_FILES_TYPES = list(MODEL_DATA_FILES_ENDINGS.keys())


def log(text, prefix='Utils: '):
    print(prefix + text)


def move_model_data(model_name, source_folder_path, target_folder_path=None):
    log('move model data')

    if not os.path.isdir(source_folder_path):
        raise IOError('Source folder "{}" not exists'.format(source_folder_path))

    if target_folder_path is not None:
        if not os.path.isdir(target_folder_path):
            raise IOError('Tagret folder "{}" not exists'.format(target_folder_path))
    else:
        target_folder_path = MODELS_FOLDER_PATH

    for f in os.listdir(source_folder_path):
        if not f.startswith(model_name):
            continue

        for ending in MODEL_DATA_FILES_ENDINGS.values():
            if f.endswith(ending):
                shutil.move(
                    os.path.join(source_folder_path, f),
                    os.path.join(target_folder_path, f))


def restore_model_from_data(model_name, data_folder_path=None):
    log('restore model "{}"'.format(model_name))

    if data_folder_path is not None:
        if not os.path.isdir(data_folder_path):
            raise IOError('Data folder "{}" not exists'.format(data_folder_path))
    else:
        data_folder_path = MODELS_FOLDER_PATH

    data_files_paths = dict.fromkeys(MODEL_DATA_FILES_TYPES, None)

    for f in os.listdir(data_folder_path):
        if not f.startswith(model_name):
            continue

        for file_type, ending in MODEL_DATA_FILES_ENDINGS.items():
            if f.endswith(ending):
                data_files_paths[file_type] = os.path.join(data_folder_path, f)

    if data_files_paths['weights'] is None:
        raise ValueError('No weights found for model "{}"'.format(model_name))

    return textgenrnn(name=model_name,
                      weights_path=data_files_paths['weights'],
                      vocab_path=data_files_paths['vocab'],
                      config_path=data_files_paths['config'])
