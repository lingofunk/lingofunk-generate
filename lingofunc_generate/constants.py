import os

MODELS_FOLDER_PATH = os.path.join('..', 'models')

TEXT_CLASS_LABELS = ['positive', 'negative', 'neutral']

MODEL_DATA_FILES_ENDINGS = {
    'weights': '_weights.hdf5',
    'vocab': '_vocab.json',
    'config': '_config.json'
}
