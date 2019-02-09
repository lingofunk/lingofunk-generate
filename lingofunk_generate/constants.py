import sys
import os

project_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_folder)

MODELS_FOLDER_PATH = os.path.join(project_folder, 'models')
DATA_FOLDER_PATH = os.path.join(project_folder, 'data')

DATA_FILE_NAME_DEFAULT = 'review.csv'
DATA_FILE_PATH_DEFAULT = os.path.join(DATA_FOLDER_PATH, DATA_FILE_NAME_DEFAULT)

TEXT_STYLES = [
    'positive',
    'negative',
    'neutral'
]

MODEL_DATA_FILES_ENDINGS = {
    'weights': '_weights.hdf5',
    'vocab': '_vocab.json',
    'config': '_config.json'
}

MAX_LENGTH_DEFAULT_WORD_LEVEL = 10
MAX_LENGTH_DEFAULT_CHAR_LEVEL = 40
MAX_GEN_LENGTH_DEFAULT_WORD_LEVEL = 50
MAX_GEN_LENGTH_DEFAULT_CHAR_LEVEL = 300

NUM_EPOCHS_DEFAULT = 10
TRAIN_SIZE_DEFAULT = 0.9
DROPOUT_DEFAULT = 0.02

DEFAULT_TEXT_IF_REQUIRED_MODEL_NOT_FOUND = 'Required model not found. Sorry'
NUM_TEXT_GENERATIONS_TO_TEST_A_MODEL_AFTER_LOADING = 2

PORT_DEFAULT = 8000
TEMPERATURE_DEFAULT = 0.5
