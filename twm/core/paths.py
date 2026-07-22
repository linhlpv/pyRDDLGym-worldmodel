'''Locations of the directories created at runtime for data, models and plots.

The directories are created lazily by ``ensure_dir`` at the point of writing,
so that merely importing the package never touches the file system.
'''

import os

PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(PACKAGE_PATH, 'data')
MODEL_PATH = os.path.join(PACKAGE_PATH, 'models')
PLOTS_PATH = os.path.join(PACKAGE_PATH, 'plots')

__all__ = ['PACKAGE_PATH', 'DATA_PATH', 'MODEL_PATH', 'PLOTS_PATH',
           'ensure_dir', 'data_file', 'model_file', 'plot_file']


def ensure_dir(path: str) -> str:
    '''Creates a directory if it does not exist yet and returns it.'''
    os.makedirs(path, exist_ok=True)
    return path


def data_file(name: str) -> str:
    '''Returns the path of a file in the data directory, creating the directory.'''
    return os.path.join(ensure_dir(DATA_PATH), name)


def model_file(name: str) -> str:
    '''Returns the path of a file in the model directory, creating the directory.'''
    return os.path.join(ensure_dir(MODEL_PATH), name)


def plot_file(name: str) -> str:
    '''Returns the path of a file in the plots directory, creating the directory.'''
    return os.path.join(ensure_dir(PLOTS_PATH), name)
