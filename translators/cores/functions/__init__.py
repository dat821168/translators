import os
import torch
import dill
from translators.logger import logger
from translators.cores.functions.lr_scheduler import WarmupMultiStepLR

from translators.cores.functions.common import init_lstm_, plot_figure, save_train_history
from translators.cores.functions.mextrics import BLUE


def save_dataset(save_dir, dataset):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_file_path = os.path.join(save_dir, 'nmt_dataset.cache')
    cache = {}
    for k, v in dataset.items():
        cache[k] = (v.dataset.examples, v.dataset.fields)
    with open(data_file_path, "wb")as f:
        dill.dump(cache, f)


def load_dataset(dataset_file: str, device: str = "cuda"):
    assert os.path.exists(dataset_file), f'{dataset_file} is not exits !!!'
    with open(dataset_file, "rb")as f:
        cache = dill.load(f)
    return cache


_all__ = ['init_lstm_', 'plot_figure', 'save_train_history',
          'WarmupMultiStepLR',
          'BLUE',
          'save_dataset', 'load_dataset']
