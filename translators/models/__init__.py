import os
import torch
import torch.nn as nn

from .gnmt import GNMT
from .nmt import NMTModel
from translators.logger import logger


def count_parameters(model, trainable: bool = True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)


def save_checkpoint(cnf, model, optimizer, fields, epoch):
    if not os.path.exists(cnf.save_dir):
        os.mkdir(cnf.save_dir)

    nmtmodel = model.module if isinstance(model, nn.DataParallel) else model
    chkpt_file_path = os.path.join(cnf.save_dir, 'nmt_model_chkpt.pt')
    chkpt = {
        'epoch': epoch,
        'model': nmtmodel.state_dict(),
        'optimizer': optimizer,
        'fields': fields
    }
    torch.save(chkpt, chkpt_file_path)
    logger.info(f'Best model was saved in {chkpt_file_path} !!!')


def load_chkpt(chkpt_file: str, optimizer=None, device: str = 'cuda'):
    assert os.path.exists(chkpt_file), f'{chkpt_file} is not exits !!!'

    chkpt = torch.load(chkpt_file, map_location=device)
    epoch = chkpt['epoch']
    fields = chkpt['fields']

    if optimizer is not None:
        optimizer = chkpt['optimizer']

    if optimizer is not None:
        return epoch, chkpt['model'], optimizer, fields
    else:
        return epoch, chkpt['model'], None, fields


__all__ = ['NMTModel', 'GNMT',
           'count_parameters', 'save_checkpoint']


