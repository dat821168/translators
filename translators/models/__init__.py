import os
import torch
import torch.nn as nn

from .gnmt import GNMT
from .nmt import NMTModel
from .seq2seq import Seq2Seq
from translators.logger import logger


def count_parameters(model, trainable: bool = True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)


def save_checkpoint(cnf, model, optimizer, tokenizer, step):
    if not os.path.exists(cnf.save_dir):
        os.mkdir(cnf.save_dir)

    nmtmodel = model.module if isinstance(model, nn.DataParallel) else model
    chkpt_file_path = os.path.join(cnf.save_dir, 'nmt_model_chkpt.pt')
    chkpt = {
        'step': step,
        'model': nmtmodel.state_dict(),
        'optimizer': optimizer,
        'vocab': {"tokens": tokenizer.vocab,
                  "feats": tokenizer.feat_vocabs}
    }
    torch.save(chkpt, chkpt_file_path)
    logger.info(f'Best model was saved in {chkpt_file_path} !!!')


def load_chkpt(chkpt_file: str, optimizer=None, device: str = 'cuda'):
    assert os.path.exists(chkpt_file), f'{chkpt_file} is not exits !!!'

    chkpt = torch.load(chkpt_file, map_location=device)
    step = chkpt['step']
    vocab = chkpt['vocab']

    if optimizer is not None:
        optimizer = chkpt['optimizer']

    if optimizer is not None:
        return step, chkpt['model'], optimizer, vocab
    else:
        return step, chkpt['model'], None, vocab


__all__ = ['NMTModel', 'GNMT', 'Seq2Seq'
           'count_parameters', 'save_checkpoint']


