import torch
import torch.nn as nn


class EncoderBase(nn.Module):
    @staticmethod
    def _check_args(src, lengths=None):
        n_batch = src.size(1)
        if lengths is not None:
            n_batch_, = lengths.size()
            assert n_batch == n_batch_, f"Error: Not all encoder arguments have the same value: [{n_batch}, {n_batch_}]"

    @classmethod
    def forward(self, src: torch.LongTensor, lengths=None, feats=[]):
        raise NotImplementedError
