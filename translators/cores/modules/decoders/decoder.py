import torch.nn as nn


class DecoderBase(nn.Module):
    def __init__(self):
        super(DecoderBase, self).__init__()

    @classmethod
    def forward(self, *input):
        raise NotImplementedError
