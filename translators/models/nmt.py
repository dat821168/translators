import torch
import torch.nn as nn

from torch.nn.functional import log_softmax
from translators.cores.modules import EncoderBase, DecoderBase


class NMTModel(nn.Module):
    def __init__(self):
        super(NMTModel, self).__init__()
        self.encoder = EncoderBase
        self.decoder = DecoderBase

    def encode(self, src: torch.LongTensor, lengths, feats):
        return self.encoder(src, lengths, feats)

    def decode(self, dec_in: torch.LongTensor, context: tuple,  inference=False):
        return self.decoder(dec_in, context, inference)

    def generate(self, inputs, context, beam_size):
        logits, scores, new_context = self.decode(inputs, context, True)
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context

    def forward(self, *input):
        raise NotImplementedError