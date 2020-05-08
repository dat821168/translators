import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from translators.cores.functions.common import init_lstm_

from .encoder import EncoderBase
from translators.cores.modules.inputters.embedder import Embedder


class RNNEncoder(EncoderBase):
    def __init__(self, config, embedder: Embedder = None, num_rnn_layers: int = 1, bidirectional: bool = False,
                 rnn_type: str = 'LSTM'):
        super(RNNEncoder, self).__init__()

        total_embedd_size = config.embedd_size + sum([size for _, size in config.feats])

        self.embedding = embedder

        self.rnn = nn.LSTM(total_embedd_size, config.hidden_size, num_layers=num_rnn_layers, dropout=config.dropout,
                           bidirectional=False, batch_first=True)
        init_lstm_(self.rnn, config.init_weight)
        # Dropout
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, src: torch.LongTensor, lengths=None, feats=[]):
        embedded = self.embedding(src, feats)
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_outputs, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_outputs)
        return output, hidden