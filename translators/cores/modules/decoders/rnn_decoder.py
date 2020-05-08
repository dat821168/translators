import torch
import torch.nn as nn

from .decoder import DecoderBase
from translators.cores.functions.common import init_lstm_
from translators.cores.modules.generator import Classifier


class RNNDecoder(DecoderBase):
    def __init__(self, config, embedder, num_rnn_layers: int = 1,):
        super().__init__()

        self.embedder = embedder

        self.rnn = nn.LSTM(config.embedd_size, config.hidden_size, num_rnn_layers, dropout=config.dropout, batch_first=True)

        init_lstm_(self.rnn, config.init_weight)

        self.classifier = Classifier(config.hidden_size,  config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.LongTensor, hidden):
        embedded = self.dropout(self.embedder(inputs, []))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.classifier(output)
        return prediction, hidden