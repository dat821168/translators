import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .encoder import EncoderBase
from translators.cores.modules.inputters.embedder import Embedder
from translators.cores.functions import init_lstm_


class RREncoder(EncoderBase):
    def __init__(self, config, embedder: Embedder = None):
        super(RREncoder, self).__init__()

        assert embedder is not None, "Error: Embedder needs to be initialized!"

        # Embedding layer
        self.embedder = embedder

        # Encoder stacked LSTM layer
        self.encoder_layers = nn.ModuleList()
        # Bottom encoder layer is bi-directional
        self.encoder_layers.append(
            nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, bias=True,
                    batch_first=True, bidirectional=True))
        # 2nd encoder layer with input_size x2
        self.encoder_layers.append(
            nn.LSTM((2*config.hidden_size), config.hidden_size, num_layers=1, bias=True,
                    batch_first=True, bidirectional=False))
        # Remaining LSTM layers
        for _ in range(config.num_enc_layers - 2):
            self.encoder_layers.append(
                nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, bias=True,
                        batch_first=True, bidirectional=False))
        for layer in self.encoder_layers:
            init_lstm_(layer, config.init_weight)

        # Dropout
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, src: torch.LongTensor, lengths=None) -> torch.FloatTensor:
        self._check_args(src, lengths)

        emb = self.embedder(src)

        # Bottom encoder layer (bi-directional)
        emb = self.dropout(emb)
        packed_emb = pack_padded_sequence(emb, lengths.cpu().numpy(), batch_first=True)
        memory_bank, _ = self.encoder_layers[0](packed_emb)
        memory_bank, _ = pad_packed_sequence(memory_bank, batch_first=True)

        # 2nd encoder layer (uni-directional layers)
        memory_bank = self.dropout(memory_bank)
        memory_bank, _ = self.rnn_layers[1](memory_bank)

        # others encoder layer (uni-directional layers and residual connections),
        for enc_layer in self.encoder_layers[2:]:
            residual = memory_bank
            memory_bank = self.dropout(memory_bank)
            memory_bank, _ = enc_layer(memory_bank)
            memory_bank = memory_bank + residual

        return memory_bank
