import torch
import torch.nn as nn
import itertools

from .decoder import DecoderBase
from translators.cores.functions.common import init_lstm_
from translators.cores.modules.attentions import BahdanauAttention


class RecurrentAttention(nn.Module):
    def __init__(self, config, input_size: int = 1024, context_size: int = 1024, hidden_size: int = 1024):
        super(RecurrentAttention, self).__init__()

        self.bottom_layer = nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
                                    batch_first=True, bidirectional=False)
        init_lstm_(self.bottom_layer, config.init_weight)

        self.attn = BahdanauAttention(config, hidden_size, context_size, context_size, normalize=True)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.FloatTensor, hidden: torch.FloatTensor, context: torch.FloatTensor, context_len):
        # set attention mask, sequences have different lengths, this mask
        # allows to include only valid elements of context in attention's
        # softmax

        self.attn.set_mask(context_len, context)

        inputs = self.dropout(inputs)
        recurrent_outputs, hidden = self.bottom_layer(inputs, hidden)
        attn_outputs, scores = self.attn(recurrent_outputs, context)

        return recurrent_outputs, hidden, attn_outputs, scores


class Classifier(nn.Module):
    """
    Fully-connected classifier
    """
    def __init__(self, in_features, out_features, init_weight=0.1):
        """
        Constructor for the Classifier.
        :param in_features: number of input features
        :param out_features: number of output features (size of vocabulary)
        :param init_weight: range for the uniform initializer
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        """
        Execute the classifier.
        :param x: output from decoder
        """
        out = self.classifier(x)
        return out


class RRDecoder(DecoderBase):
    def __init__(self, config, embedder):
        super(RRDecoder, self).__init__()

        assert embedder is not None, "Error: Embedder needs to be initialized!"

        self.num_layers = config.num_dec_layers
        # Embedding layer
        self.embedder = embedder

        # Attention layer
        self.att_rnn = RecurrentAttention(config, config.hidden_size, config.hidden_size, config.hidden_size)

        # Decoder stacked LSTM layer
        self.decoder_layers = nn.ModuleList()

        for _ in range(config.num_dec_layers - 1):
            self.decoder_layers.append(
                nn.LSTM((2*config.hidden_size), config.hidden_size, num_layers=1, bias=True,
                        batch_first=True, bidirectional=False))

        for layer in self.decoder_layers:
            init_lstm_(layer, config.init_weight)

        # Dropout
        self.dropout = nn.Dropout(p=config.dropout)

        # Classifier layer
        self.classifier = Classifier(config.hidden_size, config.vocab_size)

    def init_hidden(self, hidden):
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(self.num_layers)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers

        self.next_hidden = []

        return hidden

    def append_hidden(self, h):
        if self.inference:
            self.next_hidden.append(h)

    def package_hidden(self):
        if self.inference:
            hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
        else:
            hidden = None
        return hidden

    def forward(self, inputs: torch.LongTensor, context: tuple, inference: bool = False):

        self.inference = inference

        enc_context, enc_len, hidden = context
        hidden = self.init_hidden(hidden)

        emb = self.embedder(inputs)

        x, h, attn, scores = self.att_rnn(emb, hidden[0], enc_context, enc_len)
        self.append_hidden(h)

        x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.decoder_layers[0](x, hidden[1])
        self.append_hidden(h)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = torch.cat((x, attn), dim=2)
            x = self.dropout(x)
            x, h = self.decoder_layers[i](x, hidden[i + 1])
            self.append_hidden(h)
            x = x + residual

        x = self.classifier(x)
        hidden = self.package_hidden()

        return x, scores, [enc_context, enc_len, hidden]