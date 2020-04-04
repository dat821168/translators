import torch
import torch.nn as nn


"""
    Kế hoạch rộng thêm:
        + Position Embedding
        + Feature Embedding
        + Load embedding weights from pretrain files
"""


class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()
        self.emb_lut = nn.Embedding(config.vocab_size, config.embedd_size, padding_idx=config.PAD_idx)
        if config.pretrain_vectors:
            self.load_pretrained_vectors(config.pretrain_vectors)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -config.init_weight, config.init_weight)

    def load_pretrained_vectors(self, emb_file: str):
        raise NotImplementedError

    def forward(self, source: torch.LongTensor) -> torch.FloatTensor:
        return self.emb_lut(source)

