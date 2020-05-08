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
        self.emb_lut = nn.Embedding(config.vocab_size, config.embedd_size, padding_idx=config.PAD_IDX)
        if config.pretrain_vectors:
            self.load_pretrained_vectors(config.pretrain_vectors)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -config.init_weight, config.init_weight)

        self.feature_num = len(config.feats)
        self.feature_embedding = nn.ModuleDict()
        for feat_name, feat_dim in config.feats:
            vocab = config.feat_vocab_sizes[feat_name]
            self.feature_embedding.add_module(feat_name, nn.Embedding(vocab['size'], feat_dim, padding_idx=vocab['pad_idx']).to(config.device))
            nn.init.uniform_(self.feature_embedding[feat_name].weight.data, -config.init_weight, config.init_weight)

    def load_pretrained_vectors(self, emb_file: str):
        raise NotImplementedError

    def forward(self, source: torch.LongTensor, feats: list) -> torch.FloatTensor:
        word_embs = self.emb_lut(source)
        word_list = [word_embs]
        for feat_name, idx in feats:
            feat_embs = self.feature_embedding[feat_name](idx)
            word_list.append(feat_embs)
        word_embs = torch.cat(word_list, 2)
        return word_embs

