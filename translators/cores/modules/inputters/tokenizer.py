import os
import re
import stanza

from collections import Counter, OrderedDict
from torchtext.vocab import Vocab

NLP = stanza.Pipeline(lang="en", processors='tokenize,mwt,pos,lemma,depparse')


class Tokenizer(object):
    def __init__(self, vocab: str = None, features: list = [], separator: str = '@@',
                 pad_token: str = '<pad>', unk_token: str = '<unk>',
                 sos_token: str = '<sos>', eos_token: str = '<eos>'):

        # Special tokens:
        self.separator = separator
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.feats = features
        if isinstance(vocab, str):
            assert vocab is not None and os.path.exists(vocab), f"Error: Vocab file {vocab} not exits!!!"
            self.feat_vocabs = {}
            self.__get_feats_vocabs()
            self.vocab: Vocab = self.__build_vocab(vocab)
        else:
            self.vocab: Vocab = vocab['tokens']
            self.feat_vocabs = vocab['feats']

    def __len__(self):
        return len(self.vocab)

    def __get_feats_vocabs(self):
        for feat_name, _ in self.feats:
            if feat_name == 'deprel':
                counter = Counter(NLP.processors['depparse'].vocab._vocabs['deprel']._id2unit)
                vocab = Vocab(counter=counter, specials=[self.sos_token, self.eos_token])
                self.feat_vocabs['deprel'] = vocab

    def __build_vocab(self, vocab_file: str) -> Vocab:
        specials = list(OrderedDict.fromkeys(tok for tok in [self.pad_token, self.unk_token,
                                                             self.sos_token, self.eos_token]))
        counter = Counter()
        with open(vocab_file, "r", encoding="utf-8") as f:
            while True:
                token = f.readline()
                if not token:
                    break
                counter.update([token.strip()])
            f.close()
        return Vocab(counter=counter, specials=specials)

    def tokenize(self, sent: str, layer: int = 0, tok_delim: str = " ", feat_delim: str = "|") -> list:
        tokens = sent.strip().split(tok_delim)
        if feat_delim is not None:
            tokens = [t.split(feat_delim)[layer] for t in tokens]
        return tokens

    def detokenize(self, idxs: list, delim: str = ' ') -> str:
        detok = delim.join([self.vocab.itos[idx] for idx in idxs])
        detok = re.sub(self.separator + ' ', '', detok)
        detok = re.sub(self.separator, '', detok)

        detok = re.sub(self.sos_token, '', detok)
        detok = re.sub(self.eos_token, '', detok)
        detok = re.sub(self.pad_token, '', detok)
        detok = detok.strip()
        return detok


if __name__ == "__main__":
    tok = Tokenizer(vocab="E:/Projects/Cores/translators/datasets/bpe_data/vocab.bpe.10000")
    tok.get_feature("Bar@@ ack Ob@@ am@@ a was born in Ha@@ waii@@ .  He was elected president in 2008@@ .")