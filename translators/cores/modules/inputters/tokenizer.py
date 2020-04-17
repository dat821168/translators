import os

from collections import Counter, OrderedDict
from torchtext.vocab import Vocab


class Tokenizer(object):
    def __init__(self, vocab_file: str = None, separator: str = '@@',
                 pad_token: str = '<pad>', unk_token: str = '<unk>',
                 sos_token: str = '<sos>', eos_token: str = '<eos>'):
        assert vocab_file is not None and os.path.exists(vocab_file), f"Error: Vocab file {vocab_file} not exits!!!"
        # Special tokens:
        self.separator = separator
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        if vocab_file:
            self.vocab: Vocab = self.__build_vocab(vocab_file)

    def __len__(self):
        return len(self.vocab)

    def __build_vocab(self, vocab_file: str) -> Vocab:
        specials = list(OrderedDict.fromkeys(tok for tok in [self.unk_token, self.pad_token,
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

    def tokenize(self, sent: str) -> list:
        tokens = sent.strip().split()
        return tokens

    def detokenize(self, idxs: list, delim: str = ' ') -> str:
        detok = delim.join([self.vocab.itos[idx] for idx in idxs])
        detok = detok.replace(self.separator + ' ', '')
        detok = detok.replace(self.separator, '')

        detok = detok.replace(self.sos_token, '')
        detok = detok.replace(self.sos_token, '')
        detok = detok.replace(self.pad_token, '')
        detok = detok.strip()
        return detok
