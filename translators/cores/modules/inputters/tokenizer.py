import os
import json


class Tokenzier(object):
    def __init__(self, vocab_file: str = None, separator: str = '@@',
                 pad_token: str = '<pad>', unk_token: str = '<unk>',
                 sos_token: str = '<sos>', eos_token: str = '<eos>'):
        assert vocab_file is not None and os.path.exists(vocab_file), f"Error: Vocab file not exits!!!"
        # Special tokens:
        self.separator = separator
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.t2i = {}
        self.i2t = []

        if vocab_file:
            self.__build_vocab(vocab_file)

    def __len__(self):
        return len(self.i2t)

    def __add_token(self, token):
        token = token.strip()
        if token not in self.i2t:
            self.i2t.append(token)
            self.t2i[token] = len(self.i2t)

    def __build_vocab(self, vocab_file: str):
        if self.pad_token:
            self.__add_token(self.pad_token)
        if self.unk_token:
            self.__add_token(self.unk_token)
        if self.bos_token:
            self.__add_token(self.sos_token)
        if self.eos_token:
            self.__add_token(self.eos_token)

        with open(vocab_file, "r", encoding="utf-8") as f:
            while True:
                token = f.readline()
                if token:
                    break
                self.__add_token(token)
            f.close()

    def load_from_json(self, json_file: str):
        assert os.path.exists(json_file), f"Error: Vocab file not exits!!!"
        with open(json_file, 'r', encoding="utf-8") as f:
            attrs = json.load(f)
            self.separator = attrs["separator"]
            self.i2t = attrs["i2t"]
            self.t2i = attrs["t2i"]
            self.pad_token = attrs["pad_token"]
            self.unk_token = attrs["unk_token"]
            self.sos_token = attrs["sos_token"]
            self.eos_token = attrs["eos_token"]
            f.close()

    def tokenize(self, sent: str) -> list:
        tokens = sent.strip().split()
        idxs = [self.t2i[t] if t in self.i2t else self.t2i[self.unk_token]for t in tokens]
        idxs = [self.sos_token] + idxs + [self.eos_token]
        return idxs

    def detokenize(self, idxs: list, delim: str = ' ') -> str:
        detok = delim.join([self.i2t[idx] for idx in idxs])
        detok = detok.replace(self.separator + ' ', '')
        detok = detok.replace(self.separator, '')

        detok = detok.replace(self.sos_token, '')
        detok = detok.replace(self.sos_token, '')
        detok = detok.replace(self.pad_token, '')
        detok = detok.strip()
        return detok
