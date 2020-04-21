import os
import torch

from torchtext.data import Field, RawField, Example, Iterator, Dataset
from translators.cores.modules.inputters import Tokenizer


def get_field(tokenizer: Tokenizer, fields: tuple = None, use_test: bool = False) -> (Field, Field, RawField):
    raw_field = None
    if fields:
        src_field, tgt_field = fields
    else:
        src_field = Field(tokenize=tokenizer.tokenize, init_token=tokenizer.sos_token, eos_token=tokenizer.eos_token,
                          pad_token=tokenizer.pad_token, unk_token=tokenizer.unk_token, lower=True, batch_first=True,
                          include_lengths=True)
        tgt_field = Field(tokenize=tokenizer.tokenize, init_token=tokenizer.sos_token, eos_token=tokenizer.eos_token,
                          pad_token=tokenizer.pad_token, unk_token=tokenizer.unk_token, lower=True, batch_first=True,
                          include_lengths=True)
        src_field.vocab = tokenizer.vocab
        tgt_field.vocab = tokenizer.vocab
    if use_test:
        raw_field = RawField(preprocessing=None, postprocessing=None, is_target=True)
    return src_field, tgt_field, raw_field


class NMTDataset(object):
    def __init__(self, src_file: str, tgt_file: str, fields: tuple,
                 min_len: int = 1, max_len: int = 256,
                 device: str = None, is_train: bool = False,
                 batch_size: int = 8):

        assert os.path.isfile(src_file), f'Error: Source file {src_file} is not exits!!!'
        assert os.path.isfile(tgt_file), f'Error: Target file {tgt_file} is not exits!!!'

        self.min_len = min_len
        self.max_len = max_len

        self.batch_size = batch_size
        self.is_train = is_train

        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.src_field, self.tgt_field = fields

        self.dataset = self.__read_data(src_file, tgt_file)

    def __read_data(self, src_file: str, tgt_file: str) -> (list, list):
        examples = []
        srcs = open(src_file, 'r', encoding="utf-8").readlines()
        tgts = open(tgt_file, 'r', encoding="utf-8").readlines()
        fields = [('src', self.src_field), ('tgt', self.tgt_field)]
        assert len(srcs) == len(tgts), \
            f"Number of examples in source and targets ({len(srcs)} and (len{tgts}) are not matches!!!"

        for src, tgt in list(zip(srcs, tgts)):
            src_len = src.count(' ') + 1
            tgt_len = tgt.count(' ') + 1
            if self.min_len <= src_len <= self.max_len and \
                    self.min_len <= tgt_len <= self.max_len:
                examples.append(Example.fromlist((src, tgt), fields))
        return Dataset(examples, fields)

    def iter_dataset(self):
        cur_iter = Iterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            device=self.device,
            batch_size_fn=None,
            train=self.is_train,
            repeat=False,
            shuffle=True,
            sort=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
        )
        return cur_iter

    def __iter__(self):
        for batch in self.iter_dataset():
            yield batch
        return
