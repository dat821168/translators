import torch
import torch.nn as nn

from translators.configuration import Config
from translators.models import NMTModel, GNMT
from translators.cores.modules.inputters import Tokenizer, NMTDataset, get_field
from translators.cores.functions import WarmupMultiStepLR


def build_criterion(vocab_size: int, padding_idx: int, device: str):
    loss_weight = torch.ones(vocab_size)
    loss_weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    criterion = criterion.to(device)
    return criterion


def build_model(config: Config) -> NMTModel:
    if config.model_type == "GNMT":
        model = GNMT(config)
    else:
        model = None
    model = model.to(config.device)
    return model


def build_tokenizer(config: Config) -> Tokenizer:
    tokenizer = Tokenizer(config.vocab_file)
    config.vocab_size = len(tokenizer)
    return tokenizer


def build_dataset(config, tokenizer: Tokenizer):
    dataset = {}
    fields = get_field(tokenizer)
    if config.train_src_file:
        dataset['train'] = NMTDataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file, fields=fields,
                                      min_len=config.min_len, max_len=config.max_len,
                                      device=config.device, is_train=True,
                                      batch_size=config.batch_size)
    if config.dev_src_file:
        dataset['dev'] = NMTDataset(src_file=config.dev_src_file, tgt_file=config.dev_tgt_file, fields=fields,
                                    min_len=config.min_len, max_len=config.max_len,
                                    device=config.device, is_train=False,
                                    batch_size=config.batch_size)
    if config.test_src_file:
        dataset['test'] = NMTDataset(src_file=config.test_src_file, tgt_file=config.test_tgt_file, fields=fields,
                                     min_len=config.min_len, max_len=config.max_len,
                                     device=config.device, is_train=True,
                                     batch_size=config.batch_size)

    return dataset