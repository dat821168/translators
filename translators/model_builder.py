import torch
import torch.nn as nn

from translators.logger import logger
from translators.configuration import Config
from translators.models import NMTModel, GNMT, count_parameters
from translators.cores.modules.inputters import Tokenizer, NMTDataset, get_field
from translators.cores.modules.generator import SequenceGenerator


def build_criterion(vocab_size: int, padding_idx: int, device: str):
    loss_weight = torch.ones(vocab_size)
    loss_weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight=loss_weight, reduction='mean')
    criterion = criterion.to(device)
    return criterion


def build_model(config: Config) -> NMTModel:
    logger.info("Building Model ...")

    if config.model_type == "GNMT":
        model = GNMT(config)
    else:
        model = None
    model = model.to(config.device)

    parameters = [count_parameters(model, True), count_parameters(model, False)]
    logger.info(f'{"#"*15}Model Summary{"#"*15}')
    logger.info(f'{model}')
    logger.info(f'{"="*40}')
    logger.info(f'Total parameters: {sum(parameters):,}')
    logger.info(f'Trainable parameters: {parameters[0]:,}')
    logger.info(f'Non-trainable parameters: {parameters[1]:,}')
    logger.info(f'{"=" * 40}')
    logger.info('\n')
    return model


def build_generator(model, beam_size: int, max_seq_len: int, cuda: bool, len_norm_factor: float = 0.6,
                    len_norm_const: float = 5.0, cov_penalty_factor: float = 0.1,
                    sos_idx: int = 2, eos_idx: int = 3):
    logger.info("Building Generator ...")
    generator = SequenceGenerator(model=model, beam_size=beam_size, max_seq_len=max_seq_len, cuda=cuda,
                                  len_norm_factor=len_norm_factor, len_norm_const=len_norm_const,
                                  cov_penalty_factor=cov_penalty_factor, sos_idx=sos_idx, eos_idx=eos_idx)
    logger.info('\n')
    return generator


def build_tokenizer(config: Config, vocab=None) -> Tokenizer:
    if vocab:
        logger.info("Building Tokenizer by pre-train ...")
        tokenizer = Tokenizer(vocab)
    else:
        logger.info(f"Building Tokenizer from {config.vocab_file} ...")
        tokenizer = Tokenizer(config.vocab_file)
    config.vocab_size = len(tokenizer)

    logger.info('\n')
    return tokenizer


def build_dataset(config, tokenizer: Tokenizer, fields: tuple = None):
    dataset = {}
    src_field, tgt_field, raw_field = get_field(tokenizer=tokenizer,
                                                fields=fields,
                                                use_test=True if config.test_src_file else False)
    logger.info("Building Corpus ...")
    if config.train_src_file:
        logger.info(f"\tReading train dataset ...")
        fields = (src_field, tgt_field)
        dataset['train'] = NMTDataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file, fields=fields,
                                      min_len=config.min_len, max_len=config.max_len,
                                      device=config.device, is_train=True,
                                      batch_size=config.batch_size)
    if config.dev_src_file:
        logger.info(f"\tReading eval dataset ...")
        fields = (src_field, tgt_field)
        dataset['eval'] = NMTDataset(src_file=config.dev_src_file, tgt_file=config.dev_tgt_file, fields=fields,
                                     min_len=config.min_len, max_len=config.max_len,
                                     device=config.device, is_train=False,
                                     batch_size=config.batch_size)
    if config.test_src_file:
        logger.info(f"\tReading test dataset ...")
        fields = (src_field, raw_field)
        dataset['test'] = NMTDataset(src_file=config.test_src_file, tgt_file=config.test_tgt_file, fields=fields,
                                     min_len=config.min_len, max_len=config.max_len,
                                     device=config.device, is_train=False,
                                     batch_size=config.batch_size)

    logger.info(f'{"#"*15}Corpus Summary{"#"*15}')
    logger.info(f'Batch size: {config.batch_size}')
    logger.info(f'Min length: {config.min_len}')
    logger.info(f'Max length: {config.max_len}')
    logger.info(f'{"=" * 40}')
    logger.info(f'Num of train examples: {len(dataset["train"].dataset.examples) if "train" in dataset else 0:,}')
    logger.info(f'Num of eval examples: {len(dataset["eval"].dataset.examples) if "eval" in dataset else 0:,}')
    logger.info(f'Num of test examples: {len(dataset["test"].dataset.examples) if "test" in dataset else 0:,}')
    logger.info(f'{"=" * 40}')
    logger.info('\n')
    return dataset, fields