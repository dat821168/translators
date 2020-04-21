import os
import time
import math
import torch
import torch.nn as nn

from tqdm import tqdm
from translators import logger


class Statistics(object):
    """
    Accumulator for statistics:
       * average loss
       * accuracy
       * perplexity
       * elapsed time
    """
    def __init__(self, n_batch: int = 0, pad_idx: int = 0):
        """
        Args:
            n_batch (int): number of batches.
            pad_idx (int): the padding token index in vocabulary.
        """
        self.loss = 0.
        self.n_words = 0.
        self.n_correct = 0.
        self.start_time = time.time()
        self.pad_idx = pad_idx
        self.n_batch = n_batch

    def update(self, loss: float, probs: torch.FloatTensor, target: torch.LongTensor):
        """
        Args:
            loss (FloatTensor): the loss computed by the loss criterion.
            probs (FloatTensor): the generated probs of the model.
            target (LongTensor): true targets.
        """
        pred = probs.max(1)[1]  # predicted targets
        non_padding = target.ne(self.pad_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()

        self.loss += loss
        self.n_words += non_padding.sum().item()
        self.n_correct += num_correct

    def avg_loss(self) -> float:
        """
        Calculate the average loss value by batch.
        """
        return self.loss/self.n_batch

    def accuracy(self) -> float:
        """
        Calculate the accuracy.
        """
        return 100 * (self.n_correct / self.n_words)

    def ppl(self) -> float:
        """
        Calculate the perplexity.
        """
        return math.exp(self.avg_loss())

    def elapsed_time(self) -> float:
        """
        Caculate the cost time.
        """
        return round((time.time() - self.start_time) / 60, 4)


class NMTTrainer(object):
    def __init__(self, config, model, criterion, optimizer, pad_idx):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.pad_idx = pad_idx
        self.training_step = 0
        # Statistics
        self.historys = {'AVG_TRAIN_LOSSES': [],
                         'AVG_EVAL_LOSSES': [],
                         'TRAIN_ACCS': [],
                         'EVAL_ACCS': [],
                         'TRAIN_PPLS': [],
                         'EVAL_PPLS': [],
                         'TRAIN_EPOCH_TIMES': [],
                         'EVAL_EPOCH_TIMES': []}

        logger.info(f'{"#" * 15}Training Parameter{"#" * 15}')
        logger.info(f'Vocabulary size: {config.vocab_size:,}')
        logger.info(f'Num of epochs: {config.epochs}')
        logger.info(f'Learning rate: {config.learning_rate}')
        logger.info(f'Batch size: {config.batch_size}')
        logger.info(f'Device: {config.device}')
        logger.info(f'Early stop: {config.early_stop}')
        logger.info(f"Save dir: '{config.save_dir}'")
        logger.info(f'{"=" * 40}')
        logger.info('\n')

    def feed_data(self, data_iter, stats, is_train: bool = True):
        data_iter.init_epoch()
        tqdm_bar = tqdm(enumerate(data_iter), total=len(data_iter), desc='TRAIN' if is_train else 'EVAL')
        for i, batch in tqdm_bar:
            self.training_step += 1
            src, src_length = batch.src
            tgt, tgt_length = batch.tgt
            probs = self.model(src, src_length, tgt[:, :-1])
            probs = probs.contiguous().view(probs.size(0) * probs.size(1), -1)
            tgt_labels = tgt[:, 1:].contiguous().view(-1)

            loss = self.criterion(probs, tgt_labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            batch_loss = loss.item()
            stats.update(batch_loss, probs.data, tgt_labels.data)
            self.optimizer.zero_grad()
        return stats

    def print_stats(self, stats: Statistics, mode: str = 'Train'):
        assert mode in ['TRAIN', 'EVAL'], \
            f'ERROR | print_stats(): Mode must in ["TRAIN", "EVAL"] !!!'

        logger.info(f'\t{mode}\t| AVG LOSS: {stats.avg_loss()} |\tACC: {stats.accuracy():.4f}% |'
                    f'\t Perplexity: {stats.ppl():.4f} |\tEPOCH TIME: {stats.elapsed_time()} minutes|'
                    f'\tlr: {self.optimizer.param_groups[0]["lr"]}')

    def optimize(self, train_iter):
        torch.set_grad_enabled(True)
        self.model.train()
        self.model.zero_grad()
        stats = Statistics(n_batch=len(train_iter), pad_idx=self.pad_idx)
        self.feed_data(train_iter, stats, is_train=True)
        self.historys['AVG_TRAIN_LOSSES'].append(stats.avg_loss())
        self.historys['TRAIN_EPOCH_TIMES'].append(stats.elapsed_time())
        self.historys['TRAIN_ACCS'].append(stats.accuracy())
        self.historys['TRAIN_PPLS'].append(stats.ppl())
        return stats

    def evaluate(self, eval_iter, epoch: int):
        torch.set_grad_enabled(False)
        stats = Statistics(n_batch=len(eval_iter), pad_idx=self.pad_idx)
        self.model.train()
        self.model.zero_grad()
        stats = Statistics(n_batch=len(eval_iter), pad_idx=self.pad_idx)
        self.feed_data(eval_iter, stats, is_train=False)
        self.historys['AVG_EVAL_LOSSES'].append(stats.avg_loss())
        self.historys['EVAL_EPOCH_TIMES'].append(stats.elapsed_time())
        self.historys['EVAL_ACCS'].append(stats.accuracy())
        self.historys['EVAL_PPLS'].append(stats.ppl())
        return stats

