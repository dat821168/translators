import time
import math
import torch


from translators.models import save_checkpoint
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

    def reset(self, loss: float, probs: torch.FloatTensor, target: torch.LongTensor):
        """
        Args:
            loss (FloatTensor): the loss computed by the loss criterion.
            probs (FloatTensor): the generated probs of the model.
            target (LongTensor): true targets.
        """
        pred = probs.max(1)[1]  # predicted targets
        non_padding = target.ne(self.pad_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()

        self.loss = loss
        self.n_words = non_padding.sum().item()
        self.n_correct = num_correct

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
    def __init__(self, config, model, tokenizer, criterion, optimizer, pad_idx):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.pad_idx = pad_idx
        self.training_step = 0
        self.bad_step = (float('inf'), 0)
        self.early_stop = False
        # Statistics
        self.historys = {'TRAIN_STEP_LOSSES': [],
                         'AVG_EVAL_LOSSES': [],
                         'TRAIN_STEP_ACCS': [],
                         'EVAL_ACCS': [],
                         'TRAIN_STEP_PPLS': [],
                         'EVAL_PPLS': []}
        self.tqdm_bar = None
        self.best_step = 0
        self.best_eval_loss = float('inf')

        logger.info(f'{"#" * 15}Training Parameter{"#" * 15}')
        logger.info(f'Vocabulary size: {config.vocab_size:,}')
        logger.info(f'Num of steps: {config.train_steps}')
        logger.info(f'Eval steps: {config.eval_steps}')
        logger.info(f'Save steps: {config.save_checkpoint_steps}')
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
        for i, batch in enumerate(data_iter):
            self.training_step += 1
            src, src_length = batch.src
            tgt, tgt_length = batch.tgt
            feats = [(feat_name, getattr(batch, feat_name)) for feat_name in list(self.config.feat_vocab_sizes.keys())]
            probs = self.model(src, src_length, tgt[:, :-1], feats)
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

    def optimize(self, train_iter, eval_iter):
        torch.set_grad_enabled(True)
        self.model.train()
        self.model.zero_grad()
        stats = Statistics(n_batch=len(train_iter), pad_idx=self.pad_idx)
        train_iter.init_epoch()
        for i, batch in enumerate(train_iter):
            self.tqdm_bar.update()
            self.training_step += 1
            src, src_length = batch.src
            tgt, tgt_length = batch.tgt
            feats = [(feat_name, getattr(batch, feat_name)) for feat_name in list(self.config.feat_vocab_sizes.keys())]
            probs = self.model(src, src_length, tgt[:, :-1], feats)
            probs = probs.contiguous().view(probs.size(0) * probs.size(1), -1)
            tgt_labels = tgt[:, 1:].contiguous().view(-1)
            loss = self.criterion(probs, tgt_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            batch_loss = loss.item()/src.size(0)
            stats.reset(batch_loss, probs.data, tgt_labels.data)
            self.optimizer.zero_grad()

            self.tqdm_bar.desc = f"LOSS: {batch_loss:.4f}| ACC: {stats.accuracy():.4f}| PPL: {stats.ppl():.4f}| " \
                                 f"Lr: {self.optimizer.param_groups[0]['lr']}| Bad step: {self.bad_step[-1]}"
            self.historys['TRAIN_STEP_LOSSES'].append(batch_loss)
            self.historys['TRAIN_STEP_ACCS'].append(stats.accuracy())
            self.historys['TRAIN_STEP_PPLS'].append(stats.ppl())

            if self.training_step % self.config.eval_steps == 0:
                self.evaluate(eval_iter)
                torch.set_grad_enabled(True)
                self.model.train()
            if self.training_step % self.config.save_checkpoint_steps == 0:
                save_checkpoint(cnf=self.config, model=self.model, optimizer=self.optimizer,
                                tokenizer=self.tokenizer, step=self.training_step)
            if batch_loss < self.bad_step[0]:
                self.bad_step = (batch_loss, 0)
            else:
                self.bad_step = (batch_loss, self.bad_step[-1]+1)
                if 0 < self.config.early_stop == self.bad_step[-1]:
                    self.early_stop = True
                    break
            if self.training_step == self.config.train_steps:
                break

    def evaluate(self, eval_iter):
        torch.set_grad_enabled(False)
        self.model.eval()
        self.model.zero_grad()
        stats = Statistics(n_batch=len(eval_iter), pad_idx=self.pad_idx)
        eval_iter.init_epoch()
        tqdm_bar = tqdm(enumerate(eval_iter), total=len(eval_iter), desc='EVAL',  position=0, leave=True)
        for i, batch in tqdm_bar:
            src, src_length = batch.src
            tgt, tgt_length = batch.tgt
            feats = [(feat_name, getattr(batch, feat_name)) for feat_name in list(self.config.feat_vocab_sizes.keys())]
            probs = self.model(src, src_length, tgt[:, :-1], feats)
            probs = probs.contiguous().view(probs.size(0) * probs.size(1), -1)
            tgt_labels = tgt[:, 1:].contiguous().view(-1)
            loss = self.criterion(probs, tgt_labels)
            batch_loss = loss.item()
            stats.update(batch_loss, probs.data, tgt_labels.data)
            self.optimizer.zero_grad()
        eval_loss = stats.avg_loss()
        if eval_loss < self.best_eval_loss:
            self.best_step = self.training_step
            self.best_eval_loss = eval_loss
        self.historys['AVG_EVAL_LOSSES'].append(eval_loss)
        self.historys['EVAL_ACCS'].append(stats.accuracy())
        self.historys['EVAL_PPLS'].append(stats.ppl())
        self.print_stats(stats, mode='EVAL')

    def train(self, train_iter, eval_iter):
        self.tqdm_bar = tqdm(total=self.config.train_steps)
        while self.training_step < self.config.train_steps:
            self.optimize(train_iter, eval_iter)
            if self.early_stop:
                logger.info("Early stopping !!!")
                break