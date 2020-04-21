import os
import torch.optim as optim

from translators import Config
from translators.cores.functions import plot_figure, save_train_history
from translators.models import save_checkpoint
from translators.model_builder import build_model, build_tokenizer, build_dataset, build_criterion
from translators.bin import NMTTrainer
from translators.logger import logger, init_logger

if __name__ == "__main__":
    logger.info('Reading configuation file ...')
    cnf = Config("examples/GNMT_Config.yaml", "GNMT")
    init_logger(log_file=os.path.join(cnf.save_dir, "train_nmt_model.log"), log_file_level='INFO')
    tokenizer = build_tokenizer(cnf)
    nmtmodel = build_model(cnf)
    dataset, fields = build_dataset(cnf, tokenizer)
    criterion = build_criterion(cnf.vocab_size, tokenizer.vocab.stoi[tokenizer.pad_token], cnf.device)
    optimizer = optim.Adam(nmtmodel.parameters(), lr=cnf.learning_rate)
    trainer = NMTTrainer(config=cnf, model=nmtmodel, criterion=criterion, optimizer=optimizer, pad_idx=fields[0].vocab.stoi[fields[0].pad_token])

    train_iter = dataset['train'].iter_dataset()
    eval_iter = dataset['eval'].iter_dataset()

    best_eval_loss = float('inf')
    best_epoch = 0
    best_training_step = 0
    bad_loss_count = 0
    total_epoch = 0
    try:
        for epoch in range(cnf.epochs):
            total_epoch += 1
            logger.info(f'Starting epoch {epoch}:')
            train_stats = trainer.optimize(train_iter)
            eval_stats = trainer.evaluate(eval_iter, epoch)
            trainer.print_stats(train_stats, mode='TRAIN')
            trainer.print_stats(eval_stats, mode='EVAL')
            if eval_stats.avg_loss() < best_eval_loss:
                save_checkpoint(cnf=cnf, model=nmtmodel, optimizer=optimizer, fields=fields, epoch=epoch)
                bad_loss_count = 0
                best_epoch = epoch
                best_training_step = trainer.training_step
                best_eval_loss = eval_stats.avg_loss()
                continue
            bad_loss_count += 1
            if not cnf.early_stop == 0 and bad_loss_count > cnf.early_stop:
                logger.info("Early stopping !!!")
                break
    except KeyboardInterrupt:
        logger.info(f"Performed {total_epoch} iterations and {trainer.training_step:,} steps")
        logger.info(f"Achieve the best model at the {best_epoch}st iteration and and the {best_training_step:,}th step")
        logger.info(f"The lowest EVAL Loss is {best_eval_loss}.")
        plot_figure(cnf.save_dir, trainer.historys)
        save_train_history(cnf.save_dir, trainer.historys)
    except Exception as e:
        print(e)
    logger.info(f"Performed {total_epoch} iterations and {trainer.training_step:,} steps")
    logger.info(f"Achieve the best model at the {best_epoch}st iteration and and the {best_training_step:,}th step")
    logger.info(f"The lowest EVAL Loss is {best_eval_loss}.")
    plot_figure(cnf.save_dir, trainer.historys)
    save_train_history(cnf.save_dir, trainer.historys)


