import os
import time
import torch.optim as optim

from translators import Config
from translators.cores.functions import plot_figure, save_train_history
from translators.model_builder import build_model, build_tokenizer, build_dataset, build_criterion
from translators.bin import NMTTrainer
from translators.logger import logger, init_logger

if __name__ == "__main__":
    logger.info('Reading configuation file ...')
    cnf = Config("examples/GNMT_Config.yaml", "GNMT")
    init_logger(log_file=os.path.join(cnf.save_dir, "train_nmt_model.log"), log_file_level='INFO')
    # Load tokenizer
    tokenizer = build_tokenizer(cnf)
    # Read dataset
    dataset = build_dataset(cnf, tokenizer)
    # Build NMT model
    nmtmodel = build_model(cnf)
    criterion = build_criterion(cnf.vocab_size, tokenizer.vocab.stoi[tokenizer.pad_token], cnf.device)
    optimizer = optim.Adam(nmtmodel.parameters(), lr=cnf.learning_rate)

    pad_idx = tokenizer.vocab.stoi[tokenizer.pad_token]
    trainer = NMTTrainer(config=cnf, model=nmtmodel, tokenizer=tokenizer,
                         criterion=criterion, optimizer=optimizer, pad_idx=pad_idx)

    train_iter = dataset['train'].iter_dataset()
    eval_iter = dataset['eval'].iter_dataset()

    total_time = time.time()
    try:
        trainer.train(train_iter, eval_iter)
    except KeyboardInterrupt:
        total_time = round((time.time() - total_time)/60, 4)
        logger.info(f'{"=" * 40}')
        logger.info('\n')
        logger.info(f"Performed {trainer.training_step:,} steps in a total of {total_time} minutes")
        logger.info(f"Achieve the best model at the {trainer.best_step:,}th step")
        logger.info(f"The lowest EVAL Loss is {trainer.best_eval_loss}.")
        save_train_history(cnf.save_dir, trainer.historys)
        logger.info(f'{"=" * 40}')
        logger.info('\n')
        plot_figure(cnf.save_dir, trainer.historys)
    except Exception as e:
        print(e)

    total_time = round((time.time() - total_time) / 60, 4)
    logger.info(f'{"=" * 40}')
    logger.info('\n')
    logger.info(f"Performed {trainer.training_step:,} steps in a total of {total_time} minutes")
    logger.info(f"Achieve the best model at the {trainer.best_step:,}th step")
    logger.info(f"The lowest EVAL Loss is {trainer.best_eval_loss}.")
    save_train_history(cnf.save_dir, trainer.historys)
    logger.info(f'{"=" * 40}')
    logger.info('\n')
    plot_figure(cnf.save_dir, trainer.historys)


