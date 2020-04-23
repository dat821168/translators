import os
import nltk

from translators import Config
from translators.models import load_chkpt
from translators.model_builder import build_model, build_tokenizer, build_dataset
from translators.bin import Translator
from translators.logger import logger, init_logger
from translators.cores.functions import BLUE

if __name__ == "__main__":
    logger.info('Reading configuation file ...')
    cnf = Config("examples/GNMT_Config_translate.yaml", "GNMT")
    init_logger(log_file=os.path.join(cnf.save_dir, "train_nmt_model_translate.log"), log_file_level='INFO')
    epoch, model_chkpt, _, vocab = load_chkpt(chkpt_file=cnf.chkpt_file, device=cnf.device)
    tokenizer = build_tokenizer(cnf, vocab)
    nmtmodel = build_model(cnf)
    nmtmodel.load_state_dict(model_chkpt, strict=False)
    dataset = build_dataset(cnf, tokenizer)

    blue_score = BLUE(ngrams=1)
    # blue_score = nltk.translate.bleu_score

    sos_idx = vocab['tokens'].stoi[tokenizer.sos_token]
    eos_idx = vocab['tokens'].stoi[tokenizer.eos_token]

    translator = Translator(config=cnf,
                            model=nmtmodel,
                            beam_size=cnf.beam_size,
                            max_length=cnf.max_length,
                            save_dir=cnf.save_dir,
                            metric=blue_score,
                            device=cnf.device,
                            sos_idx=sos_idx,
                            eos_idx=eos_idx)
    test_iter = dataset['test'].iter_dataset()

    best_eval_loss = float('inf')
    bad_loss_count = 0
    blue_score = translator.translate(test_iter, tokenizer)
    try:
        blue_score = translator.translate(test_iter, tokenizer)
    except KeyboardInterrupt as e:
         print(e)
    except Exception as e:
         print(e)



