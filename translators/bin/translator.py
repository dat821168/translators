import os
import torch
import sacrebleu

from translators.model_builder import build_generator
from translators.logger import logger
from tqdm import tqdm


class Translator(object):
    def __init__(self, model, beam_size, max_length, save_dir, metric, device, sos_idx, eos_idx):
        self.model = model
        self.beam_size = beam_size
        self.max_lenght = max_length
        self.save_dir = save_dir
        self.cuda = 'cuda' in device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.metric = metric
        self.generator = build_generator(model=model, beam_size=beam_size, max_seq_len=max_length, cuda=self.cuda,
                                         sos_idx=self.sos_idx, eos_idx=self.eos_idx)

    def translate(self, data_iter, tokenizer):
        logger.info("Start translate ...")
        self.model.eval()
        predicts, golds, srcs = self.evaluate(data_iter, tokenizer)
        # score = self.metric.corpus_bleu(golds, predicts, [1,0])
        # self.write_predicts(predicts, golds, srcs)
        bleu = sacrebleu.corpus_bleu(predicts, [golds], tokenize='13a', force=True)
        logger.info(f"The model achieved {bleu.score} BLEU score on the TEST set.")
        return bleu.score

    def evaluate(self, data_iter, tokenizer):
        """
        Runs evaluation on test dataset.
        """
        srcs = []
        predicts = []
        golds = []
        tqdm_bar = tqdm(enumerate(data_iter), total=len(data_iter), desc='TEST')
        for i, (src, tgts) in tqdm_bar:
            src, src_length = src
            batch_size = src.size(0)
            beam_size = self.beam_size
            bos = [[self.sos_idx]] * (batch_size * beam_size)
            bos = torch.LongTensor(bos)
            bos = bos.view(-1, 1)
            if self.cuda:
                src = src.cuda()
                src_length = src_length.cuda()
                bos = bos.cuda()
            with torch.no_grad():
                context = self.model.encode(src, src_length)
                context = [context, src_length, None]
                if beam_size == 1:
                    generator = self.generator.greedy_search
                else:
                    generator = self.generator.beam_search
                preds, lengths, counter = generator(batch_size, bos, context)
            for pred, tgt, raw in list(zip(preds, tgts, src)):
                pred = pred.tolist()
                detok = tokenizer.detokenize(pred)
                src_detok = tokenizer.detokenize(raw.long().tolist())
                predicts.append(detok)
                golds.append(tgt)
                srcs.append([src_detok])
        return predicts, golds, srcs

    def write_predicts(self, predicts, golds, srcs):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        scr_file_path = os.path.join(self.save_dir, 'src.en')
        ref_file_path = os.path.join(self.save_dir, 'ref.vi')
        predic_file_path = os.path.join(self.save_dir, 'predict.vi')
        with open(predic_file_path, 'w', encoding='utf-8') as writer, \
                open(ref_file_path, 'w', encoding='utf-8') as ref_writer, \
                open(scr_file_path, 'w', encoding='utf-8') as src_writer:
            for pred, gold, src in list(zip(predicts, golds, srcs)):
                writer.write(f'{" ".join(pred)}\n')
                ref_writer.write(f'{" ".join(gold)}\n')
                src_writer.write(f'{src}\n')
            writer.close()


