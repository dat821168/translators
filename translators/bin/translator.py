import os
import torch
import sacrebleu
import stanza

from nltk import word_tokenize
from translators.model_builder import build_generator
from translators.logger import logger
from tqdm import tqdm
from subword_nmt import apply_bpe


NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')


class Translator(object):
    def __init__(self, config, model, beam_size, max_length, save_dir, metric, tokenizer, device, sos_idx, eos_idx):
        self.config = config
        self.model = model
        self.beam_size = beam_size
        self.max_lenght = max_length
        self.save_dir = save_dir
        self.cuda = 'cuda' in device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.metric = metric
        self.tokenizer = tokenizer
        self.generator = build_generator(model=model, beam_size=beam_size, max_seq_len=max_length, cuda=self.cuda,
                                         sos_idx=self.sos_idx, eos_idx=self.eos_idx)
        if config.bpe_codes_file is not None:
            with open(config.bpe_codes_file, encoding='UTF-8') as codes:
                self.bpe = apply_bpe.BPE(codes=codes)

    def segment_tokens(self, tokens, features=[], dropout=0):
        """
            Overwrite segment_tokens function of subword_nmt for apply linguistics features.
            Segment a sequence of tokens with BPE encoding.
        """
        output = []
        out_feats = []
        if len(features) == 0:
            features = [""]*len(tokens)
        for word, feat in list(zip(tokens, features)):
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self.bpe._isolate_glossaries(word)
                        for out in apply_bpe.encode(segment,
                                          self.bpe.bpe_codes,
                                          self.bpe.bpe_codes_reverse,
                                          self.bpe.vocab,
                                          self.bpe.separator,
                                          self.bpe.version,
                                          self.bpe.cache,
                                          self.bpe.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.bpe.separator)
            output.append(new_word[-1])
            out_feats.extend([feat]*len(new_word))
        return output, out_feats

    def encode_ids(self, src, feat):
        src_ids = [self.tokenizer.vocab.stoi[token] for token in src]
        feat_ids = [self.tokenizer.feat_vocabs["deprel"].stoi[f] for f in feat]

        src_tensor = torch.LongTensor([src_ids]).cuda() if self.cuda else torch.LongTensor([src_ids])
        feat_tensor = torch.LongTensor([feat_ids]).cuda() if self.cuda else torch.LongTensor([feat_ids])
        src_len = torch.LongTensor([len(src_ids)]).cuda() if self.cuda else torch.LongTensor([len(src_ids)])

        return src_tensor, feat_tensor, src_len

    def preprocess(self, text):
        seg_words = word_tokenize(text)
        seg_text = " ".join(seg_words)
        deprel = [word.deprel for sent in NLP(seg_text).sentences for word in sent.words]
        src, feat = self.segment_tokens(seg_words, deprel)
        src = [self.tokenizer.sos_token] + src + [self.tokenizer.eos_token]
        feat = [self.tokenizer.sos_token] + feat + [self.tokenizer.eos_token]
        return src, feat

    def translate(self, text):
        src_tok, feat = self.preprocess(text)
        batch = self.encode_ids(src_tok, feat)
        tgt_text, tgt_tok, att_scores = self.__translate(batch)
        return tgt_text, src_tok, feat, tgt_tok, att_scores

    def __translate(self, batch):
        self.model.eval()
        src_tensor, feat_tensor, src_len = batch
        feats = [("deprel", feat_tensor)]
        batch_size = 1
        beam_size = self.beam_size
        bos = [[self.sos_idx]] * (batch_size * beam_size)
        bos = torch.LongTensor(bos).to(src_tensor.device)
        bos = bos.view(-1, 1)
        with torch.no_grad():
            context = self.model.encode(src_tensor, src_len, feats)
            context = [context, src_len, None]
            if beam_size == 1:
                generator = self.generator.greedy_search
            else:
                generator = self.generator.beam_search
            preds, lengths, counter, att_score = generator(batch_size, bos, context)
        return self.tokenizer.detokenize(preds[0]), \
               self.tokenizer.detokenize(preds[0], keep_subword=True), \
               att_score[0][1:lengths[0].item()]

    def evaluate(self, data_iter):
        logger.info("Start translate ...")
        self.model.eval()
        predicts, golds, srcs = self.__evaluate(data_iter)
        # score = self.metric.corpus_bleu(golds, predicts, [1,0])
        # self.write_predicts(predicts, golds, srcs)
        bleu = sacrebleu.corpus_bleu(predicts, [golds], tokenize='13a', force=True)
        logger.info(f"The model achieved {bleu.score} BLEU score on the TEST set.")
        return bleu.score

    def __evaluate(self, data_iter):
        """
        Runs evaluation on test dataset.
        """
        srcs = []
        predicts = []
        golds = []
        tqdm_bar = tqdm(enumerate(data_iter), total=len(data_iter), desc='TEST')
        for i, batch in tqdm_bar:
            src, src_length = batch.src
            feats = [(feat_name, getattr(batch, feat_name)) for feat_name in list(self.config.feat_vocab_sizes.keys())]
            batch_size = src.size(0)
            beam_size = self.beam_size
            bos = [[self.sos_idx]] * (batch_size * beam_size)
            bos = torch.LongTensor(bos).to(src.device)
            bos = bos.view(-1, 1)
            with torch.no_grad():
                context = self.model.encode(src, src_length, feats)
                context = [context, src_length, None]
                if beam_size == 1:
                    generator = self.generator.greedy_search
                else:
                    generator = self.generator.beam_search
                preds, lengths, counter = generator(batch_size, bos, context)
            for pred, tgt, raw in list(zip(preds, batch.tgt, src)):
                pred = pred.tolist()
                detok = self.tokenizer.detokenize(pred)
                src_detok = self.tokenizer.detokenize(raw.long().tolist())
                predicts.append(detok)
                golds.append(" ".join(tgt))
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


