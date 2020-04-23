import stanza
import re
from tqdm import tqdm
from multiprocessing import Pool

NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
def string_detokenize(text, feat):
    text = text.strip()
    detok = ''
    index = -1
    is_subword = False
    for token in text.split():
        if "@@" in token:
            if is_subword:
                detok += f"{token}|{feat[index]} "
            else:
                index += 1
                detok += f"{token}|{feat[index]} "
                is_subword = True
        else:
            if is_subword:
                detok += f"{token}|{feat[index]} "
                is_subword = False
            else:
                index += 1
                detok += f"{token}|{feat[index]} "
    return detok


def get_feature(text):
    raw_text, bpe_text, tgt_text = text
    if len(raw_text.strip())==0:
        return None
    doc = NLP(raw_text)
    deprel = [word.deprel for sent in doc.sentences for word in sent.words]
    detok = string_detokenize(bpe_text, deprel)
    return detok, tgt_text


if __name__ == "__main__":
    with open("datasets/bpe_data/dev.BPE.10000.en", 'r', encoding='utf-8') as fsrc, \
            open("datasets/bpe_data/dev.BPE.10000.vi", 'r', encoding='utf-8') as ftgt,\
            open("datasets/raw_data/dev.en", 'r', encoding='utf-8') as fraw,\
            open("datasets/feat_data/dev.en", 'w', encoding='utf-8') as fen,\
            open("datasets/feat_data/dev.vi", 'w', encoding='utf-8') as fvi:
            counters= {}
            srcs = fsrc.readlines()
            tgts = ftgt.readlines()
            raw_sents = fraw.readlines()
            with Pool(4) as p:
                for sub_counter in tqdm(p.imap(get_feature, list(zip(raw_sents, srcs, tgts))), total=len(raw_sents)):
                    if sub_counter:
                        fen.write(f"{sub_counter[0].strip()}\n")
                        fvi.write(f"{sub_counter[1].strip()}\n")
            fen.close()
            fvi.close()
