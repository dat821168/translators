import os
import re
import stanza


from collections import Counter, OrderedDict
from torchtext.vocab import Vocab

stanza.download('en')


class Tokenizer(object):
    def __init__(self, vocab: str = None, features: list = ['deprel'], separator: str = '@@',
                 pad_token: str = '<pad>', unk_token: str = '<unk>',
                 sos_token: str = '<sos>', eos_token: str = '<eos>'):

        # Special tokens:
        self.separator = separator
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.features = features
        self.nlp = stanza.Pipeline(lang='en')
        if isinstance(vocab, str):
            assert vocab is not None and os.path.exists(vocab), f"Error: Vocab file {vocab} not exits!!!"
            self.feat_vocabs = {}
            self.__get_feats_vocabs()
            self.vocab: Vocab = self.__build_vocab(vocab)
        else:
            self.vocab: Vocab = vocab['tokens']
            self.feat_vocabs = vocab['feats']

    def __len__(self):
        return len(self.vocab)

    def __get_feats_vocabs(self):
        if 'deprel' in self.features:
            counter = Counter(self.nlp.processors['depparse'].vocab._vocabs['deprel']._id2unit)
            self.feat_vocabs['deprel'] = Vocab(counter=counter, specials=[])
        self.features = list(self.feat_vocabs.keys())

    def __build_vocab(self, vocab_file: str) -> Vocab:
        specials = list(OrderedDict.fromkeys(tok for tok in [self.pad_token, self.unk_token,
                                                             self.sos_token, self.eos_token]))
        counter = Counter()
        with open(vocab_file, "r", encoding="utf-8") as f:
            while True:
                token = f.readline()
                if not token:
                    break
                counter.update([token.strip()])
            f.close()
        return Vocab(counter=counter, specials=specials)

    def __string_detokenize(self, text):
        text = text.strip()
        detok = ''
        mask = []
        index = -1
        is_subword = False
        for token in text.split():
            if self.separator in token:
                detok += re.sub(self.separator, '', token)
                if is_subword:
                    mask.append(index)
                else:
                    index += 1
                    mask.append(index)
                    is_subword = True
            else:
                detok += f'{token} '
                if is_subword:
                    mask.append(index)
                    is_subword = False
                else:
                    index += 1
                    mask.append(index)
        return detok.strip(), mask

    def get_feature(self, text) -> dict:
        text_detok, mask = self.__string_detokenize(text)
        sent_feats = {}
        if 'deprel' in self.features:
            doc = self.nlp(text_detok)
            deprel = [word.deprel for sent in doc.sentences for word in sent.words]
            feature = []
            for index in mask:
                feature.append(deprel[index])
            assert len(feature) == len(text.split())
            sent_feats['deprel'] = feature

        return sent_feats

    def tokenize(self, sent: str) -> list:
        tokens = sent.strip().split()
        return tokens

    def detokenize(self, idxs: list, delim: str = ' ') -> str:
        detok = delim.join([self.vocab.itos[idx] for idx in idxs])
        detok = re.sub(self.separator + ' ', '', detok)
        detok = re.sub(self.separator, '', detok)

        detok = re.sub(self.sos_token, '', detok)
        detok = re.sub(self.eos_token, '', detok)
        detok = re.sub(self.pad_token, '', detok)
        detok = detok.strip()
        return detok


if __name__ == "__main__":
    tok = Tokenizer(vocab="E:/Projects/Cores/translators/datasets/bpe_data/vocab.bpe.10000")
    tok.get_feature("Bar@@ ack Ob@@ am@@ a was born in Ha@@ waii@@ .  He was elected president in 2008@@ .")