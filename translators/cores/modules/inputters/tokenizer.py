from bpemb import BPEmb


class tokenizer(object):
    def __init__(self, config, lang='en'):
        super(tokenizer, self).__init__()

        self.config = config
        self.vocab = None

    def segment(self, line):
        raise NotImplementedError

    def tokenize(self, line):
        raise NotImplementedError

    def detokenize(self, line):
        raise NotImplementedError


class WordTokenizer(tokenizer):
    def __init__(self, lang):
        super(tokenizer, self).__init__()
        if lang == "en":
            self.tokenizer = None
        elif lang == "vi":
            self.tokenizer = None

    def segment(self, line):
        raise NotImplementedError

    def tokenize(self, line):
        raise NotImplementedError

    def detokenize(self, line):
        raise NotImplementedError


class BPETokenizer(tokenizer):
    def __init__(self):
        super(tokenizer, self).__init__()
        self.bpe = BPEmb(lang="multi", vs=100000, vs_fallback=False)

    def segment(self, line):
        return self.bpe.encode_with_bos_eos(line)

    def tokenize(self, line):
        return self.segment(line)

    def detokenize(self, line):
        raise self.bpe.decode()


class SubwordTokenizer(tokenizer):
    def __init__(self):
        super(tokenizer, self).__init__()

    def segment(self, line):
        raise NotImplementedError

    def tokenize(self, line):
        raise NotImplementedError

    def detokenize(self, line):
        raise NotImplementedError