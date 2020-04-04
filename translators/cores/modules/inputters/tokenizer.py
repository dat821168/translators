from bpemb import BPEmb


class tokenizer(object):
    def __init__(self, config):
        super(tokenizer, self).__init__()

        self.config = config
        self.vocab = None

    def segment(self, line):
        raise NotImplementedError

    def tokenize(self, line):
        raise NotImplementedError

    def detokenize(self, line):
        raise NotImplementedError


class BPETokenizer(tokenizer):
    def __init__(self):
        super(tokenizer, self).__init__()
        self.bpe = BPEmb(lang="multi", vs=100000)

    def segment(self, line):
        raise NotImplementedError

    def tokenize(self, line):
        raise NotImplementedError

    def detokenize(self, line):
        raise NotImplementedError


class SubwordTokenizer(tokenizer):
    def __init__(self):
        super(tokenizer, self).__init__()

    def segment(self, line):
        raise NotImplementedError

    def tokenize(self, line):
        raise NotImplementedError

    def detokenize(self, line):
        raise NotImplementedError