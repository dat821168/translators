import yaml

import torch

MODEL_TYPES = ['GNMT', 'Seq2Seq']


class Config(object):
    def __init__(self, config_path: str, model_type: str):
        # Verify model type
        assert model_type in MODEL_TYPES, f'{model_type} not supported, please chose in '

        self.model_type: str = model_type

        # Paths
        self.config: str = config_path
        self.chkpt_file = None

        self.train_src_file = None
        self.train_tgt_file = None

        self.dev_src_file = None
        self.dev_tgt_file = None

        self.test_src_file = None
        self.test_tgt_file = None

        self.vocab_file: str = None
        self.save_dir: str = "outputs/"

        self.bpe_codes_file: str = None
        # Embedder
        self.vocab_size: int = None

        self.feats: list = []
        self.feat_vocab_sizes = {}
        # Special token
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '<\s>'

        self.min_len = 1
        self.max_len = 256

        self.PAD_IDX, self.UNK_IDX, self.SOS_IDX, self.EOS_IDX = [0, 1, 2, 3]

        self.early_stop = 0

        # Read config file
        self.build_from_file()

        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_from_file(self):
        with open(self.config, 'r', encoding='utf-8') as f:
            try:
                pairs = yaml.safe_load(f)
                self.__dict__.update(pairs)
            except Exception as exc:
                raise exc
