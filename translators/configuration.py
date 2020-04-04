import yaml

MODEL_TYPES = ['GNMT']


class Config(object):
    def __init__(self, config_path: str, model_type: str):

        assert model_type in MODEL_TYPES, f'{model_type} not supported, please chose in '
        self.model_type: str = model_type

        self.config: str = config_path
        self.corpus: str = None

        self.vocab_size: int = None

        self.build_from_file()

    def build_from_file(self):
        with open(self.config, 'r', encoding='utf-8') as f:
            try:
                pairs = yaml.safe_load(f)
                self.__dict__.update(pairs)
            except Exception as exc:
                raise exc
