from translators.configuration import Config
from translators.models import NMTModel, GNMT


def build_model(config: Config) -> NMTModel:
    if config.model_type == "GNMT":
        model = GNMT(config)
    else:
        model = None
    return model


def build_examples(config: Config):
    raise NotImplementedError