from translators import Config
from translators.model_builder import build_model

if __name__ == "__main__":
    cnf = Config("examples/GNMT_Config.yaml", "GNMT")
    model = build_model(cnf)
