from translators import Config
from translators.models import load_chkpt
from translators.model_builder import build_model, build_tokenizer
from translators.bin import Translator
from translators.cores.functions.common import showAttention, make_table

if __name__ == "__main__":
    cnf = Config("examples/GNMT/GNMT_Config_translate.yaml", "GNMT")
    epoch, model_chkpt, _, vocab = load_chkpt(chkpt_file=cnf.chkpt_file, device=cnf.device)
    tokenizer = build_tokenizer(cnf, vocab)
    nmtmodel = build_model(cnf)
    nmtmodel.load_state_dict(model_chkpt, strict=False)

    sos_idx = vocab['tokens'].stoi[tokenizer.sos_token]
    eos_idx = vocab['tokens'].stoi[tokenizer.eos_token]

    translator = Translator(config=cnf,
                            model=nmtmodel,
                            beam_size=cnf.beam_size,
                            max_length=cnf.max_length,
                            save_dir=None,
                            metric=None,
                            tokenizer=tokenizer,
                            device=cnf.device,
                            sos_idx=sos_idx,
                            eos_idx=eos_idx)

    while True:
        src_text = input(">>> ")
        try:
            tgt_text, src_tok, feat, tgt_tok, att_scores = translator.translate(src_text)
            if len(src_tok) < 20:
                make_table(src_text, src_tok[1:-1], feat[1:-1], tgt_text)
            else:
                sep = "\t|"
                print("=" * 50)
                print(f"src text:    | {src_text}")
                print(f'src subword: | {sep.join(src_tok[1:-1])}')
                print(f"src feat:    | {sep.join(feat[1:-1])}")
                print(f"tgt text:    | {tgt_text}")
                print("="*50)
            showAttention(src_tok, tgt_tok, att_scores)
        except Exception as e:
            print(e)
