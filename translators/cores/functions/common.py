import os
import json
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from translators.logger import logger


def make_table(src_text, src_tok, feat, tgt_text):
    table_str = ''
    # dynamically decide column widths
    lens = [len(str(tok)) for tok in src_tok]
    lens += [len(str(f)) for f in feat]
    column_widths = max(lens)+1
    table_width = (column_widths*len(src_tok))+13
    table_str += '=' * table_width + '\n'
    print(table_str)
    src_text_row = f'| SRC text: | {src_text}'.ljust(table_width) + '|\n'
    table_str += src_text_row
    print(table_str)
    tgt_text_row = f'| TGT text: | {tgt_text}'.ljust(table_width) + '|\n'
    table_str += tgt_text_row
    print(table_str)

    #
    # table_str += '|'
    # for i, item in enumerate(header):
    #     table_str += ' ' + str(item).ljust(column_widths[i] - 2) + '|'
    # table_str += '\n'
    #
    # table_str += '-' * (sum(column_widths) + 1) + '\n'
    #
    # for line in content:
    #     table_str += '|'
    #     for i, item in enumerate(line):
    #         table_str += ' ' + str(item).ljust(column_widths[i] - 2) + '|'
    #     table_str += '\n'
    #
    # table_str += '=' * (sum(column_widths) + 1) + '\n'
    #
    # return table_str

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([" "] + input_sentence, rotation=90)
    ax.set_yticklabels([" "] + output_words.split(" ")+["<eos>"])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def plot_figure(save_dir: str, historys: dict):
    # plot train LOSS history
    plt.plot(historys['TRAIN_STEP_LOSSES'], label='TRAIN')
    plt.xlabel('STEP')
    plt.ylabel('LOSS')
    plt.title('Model Train Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()
    logger.info(f'Training loss diagrams was saved in {os.path.join(save_dir, "train_loss.png")} !!!')

    # plot train ACC history
    plt.plot(historys['TRAIN_STEP_ACCS'], label='TRAIN')
    plt.xlabel('STEP')
    plt.ylabel('ACC')
    plt.title('Model Train ACC')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_acc.png'))
    plt.close()
    logger.info(f'Training acc diagrams was saved in {os.path.join(save_dir, "train_acc.png")} !!!')

    # plot train PPL history
    plt.plot(historys['TRAIN_STEP_PPLS'], label='TRAIN')
    plt.xlabel('EPOCH')
    plt.ylabel('PPL')
    plt.title('Model Train Perplexity')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_perplexity.png'))
    plt.close()
    logger.info(f'Training ppl diagrams was saved in {os.path.join(save_dir, "train_perplexity.png")} !!!')
    # ========================================================================================================= #

    # plot eval LOSS history
    plt.plot(historys['AVG_EVAL_LOSSES'], label='EVAL')
    plt.xlabel('STEP')
    plt.ylabel('LOSS')
    plt.title('Model Eval Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'eval_loss.png'))
    plt.close()
    logger.info(f'EVAL loss diagrams was saved in {os.path.join(save_dir, "eval_loss.png")} !!!')

    # plot eval ACC history
    plt.plot(historys['EVAL_ACCS'], label='EVAL')
    plt.xlabel('STEP')
    plt.ylabel('ACC')
    plt.title('Model Eval ACC')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'eval_acc.png'))
    plt.close()
    logger.info(f'EVAL acc diagrams was saved in {os.path.join(save_dir, "eval_acc.png")} !!!')

    # plot eval PPL history
    plt.plot(historys['EVAL_PPLS'], label='EVAL')
    plt.xlabel('STEP')
    plt.ylabel('PPL')
    plt.title('Model Eval Perplexity')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'eval_perplexity.png'))
    plt.close()
    logger.info(f'EVAL ppl diagrams was saved in {os.path.join(save_dir, "eval_perplexity.png")} !!!')


def save_train_history(save_dir: str, historys: dict):
    with open(os.path.join(save_dir, 'training_history.json'), "w", encoding="utf-8") as out:
        json.dump(historys, out)
    logger.info(f'Training history was saved in {os.path.join(save_dir, "training_history.png")} !!!')


def init_lstm_(lstm, init_weight=0.1):
    """
    Initializes weights of LSTM layer.
    Weights and biases are initialized with uniform(-init_weight, init_weight)
    distribution.
    :param lstm: instance of torch.nn.LSTM
    :param init_weight: range for the uniform initializer
    """
    # Initialize hidden-hidden weights
    init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    # Initialize input-hidden weights:
    init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)

    # Initialize bias. PyTorch LSTM has two biases, one for input-hidden GEMM
    # and the other for hidden-hidden GEMM. Here input-hidden bias is
    # initialized with uniform distribution and hidden-hidden bias is
    # initialized with zeros.
    init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
    init.zeros_(lstm.bias_hh_l0.data)

    if lstm.bidirectional:
        init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight)
        init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight)

        init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
        init.zeros_(lstm.bias_hh_l0_reverse.data)

if __name__ == "__main__":
    src_text = 'I go to school .'
    src_tok = ['I', 'go', 'to', 'school', '.']
    tgt_text = 'Tôi đi học .'
    feat = ['nsubj', 'root', 'case', 'obl', 'punct']
    make_table(src_text, src_tok, feat, tgt_text)