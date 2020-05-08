import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    with open("outputs/GNMT-deprel BLEU 27.51/training_history.json", 'r', encoding='utf-8') as f, \
            open("outputs/BLEU 27.66/training_history.json", 'r', encoding='utf-8') as f2:
        historys = json.load(f2)
        deprel_historys = json.load(f)
        x_step = range(0, len(historys['TRAIN_STEP_LOSSES']), 1000)
        x_deprel_step = range(0, len(deprel_historys['TRAIN_STEP_LOSSES']), 1000)

        # plot train LOSS history
        xvalues = [historys['TRAIN_STEP_LOSSES'][v] for v in x_step]
        x_deprel_values = [deprel_historys['TRAIN_STEP_LOSSES'][v] for v in x_step]
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='GNMT')
        plt.plot(x_deprel_values, label='GNMT-dep')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('LOSS')
        plt.title('Model Train Loss')
        plt.legend()
        plt.savefig('outputs/train_loss.png')
        plt.show()

        # plot train ACC history
        xvalues = [historys['TRAIN_STEP_ACCS'][v] for v in x_step]
        x_deprel_values = [deprel_historys['TRAIN_STEP_ACCS'][v] for v in x_step]
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='GNMT')
        plt.plot(x_deprel_values, label='GNMT-deprel')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('ACC')
        plt.title('Model Train Accuracy')
        plt.legend()
        plt.savefig('outputs/train_acc.png')
        plt.show()

        # plot train LOSS history
        xvalues = [historys['TRAIN_STEP_PPLS'][v] for v in x_step]
        x_deprel_values = [deprel_historys['TRAIN_STEP_PPLS'][v] for v in x_step]
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='GNMT')
        plt.plot(x_deprel_values, label='GNMT-deprel')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('PPL')
        plt.title('Model Train Perplexity')
        plt.ticklabel_format(useOffset=False)
        plt.legend()
        plt.savefig('outputs/train_ppl.png')
        plt.show()

        # plot train LOSS history
        xvalues = historys['AVG_EVAL_LOSSES']
        x_deprel_values = deprel_historys['AVG_EVAL_LOSSES']
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='GNMT')
        plt.plot(x_deprel_values, label='GNMT-dep')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('LOSS')
        plt.title('Model Eval Loss')
        plt.legend()
        plt.savefig('outputs/eval_loss.png')
        plt.show()

        # plot train ACC history
        xvalues = historys['EVAL_ACCS']
        x_deprel_values = deprel_historys['EVAL_ACCS']
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='GNMT')
        plt.plot(x_deprel_values, label='GNMT-deprel')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('ACC')
        plt.title('Model Eval Accuracy')
        plt.legend()
        plt.savefig('outputs/eval_acc.png')
        plt.show()

        # plot train LOSS history
        xvalues = historys['EVAL_PPLS']
        x_deprel_values = deprel_historys['EVAL_PPLS']
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='GNMT')
        plt.plot(x_deprel_values, label='GNMT-deprel')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('PPL')
        plt.title('Model Eval Perplexity')
        plt.ticklabel_format(useOffset=False)
        plt.legend()
        plt.savefig('outputs/eval_ppl.png')
        plt.show()
