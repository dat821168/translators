import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    with open("outputs/BLEU 27.51 deprel/training_history.json", 'r', encoding='utf-8') as f:
        historys = json.load(f)
        x_step = range(0, len(historys['TRAIN_STEP_LOSSES']), 1000)

        # plot train LOSS history
        xvalues = [historys['TRAIN_STEP_LOSSES'][v] for v in x_step]
        plt.figure(figsize=(12,6))
        plt.plot(xvalues, label='TRAIN')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('LOSS')
        plt.title('Model Train Loss')
        plt.legend()
        plt.savefig('outputs/train_loss.png')
        plt.show()

        # plot train ACC history
        xvalues = [historys['TRAIN_STEP_ACCS'][v] for v in x_step]
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='TRAIN')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('ACC')
        plt.title('Model ACC Loss')
        plt.legend()
        plt.savefig('outputs/train_acc.png')
        plt.show()

        # plot train LOSS history
        xvalues = [historys['TRAIN_STEP_PPLS'][v] for v in x_step]
        plt.figure(figsize=(12, 6))
        plt.plot(xvalues, label='TRAIN')
        plt.xlabel('STEP (x1000)')
        plt.ylabel('PPL')
        plt.title('Model PPL Loss')
        plt.ticklabel_format(useOffset=False)
        plt.legend()
        plt.savefig('outputs/train_ppl.png')
        plt.show()

