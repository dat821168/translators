import os
import sys
import tempfile
import codecs
from collections import defaultdict, Counter
from tools.subword import learn_bpe, apply_bpe


def get_vocabulary(fobj, is_dict=False):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    for i, line in enumerate(fobj):
        if is_dict:
            try:
                word, count = line.strip('\r\n ').split(' ')
            except:
                print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                sys.exit(1)
            vocab[word] += int(count)
        else:
            for word in line.strip('\r\n ').split(' '):
                if word:
                    vocab[word] += 1
    return vocab


def main():
    output_name = "bpe_codes"
    vocab = ["vocab_vi.txt", "vocab_en.txt"]
    symbols = 32000
    min_frequency = 2
    verbose = False
    total_symbols = False
    separator = "@@"

    input_text = [codecs.open(f, encoding='UTF-8') for f in ["C:/Project/JEEDOC/translators/datasets/train.vi",
                                                                  "C:/Project/JEEDOC/translators/datasets/train.en"]]
    vocab = [codecs.open(f, 'w', encoding='UTF-8') for f in vocab]
    # get combined vocabulary of all input texts
    full_vocab = Counter()
    for f in input_text:
        full_vocab += learn_bpe.get_vocabulary(f)
        f.seek(0)

    vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]

    # learn BPE on combined vocabulary
    with codecs.open(output_name, 'w', encoding='UTF-8') as output:
        learn_bpe.learn_bpe(vocab_list, output, symbols, min_frequency, verbose, is_dict=True,
                            total_symbols=total_symbols)
    with codecs.open(output_name, encoding='UTF-8') as codes:
        bpe = apply_bpe.BPE(codes, separator=separator)

    # apply BPE to each training corpus and get vocabulary
    for train_file, vocab_file in zip(input_text, vocab):

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        tmpout = codecs.open(tmp.name, 'w', encoding='UTF-8')

        train_file.seek(0)
        for line in train_file:
            tmpout.write(bpe.segment(line).strip())
            tmpout.write('\n')

        tmpout.close()
        tmpin = codecs.open(tmp.name, encoding='UTF-8')

        vocab = learn_bpe.get_vocabulary(tmpin)
        tmpin.close()
        os.remove(tmp.name)

        for key, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write("{0} {1}\n".format(key, freq))
        vocab_file.close()

if __name__ == "__main__":
    main()