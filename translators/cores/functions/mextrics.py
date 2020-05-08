from collections import defaultdict
import math


class BLUE(object):
    def __init__(self, ngrams: int = 1, smooth: float = 0.):
        self.ngrams = ngrams
        self.smooth = smooth

    def _update_ngrams_count(self, sent, count):
        length = len(sent)
        for n in range(1, self.ngrams + 1):
            for i in range(length - n + 1):
                ngram = tuple(sent[i : (i + n)])
                count[ngram] += 1

    def _compute_bleu(self, p, len_pred, len_gold):
        # Brevity penalty.
        log_brevity = 1 - max(1, (len_gold + self.smooth) / (len_pred + self.smooth))
        log_score = 0
        ngrams = len(p) - 1
        for n in range(1, ngrams + 1):
            if p[n][1] > 0:
                if p[n][0] == 0:
                    p[n][0] = 1e-16
                log_precision = math.log((p[n][0] +  self.smooth) / (p[n][1] +  self.smooth))
                log_score += log_precision
        log_score /= ngrams
        return math.exp(log_score + log_brevity)

    # Calculate BLEU of prefixes of pred.
    def sentence_bleu(self, gold, pred, weight):
        scores = []
        # Get ngrams count for gold.
        count_gold = defaultdict(int)
        self._update_ngrams_count(gold, count_gold)
        # Init ngrams count for pred to 0.
        count_pred = defaultdict(int)
        # p[n][0] stores the number of overlapped n-grams.
        # p[n][1] is total # of n-grams in pred.
        p = []
        for n in range(self.ngrams + 1):
            p.append([0, 0])
        for i in range(len(pred)):
            for n in range(1, self.ngrams + 1):
                if i - n + 1 < 0:
                    continue
                # n-gram is from i - n + 1 to i.
                ngram = tuple(pred[(i - n + 1): (i + 1)])
                # Update n-gram count.
                count_pred[ngram] += 1
                # Update p[n].
                p[n][1] += 1
                if count_pred[ngram] <= count_gold[ngram]:
                    p[n][0] += 1
            scores.append(self._compute_bleu(p, i + 1, len(gold)))
        return scores

    # Calculate BLEU of a corpus.
    def corpus_bleu(self, golds, preds, weight):
        assert len(preds) == len(golds)
        p = []
        for n in range(self.ngrams + 1):
            p.append([0, 0])
        len_pred = len_gold = 0
        for pred, gold in zip(preds, golds):
            len_gold += len(gold)
            count_gold = defaultdict(int)
            self._update_ngrams_count(gold, count_gold)

            len_pred += len(pred)
            count_pred = defaultdict(int)
            self._update_ngrams_count(pred, count_pred)

            for k, v in count_pred.items():
                n = len(k)
                p[n][0] += min(v, count_gold[k])
                p[n][1] += v

        return self._compute_bleu(p, len_pred, len_gold)