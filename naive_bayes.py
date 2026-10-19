"""
Naive Bayes Spam Classifier — built from scratch (no sklearn model).

Theory (Module 1 - Basics of Probability):
    We want P(spam | message) and P(ham | message).
    By Bayes' theorem:
        P(class | words) = P(words | class) * P(class) / P(words)

    Since P(words) is the same for both classes, we only need to compare:
        P(class) * P(words | class)

    "Naive" assumption: words are conditionally independent given the class, so
        P(words | class) = P(word_1 | class) * P(word_2 | class) * ... * P(word_n | class)

    We estimate P(word | class) using word frequency counts (Multinomial Naive Bayes),
    with Laplace (add-1) smoothing so unseen words don't zero out the whole product.

    We work in log-space to avoid numerical underflow from multiplying many small
    probabilities together:
        log P(class | words) ∝ log P(class) + sum( log P(word_i | class) )
"""

import re
import math
import pickle
from collections import Counter, defaultdict
from text_model import preprocess


def tokenize(text):
    """Lowercase and split into word tokens, stripping punctuation."""
    # same preprocessing as every other model (links/phones/money -> tokens)
    return re.findall(r"[a-z']+", preprocess(text))


class NaiveBayesSpamClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing constant
        self.class_priors = {}          # P(class)
        self.word_counts = {}           # class -> Counter(word -> count)
        self.class_total_words = {}     # class -> total word count in that class
        self.class_message_counts = {}  # class -> number of training messages seen (for priors)
        self.vocab = set()              # all words seen during training
        self.classes = []

    def fit(self, messages, labels):
        """
        messages: list of raw text strings
        labels: list of class labels, e.g. 'spam' / 'ham'
        """
        self.classes = sorted(set(labels))
        self.class_message_counts = dict(Counter(labels))
        n_total = len(labels)

        # P(class) = (# messages of that class) / (total messages)
        self.class_priors = {
            c: self.class_message_counts[c] / n_total for c in self.classes
        }

        self.word_counts = {c: Counter() for c in self.classes}
        self.class_total_words = {c: 0 for c in self.classes}

        for message, label in zip(messages, labels):
            tokens = tokenize(message)
            self.word_counts[label].update(tokens)
            self.class_total_words[label] += len(tokens)
            self.vocab.update(tokens)

        return self

    def partial_fit(self, message, true_label):
        """
        ADAPTIVE LEARNING (feature 4): incrementally update the model with a
        single new labelled example, without retraining from scratch.

        Because Naive Bayes is just word-frequency counting, "learning" from
        one new example is exact and immediate: we add its words to the
        relevant class's counts and recompute the priors. This is a genuine
        online-learning update, not an approximation.
        """
        if true_label not in self.classes:
            self.classes.append(true_label)
            self.classes.sort()
            self.word_counts[true_label] = Counter()
            self.class_total_words[true_label] = 0
            self.class_message_counts[true_label] = 0

        tokens = tokenize(message)
        self.word_counts[true_label].update(tokens)
        self.class_total_words[true_label] += len(tokens)
        self.vocab.update(tokens)
        self.class_message_counts[true_label] = self.class_message_counts.get(true_label, 0) + 1

        total_messages = sum(self.class_message_counts.values())
        self.class_priors = {
            c: self.class_message_counts.get(c, 0) / total_messages for c in self.classes
        }
        return self

    def _log_likelihood(self, tokens, c):
        """log P(words | class c) using Laplace-smoothed word probabilities."""
        vocab_size = len(self.vocab)
        log_prob = 0.0
        denom = self.class_total_words[c] + self.alpha * vocab_size
        for token in tokens:
            count = self.word_counts[c].get(token, 0)
            word_prob = (count + self.alpha) / denom
            log_prob += math.log(word_prob)
        return log_prob

    def predict_proba(self, message):
        """
        Returns a dict {class: probability} using Bayes' theorem.
        We compute unnormalized log-posteriors, exponentiate, then normalize
        so the probabilities sum to 1 (this avoids ever needing P(words) directly).
        """
        tokens = tokenize(message)
        log_posteriors = {}
        for c in self.classes:
            log_prior = math.log(self.class_priors[c])
            log_posteriors[c] = log_prior + self._log_likelihood(tokens, c)

        # normalize using the log-sum-exp trick for numerical stability
        max_log = max(log_posteriors.values())
        exp_shifted = {c: math.exp(lp - max_log) for c, lp in log_posteriors.items()}
        total = sum(exp_shifted.values())
        return {c: v / total for c, v in exp_shifted.items()}

    def predict(self, message):
        probs = self.predict_proba(message)
        return max(probs, key=probs.get)

    def explain(self, message, top_n=5):
        """
        EXPLAINABILITY (feature 2): returns the words in the message that most
        pushed the decision toward spam or ham, using the log-likelihood ratio
        of each word:

            contribution(word) = log P(word | spam) - log P(word | ham)

        A positive contribution means that word made the message look more
        spam-like; negative means more ham-like. This is a direct, honest
        readout of the Bayes' theorem math already being computed — nothing
        is hidden or approximated.
        """
        if "spam" not in self.classes or "ham" not in self.classes:
            return []

        tokens = tokenize(message)
        vocab_size = len(self.vocab)
        spam_denom = self.class_total_words.get("spam", 0) + self.alpha * vocab_size
        ham_denom = self.class_total_words.get("ham", 0) + self.alpha * vocab_size

        seen = set()
        contributions = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            spam_count = self.word_counts.get("spam", Counter()).get(token, 0)
            ham_count = self.word_counts.get("ham", Counter()).get(token, 0)
            p_spam = (spam_count + self.alpha) / spam_denom
            p_ham = (ham_count + self.alpha) / ham_denom
            contribution = math.log(p_spam) - math.log(p_ham)
            contributions.append((token, contribution))

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:top_n]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    print("Train everything with:  python train_all.py")