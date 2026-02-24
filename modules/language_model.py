from nltk.util import ngrams
from collections import Counter
import math

def build_bigram_model(text):
    tokens = text.split()
    bigrams = list(ngrams(tokens, 2))
    return Counter(bigrams)

def calculate_perplexity(model, text):
    tokens = text.split()
    N = len(tokens)
    log_prob = 0
    for i in range(len(tokens)-1):
        bigram = (tokens[i], tokens[i+1])
        prob = model[bigram] / sum(model.values())
        if prob > 0:
            log_prob += math.log(prob)
    return math.exp(-log_prob / N)