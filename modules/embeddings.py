from gensim.models import Word2Vec
import nltk

def train_word2vec(text):
    sentences = [nltk.word_tokenize(text)]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
    return model