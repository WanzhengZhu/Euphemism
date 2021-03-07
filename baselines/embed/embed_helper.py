"""
Word Embedding
    - word2vec based embedding
    - character based embedding
"""

import os
import re
import numpy as np
import time
from gensim.models.word2vec import LineSentence
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


def train_char_embed(corpus_fn, model_fn, vec_dim=100, window=8):
    # train embed
    sentences = LineSentence(corpus_fn)
    sent_cnt = 0
    for sentence in sentences:
        sent_cnt += 1
    print("# of sents: {}".format(sent_cnt))
    start = time.time()
    model = FastText(sentences, min_count=10, size=vec_dim,
                     window=window, iter=8, workers=30)
    end = time.time()
    print("embed train time: {}s".format(end-start))

    # save embed model
    model.save(model_fn)
    print("Save FastText model to {}".format(model_fn))


def train_word2vec_embed(corpus_fn, embed_fn, ft=100, vec_dim=100, window=8):
    # train embed
    sentences = LineSentence(corpus_fn)
    sent_cnt = 0
    for sentence in sentences:
        sent_cnt += 1
    print("# of sents: {}".format(sent_cnt))
    start = time.time()
    model = Word2Vec(sentences, min_count=ft, size=vec_dim,
                     window=window, iter=10, workers=30)
    end = time.time()
    print("embed train time: {}s".format(end-start))

    # save embed
    model.wv.save_word2vec_format(embed_fn, binary=False)
    print("save embedding to {}".format(embed_fn))

