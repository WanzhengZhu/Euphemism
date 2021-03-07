"""
Visualize word embeddings
"""
import os
import argparse
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
import re
import pickle
from sklearn import manifold, datasets, decomposition, discriminant_analysis


word2eu = {"hillary": {"positive": ["hitlery", "killary", "hilary", "hellary", "shillary", "#killary", "#sickhillary"],\
                       "negative": ["hrc", "trump", "she", "obama", "chelsea", "bill", "kaine", "sanders", "djt"]},
           "twitter": {"positive": ["twatter", "twit", "bluebird", "twtr", "tweeter", "#twitter", "#twatter"], \
                       "negative": ["fb", "facebook", "4chan", "cuckbook", "fakebook", "tweet", "reddit"]},
           "obama": {"positive": ["0bama", "obummer", "bho", "hussein", "barack", "barry"], \
                       "negative": ["obamas", "hillary", "administration",  "congress", "nafta", "hrc", "obamacare"]},
           "trump": {"positive": ["djt", "drumpf", "dt"], \
                       "negative": ["hillary", "hitlery", "hrc", "bernie", "obama", "gop", "sanders", "killary", "pence"]},
           "commie": {"positive": ["communist", "socialist", "marxist"], \
                       "negative": ["lunatic", "traitor", "fascist", "jew", "scumbag", "zionist", "kike", "traitorous", "prick"]},}


def plot_neighbors(meta_fn, plot_fn):
    import matplotlib.pyplot as plt
    for word in word2eu:
        with open(meta_fn+"."+word, "rb") as handle:
            X_pca, target_word, pos_words, neg_words = pickle.load(handle)
        word_pca = X_pca[0]
        pos_pca = X_pca[1:1+len(pos_words)]
        neg_pca = X_pca[1+len(pos_words):]
        
        plt.figure()
        ax = plt.subplot()
        # plt target word
        ax.scatter(x=word_pca[0], y=word_pca[1], s=128, c="red")
        ax.annotate(target_word, word_pca)

        # plt positive euphemisms
        for (vector, eu) in zip(pos_pca, pos_words):
            ax.scatter(x=vector[0], y=vector[1], s=64, c="orange")
            ax.annotate(eu, vector)

        # plt negative euphemisms
        for (vector, eu) in zip(neg_pca, neg_words):
            ax.scatter(x=vector[0], y=vector[1], s=64, c="darkblue")
            ax.annotate(eu, vector)
        plt.savefig(plot_fn+"."+word+".png", bbox_inches="tight")
        plt.close()
    


def plot_neighbors_3D(meta_fn, plot_fn):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    for word in word2eu:
        with open(meta_fn+"."+word, "rb") as handle:
            X_pca, target_word, pos_words, neg_words = pickle.load(handle)
        word_pca = X_pca[0]
        pos_pca = X_pca[1:1+len(pos_words)]
        neg_pca = X_pca[1+len(pos_words):]
        
        plt.figure()
        ax = plt.axes(projection="3d")
        # plt target word
        ax.scatter(word_pca[0], word_pca[1], word_pca[2], s=128, c="red")
        #ax.annotate(target_word, word_pca)
        ax.text(word_pca[0], word_pca[1], word_pca[2], target_word, color="red")

        # plt positive euphemisms
        for (vector, eu) in zip(pos_pca, pos_words):
            ax.scatter(vector[0], vector[1], vector[2], s=64, c="orange")
            #ax.annotate(eu, vector)
            ax.text(vector[0], vector[1], vector[2], eu, color="orange")

        # plt negative euphemisms
        for (vector, eu) in zip(neg_pca, neg_words):
            ax.scatter(vector[0], vector[1], vector[2], s=64, c="darkblue")
            #ax.annotate(eu, vector)
            ax.text(vector[0], vector[1], vector[2], eu, color="darkblue")
        plt.savefig(plot_fn+"."+word+".png", bbox_inches="tight")
        plt.close()
    


def load_words_vectors(embed_fn, meta_fn):
    # load embedding model
    wv = KeyedVectors.load_word2vec_format(embed_fn, binary=False)

    # retrieve word vectors
    for word in word2eu:
        word_vec = wv[word]
        pos_vectors = []
        pos_words = []
        for eu in word2eu[word]["positive"]:
            try:
                vec = wv[eu]
                pos_vectors.append(np.array(vec))
                pos_words.append(eu)
            except:
                print("no positive word: {}".format(eu))
        neg_vectors = []
        neg_words = []
        for eu in word2eu[word]["negative"]:
            try:
                vec = wv[eu]
                neg_vectors.append(np.array(vec))
                neg_words.append(eu)
            except:
                print("no negative word: {}".format(eu))

        # plot
        X = [word_vec] + pos_vectors + neg_vectors
        X_pca = decomposition.PCA(n_components=3).fit_transform(X)

        with open(meta_fn+"."+word, "wb") as handle:
            pickle.dump((X_pca, word, pos_words, neg_words), handle)
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, default=None, type=str)
    parser.add_argument("--embed_path", required=True, default=None, type=str)
    parser.add_argument("--save_folder", required=True, default=None, type=str)
    args = parser.parse_args()

    # train word2vec embed
    train_word2vec_embed(args.corpus, args.embed_path)

    # get word vectors for neighbors
    meta_prefix = os.path.join(args.save_folder, "visualize_vec")
    load_words_vectors(args.embed_path, meta_prefix)

    # plot vectors for neighbors
    2d_plot_prefix = os.path.join(args.save_folder, "2d_plt")
    plot_neighbors(meta_prefix, 2d_plot_prefix)
    3d_plot_prefix = os.path.join(args.save_folder, "3d_plt")
    plot_neighbors_3D(meta_prefix, 3d_plot_prefix)
    
