"""
 identify euphemisms with
 - general wiki word embeddings
 - graph for eigenvector centrality
"""
import argparse
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import operator
import copy
import pickle
from sklearn.metrics.pairwise import cosine_distances as cosine_dist_func
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import sys, os
sys.path.insert(0, os.path.abspath("."))
from embed.embed_helper import train_char_embed, train_word2vec_embed


def load_emb_from_wiki(embed_fn):
    emb_dict = KeyedVectors.load_word2vec_format(embed_fn, binary=False, limit=20000)
    return emb_dict


def build_vocab_emb_from_corpus(corpus_fn, emb_dict, ft=5):
    vocab_cnt = Counter()
    with open(corpus_fn, "r") as fin:
        for line in fin:
            seq = line.strip().split()
            for tok in seq:
                vocab_cnt[tok] += 1
    sorted_vocab = vocab_cnt.most_common()
    vocab = []
    word2idx = {}
    emb_arr = []
    for (word, cnt) in sorted_vocab:
        if cnt <= ft:
            continue
        try:
            vec = emb_dict[word]
            vocab.append(word)
            word2idx[word] = len(vocab) - 1
            emb_arr.append(list(vec))
        except:
            continue
    print("vocab size: {}, emb_size: {}".format(len(vocab), len(emb_arr)))
    return vocab, word2idx, emb_arr


def create_graph_with_pairwise_cosine(emb_arr, vocab):
    graph = nx.Graph()
    for word in vocab:
        graph.add_node(word)
    vocab_size = len(vocab)
    for i in range(vocab_size-1):
        # row: (1, dim)
        row = [emb_arr[i]]
        # cosine_dist_row: (1, vocab_size)
        cosine_dist_row = cosine_dist_func(row, emb_arr[i+1:])
        for j in range(i+1, vocab_size):
            graph.add_weighted_edges_from([(vocab[i], vocab[j], cosine_dist_row[0][j-i-1])])
    return graph


def count_word_pairs(word2idx, corpus_fn, context_window=5):
    word_pair_cnt = defaultdict(int)
    with open(corpus_fn, "r") as fin:
        for line in fin:
            all_tok_seq = line.strip().lower().split()
            tok_seq = []
            idx_seq = []
            for tok in all_tok_seq:
                if tok in word2idx:
                    tok_seq.append(tok)
                    idx_seq.append(word2idx[tok])
            for pos in range(len(idx_seq)):
                idx = idx_seq[pos]
                tok = tok_seq[pos]
                for next_pos in range(pos+1, min(pos+context_window+1, len(idx_seq))):
                    next_idx = idx_seq[next_pos]
                    next_tok = tok_seq[next_pos]
                    if idx == next_idx:
                        continue
                    word_pair_cnt[(min(tok, next_tok), max(tok, next_tok))] += 1
    return word_pair_cnt


def remove_edges(graph, vocab, word_pair_cnt):
    """
    corpus with tokenized tokens
    """
    vocab_size = len(vocab)
    removed_edge_cnt = 0
    for idx in range(vocab_size):
        tok = vocab[idx]
        for next_idx in range(idx+1, vocab_size):
            next_tok = vocab[next_idx]
            word_pair = (min(tok, next_tok), max(tok, next_tok))
            if word_pair_cnt[word_pair] == 0:
                try:
                    graph.remove_edge(tok, next_tok)
                    removed_edge_cnt += 1
                except:
                    continue
    print("# of nodes: {}, # of removed edges: {}".format(vocab_size, removed_edge_cnt))
    return graph

    
def rank_words(graph, vocab, res_fn):
    centrality = nx.eigenvector_centrality(graph)
    sorted_vocab = sorted(centrality.items(), key=lambda kv: kv[1],
                          reverse=True)
    with open(res_fn, "w") as fout:
        for (word, score) in sorted_vocab:
            fout.write(word+"\t"+str(score)+"\n")
    print("saving ranked words to {}".format(res_fn))
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_corpus", required=True, default=None, type=str)
    parser.add_argument("--wiki_corpus", required=True, default=None, type=str)
    parser.add_argument("--wiki_embed", required=True, default=None, type=str)
    parser.add_argument("--save_path", required=True, default=None, type=str)
    args = parser.parse_args()

    # train wiki embedding
    train_word2vec_embed(args.wiki_corpus, args.wiki_embed)

    # load wiki embedding
    emb_dict = load_emb_from_wiki(args.wiki_embed)
    print("Finished loading wiki embedding...")

    # load vocab and emb
    vocab, word2idx, emb_arr = build_vocab_emb_from_corpus(
        args.target_corpus, emb_dict, ft=25)
    print("Finished vocabulary...")

    # create graph using pairwise distance
    graph = create_graph_with_pairwise_cosine(emb_arr, vocab)
    print("Finished constructing graph...")

    # remove edges from graph using co-occurrences
    word_pair_cnt = count_word_pairs(word2idx, args.target_corpus, context_window=10)
    graph = remove_edges(graph, vocab, word_pair_cnt)
    print("Finished cleaning graph...")
    with open("baseline_graph.pkl", "wb") as handle:
        pickle.dump(graph, handle)

    # rank words with eigenvector centrality
    rank_words(graph, vocab, args.save_path)
    print("Finished ranking words...")

