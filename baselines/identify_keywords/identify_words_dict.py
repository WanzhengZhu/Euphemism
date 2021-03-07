import os
import argparse
from nltk.corpus import words
from corpus_helper import read_vocab_from_corpus

def identify_words(corpus, ft, save_path):
    # get vocabulary of the corpus
    vocab_cnt = read_vocab_from_corpus(corpus)
    # load standard dictionary
    dictionary = set(words.words())
    # select words that are not in standard dictionary
    selected_words = []
    for word in vocab_cnt:
        if vocab_cnt[word] < ft:
            continue
        if word in dictionary:
            continue
        if not word.isalpha():
            continue
        selected_words.append(word)
    if save_path != None:
        fout = open(save_path, "w")
        for word in selected_words:
            fout.write(word+"\n")
        fout.close()
        print("save words to {}".format(save_path))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, default=None, type=str)
    parser.add_argument("--frequency_threshold", required=True, default=None, type=int)
    parser.add_argument("--save_path", required=True, default=None, type=str)
    args = parser.parse_args()

    identify_words(args.corpus, args.frequency_threshold, args.save_path)
