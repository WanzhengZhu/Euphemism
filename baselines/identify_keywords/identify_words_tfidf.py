import os
import argparse
from collections import defaultdict
from corpus_helper import read_vocab_from_wiki, read_vocab_from_corpus


def identify_words(target_corpus, wiki_corpus, ft, save_path):
    wiki_cnt = read_vocab_from_wiki(wiki_corpus)
    target_cnt = read_vocab_from_corpus(target_corpus)
    score_dict = defaultdict(float)
    for word in target_cnt:
        if target_cnt[word] < ft:
            continue
        if not word.isalpha():
            continue
        score = 1.0 * target_cnt[word] / (target_cnt[word] + wiki_cnt[word])
        score_dict[word] = score
    sorted_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

    if save_path is not None:
        fout = open(save_path, "w")
        for item in sorted_dict:
            word, score = item
            fout.write(word+"\t"+str(score)+"\n")
        fout.close()
        print("save words to {}".format(save_path))

        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_corpus", required=True, default=None, type=str)
    parser.add_argument("--wiki_corpus", required=True, default=None, type=str)
    parser.add_argument("--frequency_threshold", required=True, default=None, type=int)
    parser.add_argument("--save_path", required=True, default=None, type=str)
    args = parser.parse_args()

    identify_words(args.target_corpus, args.wiki_corpus,
                   args.frequency_threshold, args.save_path)
    


