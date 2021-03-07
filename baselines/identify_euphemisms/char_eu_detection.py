"""
char_based_euphemism_detection.py
 - Use char based embedding, e.g., FastText
 - Embedding to detect euphesmisms of given words
"""
import argparse
import os
import re
import numpy as np
import time
from nltk.corpus import words
from gensim.models import KeyedVectors
from gensim.models import FastText
from gensim.models.word2vec import LineSentence

import sys, os
sys.path.insert(0, os.path.abspath("."))
from embed.embed_helper import train_char_embed
from data_helper import read_realword_seeds, read_misspelled_seeds

def eval_char_embed(model_fn, seeds, res_fn):
    fout = open(res_fn, "w")
    fout.write(",".join(["KeyWord", "Neighbors"])+"\n")
    model = FastText.load(model_fn)
    wv = model.wv
    for seed in seeds:
        try:
            neb_scores = wv.similar_by_word(seed, topn=50)
            nebs = [ns[0] for ns in neb_scores]
            # save all neighbors
            fout.write(",".join([seed] + nebs)+"\n")
            selected_nebs = []
            for neb_score in neb_scores:
                neb, score = neb_score
                double_neb_scores = wv.similar_by_word(neb, topn=50)
                neb_nebs = [tup[0] for tup in double_neb_scores]
                if seed in neb_nebs:
                    selected_nebs.append(neb)
            # save selected neighbors
            #fout.write(",".join([seed] + selected_nebs)+"\n")
        except:
            print("{} not in the dictionary".format(seed))
    fout.close()


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, default=None, type=str)
    parser.add_argument("--embed_path", required=True, default=None, type=str)
    parser.add_argument("--keyword_path", default=None, type=str)
    parser.add_argument("--save_path", required=True, default=None, type=str)
    args = parser.parse_args()

    # train char embed
    train_char_embed(args.corpus, args.embed_path)

    dictionary = set(words.words())
    
    # keywords that are legal English words
    real_seeds = ["acetaminophen and oxycodone combination", "adderall", "alprazolam", "amphetamine", "amphetamine and dextroamphetamine combination", "buprenorphine and naloxone combination", "clonazepam", "cocaine", "concerta", "crack cocaine", "daytrana", "dilaudid", "ecstasy", "fentanyl", "flunitrazepam", "gamma-hydroxybutyric acid", "ghb", "hash oil", "heroin", "hydrocodone", "hydromorphone", "ketalar", "ketamine", "khat", "klonopin", "lorcet", "lsd", "lysergic acid diethylamide", "marijuana", "marijuana concentrates", "mdma", "mescaline", "methamphetamine", "methylphenidate", "molly", "morphine", "norco", "opium", "oxaydo", "oxycodone", "oxycontin", "pcp", "percocet", "peyote", "phencyclidine", "promethazine", "psilocybin mushrooms", "ritalin", "rohypnol", "roxicodone", "steroids", "suboxone", "synthetic cannabinoids", "synthetic cathinones", "u-47700", "vicodin", "xanax"]
    # real_seeds = ["hillary", "trump", "sanders", "timkaine", "beardson", "kurtschlichter", "kurt", "schlichter", "obama", "bolsonaro", "voxday", "facebooks", "facebook", "twitter", "google", "cnns", "cnn", "foxnews", "breitbart"]
    if args.keyword_path is not None:
        real_seeds += read_realword_seeds(args.keyword_path, dictionary)
    print("# of realword seeds: {}".format(len(real_seeds)))
    # find euphemisms for real seeds
    eval_char_embed(args.embed_path, real_seeds, args.save_path)

    # # keywords that are misspellings
    # mis_seeds = ["hilldawg", "shillary", "hitlery", "hilldog", "dahnald", "trumpkins", "obummer"]
    # if args.keyword_path is not None:
    #     mis_seeds += read_misspelled_seeds(args.keyword_path, dictionary)
    # print("# of misspelled seeds: {}".format(len(mis_seeds)))
    # # find euphemisms for misspelled seeds
    # eval_char_embed(args.embed_path, mis_seeds, args.save_path+"_mis_words.txt")

