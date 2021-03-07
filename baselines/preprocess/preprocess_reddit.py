import csv
import re
from collections import Counter
import argparse

def tokenize_str(string):
    string = re.sub(r"[^A-Za-z0-9()#@$,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def proc_reddit_corpus(fn="data/reddit.csv", save_fn="reddit_corpus.txt"):
    fout = open(save_fn, "w")
    with open(fn) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            subreddit, title, text = row
            fout.write(tokenize_str(text)+"\n")
    fout.close()
    print("Save reddit corpus to {}".format(save_fn))

def read_corpus_vocab(fn):
    cnt = Counter()
    with open(fn, "r") as fin:
        for line in fin:
            if line.strip() == "":
                continue
            tok_line = tokenize_str(line)
            seq = tok_line.split()
            for word in seq:
                cnt[word] += 1
    print("fn: {}, vocab size: {}".format(fn, len(cnt)))
    return cnt    
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reddit_path", required=True, default=None, type=str)
    parser.add_argument("--save_path", required=True, default=None, type=str)
    args = parser.parse_args()

    proc_reddit_corpus(fn=args.reddit_path, save_fn=args.save_path)


