import os
import re
from collections import Counter


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


def read_vocab_from_wiki(fn):
    cnt = Counter()
    with open(fn, "r") as fin:
        for line in fin:
            if line.strip() == "":
                continue
            tok_line = tokenize_str(line)
            seq = tok_line.split()
            for word in seq:
                cnt[word] += 1
    return cnt


def read_vocab_from_corpus(fn):
    cnt = Counter()
    with open(fn, "r") as fin:
        for line in fin:
            if line.strip() == "":
                continue
            seq = line.lower().split()
            for word in seq:
                cnt[word] += 1
    return cnt

