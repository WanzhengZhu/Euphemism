import os
import json
import re
import argparse


tok_map = {'null': 'None', 'true': 'True', 'false': 'False'}

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


def tokenize_post(string):
  import preprocessor as p
  return p.tokenize(string)


def write_gab_corpus(folder, save_fn):
    fout = open(save_fn, "w")
    for rel_fn in os.listdir(folder):
        fn = folder + "/" + rel_fn
        with open(fn, "r") as fin:
            for line in fin:
                for tok in tok_map:
                    line = line.replace(tok, tok_map[tok])
                post_dict = eval(line.strip())
                post = post_dict["body"]
                tok_post = tokenize_str(post)
                fout.write(tok_post + "\n")
    fout.close()
    print("save tok gap posts to {}".format(save_fn))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gab_folder", required=True, default=None, type=str)
    parser.add_argument("--save_path", required=True, default=None, type=str)
    args = parser.parse_args()
    
    write_gab_corpus(args.gab_folder, args.save_path)
                

