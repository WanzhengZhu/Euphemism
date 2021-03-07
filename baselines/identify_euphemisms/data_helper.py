import os
    
def read_realword_seeds(target_word_fn, dictionary, limit=200):
    selected_seeds = []
    cnt = 0
    with open(target_word_fn, "r") as fin:
        for line in fin:
            word = line.strip().split()[0]
            if word in dictionary:
                selected_seeds.append(word)
                cnt += 1
                if cnt > limit:
                    break
    return selected_seeds


def read_misspelled_seeds(target_word_fn, dictionary, limit=200):
    selected_seeds = []
    cnt = 0
    with open(target_word_fn, "r") as fin:
        for line in fin:
            word = line.strip().split()[0]
            if word not in dictionary:
                selected_seeds.append(word)
                cnt += 1
                if cnt > limit:
                    break
    return selected_seeds

