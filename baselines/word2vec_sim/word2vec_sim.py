import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import time
from collections import defaultdict
from gensim.models.word2vec import LineSentence
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from identification import get_final_test, print_final_out


def load_emb_from_wiki(embed_fn):
    emb_dict = KeyedVectors.load_word2vec_format(embed_fn, binary=False, limit=20000)
    return emb_dict


def train_word2vec_embed(corpus_fn, embed_fn, ft=100, vec_dim=50, window=8):
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
    return model


def read_basic_files(dataset):
    euphemism_answer = defaultdict(list)
    with open('../../data/answer_' + dataset + '.txt', 'r') as fin:
        for line in fin:
            ans = line.split(':')[0].strip().lower()
            for i in line.split(':')[1].split(';'):
                euphemism_answer[i.strip().lower()].append(ans)
    drug_euphemism = sorted(list(set([x[0] for x in euphemism_answer.items()])))
    drug_formal = sorted(list(set([y for x in euphemism_answer.items() for y in x[1]])))
    drug_name = {}
    count = 0
    with open('../../data/name_' + dataset + '.txt', 'r') as fin:
        for line in fin:
            for i in line.strip().split('\t'):
                drug_name[i.strip()] = count
            count += 1
    return euphemism_answer, drug_formal, drug_name


def word2vec_detection_identification(prefix):
    # train wiki embedding
    corpus = '../../data/text/' + prefix + '.txt'
    embed_file = '../../' + prefix + '_embeddings.txt'
    # embed_file = '../../results/baselines/' + prefix + '_embeddings.txt'
    word2vec_model = train_word2vec_embed(corpus, embed_file)

    # load wiki embedding
    emb_dict = load_emb_from_wiki(embed_file)
    print("Finished loading embedding...")
    real_seeds = ["acetaminophen and oxycodone combination", "adderall", "alprazolam", "amphetamine", "amphetamine and dextroamphetamine combination", "buprenorphine and naloxone combination", "clonazepam", "cocaine", "concerta", "crack cocaine", "daytrana", "dilaudid", "ecstasy", "fentanyl", "flunitrazepam", "gamma-hydroxybutyric acid", "ghb", "hash oil", "heroin", "hydrocodone", "hydromorphone", "ketalar", "ketamine", "khat", "klonopin", "lorcet", "lsd", "lysergic acid diethylamide", "marijuana", "marijuana concentrates", "mdma", "mescaline", "methamphetamine", "methylphenidate", "molly", "morphine", "norco", "opium", "oxaydo", "oxycodone", "oxycontin", "pcp", "percocet", "peyote", "phencyclidine", "promethazine", "psilocybin mushrooms", "ritalin", "rohypnol", "roxicodone", "steroids", "suboxone", "synthetic cannabinoids", "synthetic cathinones", "u-47700", "vicodin", "xanax"]
    # real_seeds = ['pistol', 'gun', 'rifles']
    # real_seeds = ['genitals', 'penis', 'nipple']

    ''' Detection '''
    target_vector = []
    seq = []
    for i, seed in enumerate(real_seeds):
        if seed in emb_dict:
            target_vector.append(emb_dict[seed])
            seq.append(i)
    target_vector = np.array(target_vector)
    target_vector_ave = np.sum(target_vector, 0) / len(target_vector)
    top_words = [x[0] for x in word2vec_model.wv.similar_by_vector(target_vector_ave, topn=2000) if x[0] not in real_seeds]
    print(top_words)
    with open('../../euphemisms_word2vec_' + prefix + '.txt', 'w') as fout:
        for i in top_words:
            fout.write(i)
            fout.write('\n')

    ''' Identification '''
    euphemism_candidates = []
    with open('../../results/top_words_reddit.txt', 'r') as fin:
        for line in fin:
            euphemism_candidates.append(line.strip())
    euphemism_answer, drug_formal, drug_name = read_basic_files('drug')
    final_test = get_final_test(euphemism_answer, euphemism_candidates, drug_formal)
    answer = []
    filtered_final_test = {}
    for i in euphemism_candidates:
        if (i in emb_dict) and (final_test[i] != ['None']):
            answer.append([drug_name[real_seeds[seq[x]]] for x in np.argsort(cosine_similarity([emb_dict[i]], target_vector)).tolist()[0][::-1]])
            filtered_final_test[i] = final_test[i]
    final_answer = []
    for answer_i in answer:
        temp = []
        for j in answer_i:
            if j not in temp:
                temp.append(j)
        final_answer.append(temp)
    print(final_answer)
    print_final_out(final_answer, filtered_final_test, drug_name)



if __name__=="__main__":
    # word2vec_detection_identification('sample')
    print()

