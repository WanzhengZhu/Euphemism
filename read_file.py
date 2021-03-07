import time
from collections import defaultdict

from tqdm import tqdm

''' Read Data '''
def read_raw_text(fname, input_keywords):
    start = time.time()
    all_text = []
    num_lines = sum(1 for line in open(fname, 'r'))
    with open(fname, 'r') as fin:
        for line in tqdm(fin, total=num_lines):
            temp = line.split()
            if any(ele in temp for ele in input_keywords) and len(line) <= 150:
                all_text.append(line.strip())
    print('[read_data.py] Finish reading data using %.2fs' % (time.time() - start))
    return all_text


def read_input_and_ground_truth(target_category_name):
    fname_euphemism_answer = './data/euphemism_answer_' + target_category_name + '.txt'
    fname_target_keywords_name = './data/target_keywords_' + target_category_name + '.txt'
    euphemism_answer = defaultdict(list)
    with open(fname_euphemism_answer, 'r') as fin:
        for line in fin:
            ans = line.split(':')[0].strip().lower()
            for i in line.split(':')[1].split(';'):
                euphemism_answer[i.strip().lower()].append(ans)
    input_keywords = sorted(list(set([y for x in euphemism_answer.values() for y in x])))
    target_name = {}
    count = 0
    with open(fname_target_keywords_name, 'r') as fin:
        for line in fin:
            for i in line.strip().split('\t'):
                target_name[i.strip()] = count
            count += 1
    return euphemism_answer, input_keywords, target_name


def read_all_data(dataset_name, target_category_name):
    """ target_name is a dict (key: a target keyword, value: index). This is for later classification purpose, since
        different target keyword can refer to the same concept (e.g., 'alprazolam' and 'xanax', 'ecstasy' and 'mdma').
    """
    print('[read_data.py] Reading data...')
    euphemism_answer, input_keywords, target_name = read_input_and_ground_truth(target_category_name)
    all_text = read_raw_text('./data/text/' + dataset_name + '.txt', input_keywords)
    return all_text, euphemism_answer, input_keywords, target_name

