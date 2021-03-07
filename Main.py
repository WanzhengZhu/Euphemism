import argparse

from detection import euphemism_detection, evaluate_detection
from identification import euphemism_identification
from read_file import read_all_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sample")  # dataset file name
parser.add_argument('--target', type=str, default="drug")  # [drug, weapon, sex]
parser.add_argument('--c1', type=int, default=2)  # Coarse Binary Classifier: select index from ['LRT', 'LSTM', 'LSTMAtten', 'RNN', 'RCNN', 'SelfAttention']
parser.add_argument('--c2', type=int, default=0)  # Fine-grained Multi-class Classifer: select index from ['LRT', 'LSTM', 'LSTMAtten', 'RNN', 'RCNN', 'SelfAttention']
parser.add_argument('--coarse', type=int, default=1)  # Whether to include the coarse binary classifier or not.
args = parser.parse_args()
print(args)


''' Read Data '''
all_text, euphemism_answer, input_keywords, target_name = read_all_data(args.dataset, args.target)

''' Euphemism Detection '''
top_words = euphemism_detection(input_keywords, all_text, ms_limit=2000, filter_uninformative=1)
evaluate_detection(top_words, euphemism_answer)

''' Euphemism Identification '''
euphemism_identification(top_words, all_text, euphemism_answer, input_keywords, target_name, args)

