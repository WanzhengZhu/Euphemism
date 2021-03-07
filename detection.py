from collections import defaultdict

import nltk
import random
import string
import torch
from nltk.corpus import stopwords
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Initialize BERT vocabulary...')
bert_tokenizer = BertTokenizer(vocab_file='data/BERT_model_reddit/vocab.txt')
print('Initialize BERT model...')
bert_model = BertForMaskedLM.from_pretrained('data/BERT_model_reddit').to(device)
bert_model.eval()


''' Printing functions '''
class print_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def color_print_top_words(top_words, gt_euphemism):
    print('[Euphemism Candidates]: ')
    gt_euphemism_upper = set([y for x in gt_euphemism for y in x.split()])
    for i in top_words[:100]:
        if i in gt_euphemism:
            print(print_color.BOLD + print_color.PURPLE + i + print_color.END, end=', ')
        elif i in gt_euphemism_upper:
            print(print_color.UNDERLINE + print_color.PURPLE + i + print_color.END, end=', ')
        else:
            print(i, end=', ')
    print()


''' Evaluation '''
def evaluate_detection(top_words, gt_euphemism):
    color_print_top_words(top_words, gt_euphemism)
    correct_list = []  # appear in the ground truth
    correct_list_upper = []  # not appear in the ground truth but contain in a ground truth phase.
    gt_euphemism_upper = set([y for x in gt_euphemism for y in x.split()])
    for i, x in enumerate(top_words):
        correct_list.append(1 if x in gt_euphemism else 0)
        correct_list_upper.append(1 if x in gt_euphemism_upper else 0)

    topk_precision_list = []
    cummulative_sum = 0
    topk_precision_list_upper = []
    cummulative_sum_upper = 0
    for i in range(0, len(correct_list)):
        cummulative_sum += correct_list[i]
        topk_precision_list.append(cummulative_sum/(i+1))
        cummulative_sum_upper += correct_list_upper[i]
        topk_precision_list_upper.append(cummulative_sum_upper/(i+1))

    for topk in [10, 20, 30, 40, 50, 60, 80, 100]:
        if topk < len(topk_precision_list):
            print('Top-{:d} precision is ({:.2f}, {:.2f})'.format(topk, topk_precision_list[topk-1], topk_precision_list_upper[topk-1]))
    return 0


''' Main Function '''
def MLM(sgs, input_keywords, thres=1, filter_uninformative=1):
    def to_bert_input(tokens, bert_tokenizer):
        token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
        sep_idx = tokens.index('[SEP]')
        segment_idx = token_idx * 0
        segment_idx[(sep_idx + 1):] = 1
        mask = (token_idx != 0)
        return token_idx.unsqueeze(0).to(device), segment_idx.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)

    def single_MLM(message):
        MLM_k = 50
        tokens = bert_tokenizer.tokenize(message)
        if len(tokens) == 0:
            return []
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
        with torch.no_grad():
            logits = bert_model(token_idx, segment_idx, mask, masked_lm_labels=None)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        for idx, token in enumerate(tokens):
            if token == MASK:
                topk_prob, topk_indices = torch.topk(probs[idx, :], MLM_k)
                topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())

        out = [[topk_tokens[i], float(topk_prob[i])] for i in range(MLM_k)]
        return out

    PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'
    MLM_score = defaultdict(float)
    temp = sgs if len(sgs) < 10 else tqdm(sgs)
    skip_ms_num = 0
    good_sgs = []
    for sgs_i in temp:
        top_words = single_MLM(sgs_i)
        seen_input = 0
        for input_i in input_keywords:
            if input_i in [x[0] for x in top_words[:thres]]:
                seen_input += 1
        if filter_uninformative == 1 and seen_input < 2:
            skip_ms_num += 1
            continue
        good_sgs.append(sgs_i)
        for j in top_words:
            if j[0] in string.punctuation:
                continue
            if j[0] in stopwords.words('english'):
                continue
            if j[0] in input_keywords:
                continue
            if j[0] in ['drug', 'drugs']:  # exclude these two for the drug dataset.
                continue
            if j[0][:2] == '##':  # the '##' by BERT indicates that is not a word.
                continue
            MLM_score[j[0]] += j[1]
        # print(sgs_i)
        # print([x[0] for x in top_words[:20]])
    out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
    out_tuple = [[x, MLM_score[x]] for x in out]
    if len(sgs) >= 10:
        print('The percentage of uninformative masked sentences is {:d}/{:d} = {:.2f}%'.format(skip_ms_num, len(sgs), float(skip_ms_num)/len(sgs)*100))
    return out, out_tuple, good_sgs


def euphemism_detection(input_keywords, all_text, ms_limit, filter_uninformative):
    print('\n' + '*' * 40 + ' [Euphemism Detection] ' + '*' * 40)
    print('[util.py] Input Keyword: ', end='')
    print(input_keywords)
    print('[util.py] Extracting masked sentences for input keywords...')
    masked_sentence = []
    for sentence in tqdm(all_text):
        temp = nltk.word_tokenize(sentence)
        for input_keyword_i in input_keywords:
            if input_keyword_i not in temp:
                continue
            temp_index = temp.index(input_keyword_i)
            masked_sentence += [' '.join(temp[: temp_index]) + ' [MASK] ' + ' '.join(temp[temp_index + 1:])]
    random.shuffle(masked_sentence)
    masked_sentence = masked_sentence[:ms_limit]
    print('[util.py] Generating top candidates...')
    top_words, _, _ = MLM(masked_sentence, input_keywords, thres=5, filter_uninformative=filter_uninformative)
    return top_words

