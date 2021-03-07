import pickle
with open('./dea_drug_list.pkl', 'rb') as fin:
    new_list = pickle.load(fin)
dea_list = list(set(new_list))

prediction = []
with open('./drug_predictions.tsv', 'r') as fin:
    for line in fin:
        temp = line.split('\t')
        if temp[0] == '':
            continue
        prediction.append(temp[1])
print()

with open('../../euphemisms_cantreader.txt', 'w') as fout:
    for i in prediction:
        fout.write(i)
        fout.write('\n')


''' Rohan's Version '''
# import pickle
# import pandas as pd
# from pandas import DataFrame
# import math
# import random
# import re
# import argparse
# parser = argparse.ArgumentParser(description='Pass the files.')
# parser.add_argument('--dea_list', default='./dea_drug_list.pkl', type=str)
# parser.add_argument('--drug_list', default='./drug_predictions.tsv', type=str)
# # parser.add_argument('--drug_threshold', required=False, default=0.6, type=float)
# args = parser.parse_args()
# #input the drug list and the dea list, setting drug threshold is optional may lead to different precision and recall metrics
# dea_list = args.dea_list
# drug_list = args.drug_list
# drug_threshold=args.drug_threshold
# with open(dea_list,'rb') as fin:
#     new_list = pickle.load(fin)
#
# dea_list = list(set(new_list))
# for word in dea_list:
#     if len(word.split())>1:
#         dea_list.remove(word)
#
#     elif len(re.findall(r"[\w']+", word))>1:
#         dea_list.remove(word)
#
# df = pd.read_csv(drug_list,sep='\t')
# drug_true_jargon=[]
# drug_filter_jargon=[]
# #Find the drug words where bcn > 20 and the probability of being a drug word is greater than threshold
# for index,row in df.iterrows():
#     if row['prob'] > drug_threshold and row['bcn']>20:
#         drug_filter_jargon.append(row['hypo'])
#
# full_drug_list = list(df.hypo)
#
# TP=0
# FP=0
#
# #Checking for True Positives and False Positives
# for word in drug_filter_jargon:
#     if word in dea_list:
#         TP +=1
#     else:
#         FP +=1
#
# FN=0
# TN=0
# for word in dea_list:
#     if word not in drug_filter_jargon:
#         FN +=1
#
# print(TP)
# print(FP)
# print(FN)
# #Computing Precision and Recall
# Precision = TP /(TP +FP)
# Recall = TP/ (TP+FN)
#
# print(f"Precision is {Precision} \nRecall is {Recall}")
