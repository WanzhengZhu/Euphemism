import numpy as np
import os
import io
import sqlite3
import pickle
from tqdm import tqdm
import scipy
import csv
import argparse
from hdf5_save import adapt_array,convert_array
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

parser = argparse.ArgumentParser(description='Pass the files.')
parser.add_argument('--embeddings_in',required=True, default=None, type=str)
parser.add_argument('--sent_out',required=True, default=None, type=str)
parser.add_argument('--database_path',required=True, default=None, type=str)
args = parser.parse_args()
con = sqlite3.connect(args.database_path, detect_types=sqlite3.PARSE_DECLTYPES)
def get_sentences(indexes,file_path):
    sents=[]
    cur.execute("SELECT vectors FROM Sentences")
    all_vecs=cur.fetchall()
    for index in indexes:
        cur.execute("SELECT * FROM Sentences WHERE vectors=?",(all_vecs[index][0],))
        arr_vec,sent=cur.fetchone()
        sents.append(sent)
    
    with open(file_path,'w') as output_file:
        wr = csv.writer(output_file)
        wr.writerow(sents)
    


def get_query_vector(filename):
    with open(filename,'rb') as fin:
        a=pickle.load(fin)

    return a

def main():
    cur = con.cursor()
    mj_vec=args.embeddings_in
    target_vec=get_query_vector(mj_vec)
    cosine_sim=[]
    for vec in cur.execute("SELECT  vectors FROM Sentences"):
        cosine_similarity = 1-distance.cosine(vec,target_vec)
        cosine_sim.append(cosine_similarity)
    
    print(len(cosine_sim))
    N=100
    cos_indexes = sorted(range(len(cosine_sim)), key = lambda sub: cosine_sim[sub])[-N:]
    get_sentences(cos_indexes,args.sent_out)


if __name__ == '__main__':
    main()
    
