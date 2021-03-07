import pickle
import numpy as np
import io
import sqlite3
from tqdm import tqdm
import argparse
from hdf5_save import adapt_array,convert_array

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
con = sqlite3.connect("sentence_and_vectors_new.db", detect_types=sqlite3.PARSE_DECLTYPES)

def get_vector_target_word(embedding, target_word):
    # Go through each of the sentences and see if the target_word is present
    # Take the sum total of the vectors 
    # Take the avg of the vectors
    # return the position of the word in vector space
    vec_sum=0
    n=0
    for sent in embedding:
        if target_word in sent[0]:
            n +=1
            index_temp = sent[0].index(target_word)
            vec_sum= np.add(vec_sum,sent[1][index_temp])
            
    
    return(vec_sum,n)
  
def get_sentence_vectors(embeddings):
    #return a dictinoary with sentences and the avg of the word vectors 
    # find the sentence embedding
    sentences=[]
    vectors=[]
    for sent in embeddings:
        if (len(sent[1])>0):
            complete_sent = " ".join(sent[0])
            sent_vector=(sum(sent[1])/len(sent[1]))
            sentences.append(complete_sent)
            vectors.append(sent_vector)
    
    return(sentences,vectors)
        
def process_embeddibngs(embeddings,word):
    vector_sum,n=get_vector_target_word(embeddings,word)
    return (vector_sum,n)


def save_word_vector(sum_vec,num,path):
    vec= np.divide(sum_vec,num)
    with open(path,'wb') as fout:
        pickle.dump(vec,fout)

def main():
    cur = con.cursor()
    
    ########## Uncomment to create a new table
    # cur.execute("create table Sentences (vectors array, sentences text)")
    
    
    parser = argparse.ArgumentParser(description='Pass the files.')
    parser.add_argument('--embeddings_in',required=True, default=None, type=str)
    parser.add_argument('--word_vec_out',required=True, default=None, type=str)
    args = parser.parse_args()
    
    file_path_word_emb=args.word_vec_out
    embedding_path = args.embeddings_in
    N=0
    vec_sum =0
    i=0
    key_word='cocaine'
    with open(embedding_path,'rb') as temp_fin:
        while 1:
            try:
                con.commit()
                i +=1
                print(f"Current batch is {i}")
                obj=(pickle.load(temp_fin))
                new_sum,n=process_embeddibngs(obj,key_word)
                vec_sum=np.add(vec_sum,new_sum)
                N +=n
                sentences,sentences_vec=get_sentence_vectors(obj)
                for sent, sent_vec in zip(sentences,sentences_vec):
                    cur.execute("INSERT INTO Sentences VALUES (?,?)",(sent_vec,sent,))

            except EOFError:
                break
        
        save_word_vector(vec_sum,N,file_path_word_emb)
        con.close()

if __name__ == '__main__':
    main()
    
