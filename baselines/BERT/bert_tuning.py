#Module used to tune BERT embeddings
#Can run on GPU, please check the BertEmbedding documentation for more details
from bert_embedding import BertEmbedding
import pickle
import argparse
from tqdm import tqdm
bert_embedding = BertEmbedding()

def batch_create(iterable, n=1):
    #Used to create batches of an iterable object(e.g-List)
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def main():
    # reddit_in_text_path = '/Volumes/My Passport for Mac/Rohan/Euphemism_Detection_2020/Wiki_data/wiki_whole.txt'
    # reddit_out_file='/Volumes/My Passport for Mac/Rohan/Euphemism_Detection_2020/Wiki_data/wiki_embeddings.pkl'
    parser = argparse.ArgumentParser(description='Pass the files.')
    parser.add_argument('--corpus_in',required=True, default=None, type=str)
    parser.add_argument('--vectors_out',required=True, default=None, type=str)
    args = parser.parse_args()
    
    #Read the input file
    with open(args.corpus_in,'r') as fin:
        str_1 = fin.read()
    sentences=str_1.split("\n")
    sentences=list(filter(None,sentences))
    sentence_batches=batch_create(sentences,3000)
    #Create batches of the text file to avoid Memory Issues
    print("#Start training the vectors")
    for batch in tqdm(sentence_batches):
        try:
            #Train the embeddings in batches
            reddit_bert_vector=bert_embedding(batch)
            # Dump the embeddings 
            with open(args.vectors_out,'ab+') as fout:
                pickle.dump(reddit_bert_vector,fout)  
            pass
        except:
            continue
        

if __name__ == "__main__":
    main()
