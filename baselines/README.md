# Baselines - Euphemism Detection & Identification
This repo is for the baseline implementations. 


## Table of Contents
- [1. Download data](#1-Download-data)
- [2. Preprocess data](#2-Preprocess-data)
- [3. Baselines Approaches - Euphemism Detection](#3-Baselines-Approaches---Euphemism-Detection)
    - [3.1. Identify and Select](#31-Identify-and-Select)
    - [3.2. Word2vec Similarities](#32-Word2vec-Similarities)
    - [3.3. Recognizing Euphemisms and Dysphemisms Using Sentiment Analysis](#33-Recognizing-Euphemisms-and-Dysphemisms-Using-Sentiment-Analysis)
    - [3.4. Cantreader](#34-Cantreader)
    - [3.5. BERT](#35-BERT)
    - [3.6. BERT embeddings of the last two layers](#36-BERT-embeddings-of-the-last-two-layers)
- [4. Baselines Approaches - Euphemism Identification](#4-Baselines-Approaches---Euphemism-Identification)
    - [4.1. Word2vec Similarities](#41-Word2vec-Similarities)
    - [4.2. Cos-Similarities](#42-Cos-Similarities)



## 1. Download data
- Download Gab posts from [here](https://files.pushshift.io/gab/), and save them in the same folder.

- Download Reddit corpus (request the `reddit.csv` file from Wanzheng Zhu (wz6@illinois.edu) or Professor Nicolas Christin

- Download Wiki corpus from [here](https://dumps.wikimedia.org/enwiki/), any version shown on the website is fine.



## 2. Preprocess data 
Preprocess Gab Posts:
```bash
python3 preprocess/preprocess_gab.py --gab_folder GAB_FOLDER --save_path SAVE_PATH
```
- `GAB_FOLDER`: the folder that contains the gab corpora

- `SAVE_PATH`: the path to save preprocessed gab texts

Preprocess the Reddit corpus:
```bash
python3 preprocess/preprocess_reddit.py --reddit_path REDDIT_PATH --save_path SAVE_PATH
```
- `REDDIT_PATH`: the path of reddit csv file

- `SAVE_PATH`: the path to save preprocessed reddit corpus



## 3. Baselines Approaches - Euphemism Detection 

### 3.1. Identify and Select
Baseline 1 aims to identify a set of potential euphemisms first (Step 3.1.1) and then select the real euphemisms (Step 3.1.2). 
 
#### 3.1.1 Identify key words
Identify words that are likely to have euphemisms. Here we have three approaches: 

(1) Identify keywords using standard dictionary
```bash
python3 identify_keywords/identify_words_dict.py 
--corpus CORPUS 
--frequency_threshold FREQUENCY_THRESHOLD
--save_path SAVE_PATH
```
- `CORPUS`: the path of preprocessed Gab or Reddit corpora

- `FREQUENCY_THRESHOLD`: the threshold of word frequency. Words with lower frequency in the corpus than this threshold will be exclued from the keywords. The corpora we are dealing with are online texts and tend to be noisy. Infrequent words are likely to be typos or strange symbols, so they are ignored when we identify keywords.

- `SAVE_PATH`: the path to save selected keywords

(2) Identify keywords using TF-IDF scoring
```bash
python3 identify_keywords/identify_words_tfidf.py
--target_corpus TARGET_CORPUS
--wiki_corpus WIKI_CORPUS
--frequency_threshold FREQUENCY_THRESHOLD
--save_path SAVE_PATH
```
- `TARGET_CORPUS`: the path of preprocessed Gab or Reddit corpora

- `WIKI_CORPUS`: the path of downloaded wiki corpus

- `FREQUENCY_THRESHOLD`: the threshold of word frequency. Words with lower frequency in the corpus than this threshold will be exclued from the keywords.

- `SAVE_PATH`: the path to save selected keywords. Words are sorted in decreasing order of their tf-idf score. High score indicates that a word is more likely to have euphemisms.

(3) Identify keywords using a graph based approach: [Determining Code Words in Euphemistic Hate Speech Using Word Embedding Networks](https://www.aclweb.org/anthology/W18-5112.pdf).
```bash
python3 identify_keywords/identify_words_graph.py
--target_corpus TARGET_CORPUS
--wiki_corpus WIKI_CORPUS
--wiki_embed WIKI_EMBED
--save_path SAVE_PATH
```
- `TARGET_CORPUS`: the path of preprocessed Gab or Reddit corpora

- `WIKI_CORPUS`: the path of downloaded wiki corpus

- `WIKI_CORPUS`: the path to save trained wiki embedding

- `SAVE_PATH`: the path to save selected keywords 


#### 3.1.2. Identify euphemisms
Enter the folder `identify_euphemisms/`, and identify the euphemisms of given words from their nearest neighbors in the embedding space. Here we have two approaches to finding euphemisms:

(1) Word2vec embedding based approach
```bash
python3 identify_euphemisms/word2vec_eu_detection.py
--corpus CORPUS
--embed_path EMBED_PATH
--keyword_path KEYWORD_PATH 
--save_path SAVE_PATH
```
- `CORPUS`: the path of preprocessed Gab or Reddit corpora

- `EMBED_PATH`: the path to saved trained word2vec embedding

- `KEYWORD_PATH`: the path of identified keywords in Step 3.1.1.

- `SAVE_PATH`: the path to save euphemisms of the given keywords

(2) Character embedding based approach
```bash
python3 identify_euphemisms/char_eu_detection.py
--corpus CORPUS
--embed_path EMBED_PATH
--keyword_path KEYWORD_PATH 
--save_path SAVE_PATH
```
- `CORPUS`: the path of preprocessed Gab or Reddit corpora

- `EMBED_PATH`: the path to saved trained character embedding model

- `KEYWORD_PATH`: the path of identified keywords in Step 3.1.1.

- `SAVE_PATH`: the path to save euphemisms of the given keywords


#### 3.1.3. Visualize neighbors of euphemisms
```bash
python3 plot/visualize.py 
--corpus CORPUS
--embed_path EMBED_PATH
--save_folder SAVE_FOLDER
```
- `CORPUS`: the path of preprocessed Gab or Reddit corpora

- `EMBED_PATH`: the path to saved trained word2vec embedding

- `SAVE_FOLDER`: the folder to save 2d and 3d plots of keyword and neighbor vectors


### 3.2. Word2vec Similarities
In `word2vec_sim/word2vec_sim.py`, it outputs the nearest neighbors with the seeds. 
The code for evaluation is in `misc.py`.

Besides, it performs euphemism identification as well by comparing the euphemisms embeddings with the drug proper name's embedding. 


### 3.3. Recognizing Euphemisms and Dysphemisms Using Sentiment Analysis
The implementation of [Recognizing Euphemisms and Dysphemisms Using Sentiment Analysis](https://www.aclweb.org/anthology/2020.figlang-1.20.pdf) is in the folder `Recognizing Euphemisms and Dysphemisms Using Sentiment Analysis/`.


### 3.4. Cantreader
The implementation of [Reading Thieves' Cant: Automatically Identifying and Understanding Dark Jargons from Cybercrime Marketplaces](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-yuan_0.pdf) is in the folder `cantreader_calculations/`.

Run `cantreader_calculations/cantreader_precision_recall.py`. 


### 3.5. BERT
The BERT model is used to find the nearest sentences of a keyword. Run `BERT/bert_tuning.py` with the corpus file to obtain the embeddings.

NOTE: Can also be tuned on a GPU, please refer to [this link](https://pypi.org/project/bert-embedding).

`BERT/bert_embedder.py` is used to process the embeddings and store the data onto a database.
`BERT/neighbouring_sentences.py` is used to query the database and select nearest sentences based on cosine similarities.



## 4. Baselines Approaches - Euphemism Identification
### 4.1. Word2vec Similarities
In `word2vec_sim/word2vec_sim.py`, it outputs the nearest neighbors with the seeds. 
The code for evaluation is in `misc.py`.

Besides, it performs euphemism identification as well by comparing the euphemisms embeddings with the drug proper name's embedding. 


### 4.2. Cos-Similarities
In both `word2vec_sim/word2vec_sim.py` and `Main.py`. 

