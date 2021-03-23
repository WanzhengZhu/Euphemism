![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Self-Supervised Euphemism Detection and Identification for Content Moderation
This repo is the Python 3 implementation of __Self-Supervised Euphemism Detection and Identification for Content Moderation__ (42nd IEEE Symposium on Security and Privacy 2021).


## Table of Contents
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Data](#Data)
- [Code](#Code)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)


## Introduction
This project aims at __Euphemism Detection__ and __Euphemism Identification__. 


## Requirements
The code is based on Python 3.7. Please install the dependencies as below:  
```
pip install -r requirements.txt
```


## Data
Due to the license issue, we will not distribute the dataset ourselves, but we will direct the readers to their respective sources.  

__Drug__: 
- _Raw Text Corpus_: Please request the raw text corpus --- `reddit.csv` from Wanzheng Zhu (wz6@illinois.edu) or Professor Nicolas Christin.
- _Ground Truth_: we summarize the drug euphemism ground truth list (provided by the DEA Intelligence Report -- [Slang Terms and Code Words: A Reference for Law Enforcement Personnel](https://www.dea.gov/sites/default/files/2018-07/DIR-022-18.pdf)) in `data/euphemism_answer_drug.txt` and `data/target_keywords_drug.txt`. 

__Weapon__: 
- _Raw Text Corpus_: Please request the dataset from [What is gab: A bastion of free speech or an alt-right echo chamber (Zanettou et al. 2018)](https://dl.acm.org/doi/pdf/10.1145/3184558.3191531), [Identifying products in online cybercrime marketplaces: A dataset for fine-grained domain adaptation (Durrett et al. 2017)](https://www.aclweb.org/anthology/D17-1275.pdf), [Tools for Automated Analysis of Cybercriminal Markets (Portnoff et al. 2017)](https://dl.acm.org/doi/pdf/10.1145/3038912.3052600), and the examples on [Slangpedia](https://slangpedia.org/). 
- _Ground Truth_:  Please refer to [The Online Slang Dictionary](http://onlineslangdictionary.com/), [Slangpedia](https://slangpedia.org/), and [The Urban Thesaurus](https://urbanthesaurus.org/).  

__Sexuality__: 
- _Raw Text Corpus_: We use 2,894,869 processed [Gab](https://gab.com/) posts, collected from Jan 2018 to Oct 2018 by [PushShift](https://files.pushshift.io/gab/). 
- _Ground Truth_: Please refer to [The Online Slang Dictionary](http://onlineslangdictionary.com/).  

__Sample__:
- _Raw Text Corpus_: we provide a sample dataset `data/sample.txt` for the readers to run the code.
- _Ground Truth_: same as the Drug dataset (see `data/euphemism_answer_drug.txt` and `data/target_keywords_drug.txt`).  
- This Sample dataset is only for you to play with the code and it does not represent any reliable results. 


## Code
### 1. Fine-tune the BERT model. 
Please refer to [this link from Hugging Face](https://github.com/huggingface/transformers/tree/master/examples/language-modeling) to fine-tune a BERT on a raw text corpus.

You may download our pre-trained BERT model on the `reddit` text corpus (from the Drug dataset) [here](https://drive.google.com/file/d/1kLZ0IWchWywXaxs61Vk6-eFmlx2rexU3/view?usp=sharing). Please unzip it and put it under `data/`.

### 2. Euphemism Detection and Euphemism Identification
```
python ./Main.py --dataset sample --target drug  
```
You may find other tunable arguments --- `c1`, `c2` and `coarse` to specify different classifiers for euphemism identification. 
Please go to `Main.py` to find out their meanings. 


### Baselines:
Please refer to `baselines/README.md`. 


## Acknowledgement
We use the code [here](https://github.com/prakashpandey9/Text-Classification-Pytorch) for the text classification in Pytorch. 


## Citation
```bibtex
@inproceedings{zhu2021selfsupervised,
    title = {Self-Supervised Euphemism Detection and Identification for Content Moderation},
    author = {Zhu, Wanzheng and Gong, Hongyu and Bansal, Rohan and Weinberg, Zachary and Christin, Nicolas and Fanti, Giulia and Bhat, Suma},
    booktitle = {42nd IEEE Symposium on Security and Privacy},
    year = {2021}
}
```
