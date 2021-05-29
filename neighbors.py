from word2vec import add_stopwords, clean_text
import pandas as pd
import numpy as np
import fasttext.util
import nltk
from scipy.spatial import distance
import treetaggerwrapper
from stop_words import get_stop_words
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import configparser
from datetime import datetime

conf = configparser.ConfigParser()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
conf.read(main_path+'\configurations\configurations.ini')
skip = False
def save_lemmatized_text(df,cleaned_coprus,column_name='testo',save=False):
    del df[column_name]
    df[column_name] = cleaned_coprus
    if save:
        df.to_excel(main_path+'/data/df_lemmatized.xlsx',index=False)
    return df

df = pd.read_excel(os.getcwd()+conf.get("INPUT","metadati"),sheet_name=2)
df = df[df['testo'].notna()]
row_id = df['id_lettera'].values
stopwords = get_stop_words('it')
stopwords = add_stopwords(main_path+'/data/stp-aggettivi.txt',stopwords=stopwords)
stopwords = add_stopwords(main_path+'/data/stp-varie.txt',stopwords=stopwords)
stopwords = add_stopwords(main_path+'/data/stp-verbi.txt',stopwords=stopwords)
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
ft = fasttext.load_model(main_path+'/data/cc.it.300.bin')
cleaned_corpus = clean_text(df,stopwords=stopwords,tagger=tagger, column='testo')
df = save_lemmatized_text(df=df,cleaned_coprus=cleaned_corpus,column_name='testo',save=True)


def neighbor_value(word,fasttext,k=20):
    words= fasttext.get_nearest_neighbors(word,k)
    df = pd.DataFrame(words,columns=['value','key'])
    base_row = {'value':1,'key':word}
    df = df.append(base_row,ignore_index=True)
    return df

df_neighbor = neighbor_value("cane",ft,10)
print(1)

