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

skip = False
if skip:

    word1 = "mamma"
    word2 = "madre"
    word3 = "papà"
    word4 = "Tirannosauro"
    vec1 = ft.get_word_vector(word1)
    vec2 = ft.get_word_vector(word2)
    vec3 = ft.get_word_vector(word3)
    vec4 = ft.get_word_vector(word4)

    dist = np.round(distance.euclidean(vec1,vec2),4)

conf = configparser.ConfigParser()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
conf.read(main_path+'\configurations\configurations.ini')
skip = False
def save_lemmatized_text(df,cleaned_coprus,column_name='testo'):
    del df[column_name]
    df[column_name] = cleaned_coprus
    return df

df = pd.read_excel(os.getcwd()+conf.get("INPUT","metadati"),sheet_name=2)
df = df[df['testo'].notna()]
row_id = df['id_lettera'].values
stopwords = get_stop_words('it')
stopwords = add_stopwords(main_path+'/data/stp-aggettivi.txt',stopwords=stopwords)
stopwords = add_stopwords(main_path+'/data/stp-varie.txt',stopwords=stopwords)
stopwords = add_stopwords(main_path+'/data/stp-verbi.txt',stopwords=stopwords)
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
#ft = fasttext.load_model(main_path+'/data/cc.it.300.bin')
cleaned_corpus = clean_text(df,stopwords=stopwords,tagger=tagger, column='testo')
df = save_lemmatized_text(df=df,cleaned_coprus=cleaned_corpus,column_name='testo')

print(1)

def similarity_doc(df):
    a = 1
