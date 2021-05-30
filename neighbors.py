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
from collections import Counter

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

df = pd.read_excel(os.getcwd()+conf.get("INPUT","lemmatized"))
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


def neighbor_value(word,fasttext,k=10):
    words= fasttext.get_nearest_neighbors(word,k)
    df = pd.DataFrame(words,columns=['value','key'])
    base_row = {'value':1,'key':word}
    df = df.append(base_row,ignore_index=True)
    return df

def calculate_similarity(new_letter, old_letter, fasttext, neighbors=10):
    new_letter = new_letter.split()
    old_letter = old_letter.split()
    dict_counter = Counter(old_letter)
    similarity_value = 0
    final_list = []
    for word in new_letter:
        df_n = neighbor_value(word=word,fasttext=fasttext, k=neighbors)
        all_words = list(set(df_n.key) & set(old_letter))
        if len(all_words) > 0:
            for elem in all_words:
                count_words = dict_counter[elem]
                value = df_n.loc[df_n['key'] == elem, 'value'].values[0]
                dict = {"word":elem, "value":count_words*value}
                final_list.append(dict)
                new_letter[:] = [x for x in new_letter if x != elem]
                similarity_value = similarity_value + value
    similarity_value = similarity_value/len(old_letter)
    return final_list, similarity_value


final_list, similarity_value = calculate_similarity(df.loc[0,'testo'],df.loc[1,'testo'],fasttext=ft)
print(1)


pass



print(1)

