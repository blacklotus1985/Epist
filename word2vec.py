# Importing necessary libraries
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
import re

def add_stopwords(file,stopwords):
    with open(file) as f:
        contents = f.read()
        contents = contents.splitlines()
        stopwords.extend(contents)
        return stopwords

def removeNonAlpha(text):
	import re
	text = re.sub("[^a-zA-Z0-9]+", " ",text)
	return text


def removeStopWords(text,stopwords,remove_short_words=True):
    words = text.split()
    if remove_short_words:
        words = [i for i in words if len(i) > 2]
    words = [word for word in words if not word in set(stopwords)]
    text = ' '.join(words)
    return text

def lemmatize(text,tagger):
    tags = tagger.tag_text(text)
    tags = treetaggerwrapper.make_tags(tags)
    cleaned_text = []
    for elem in tags:
        lemma = elem.lemma
        lemma = re.sub(r'\w+\|\b', '', lemma)
        cleaned_text.append(lemma)
    text = ' '.join(cleaned_text)
    return text



def avg_w2vec(tf_idf_matrix,model):
    words = list(tf_idf_matrix.columns)
    big_list = []
    small_list = []
    for index, row in tf_idf_matrix.iterrows():
        array = row.values
        word_index = 0
        for word in words:
            tf = array[word_index]*100
            if tf:
                vector_word = model.get_word_vector(word)
                avg = np.average(vector_word)*100
                result_avg = np.round((tf * avg),3)
                small_list.append(result_avg)
            else:
                small_list.append(0)
            word_index +=1
        big_list.append(small_list)
        small_list = []
    df_result = pd.DataFrame(big_list,index=tf_idf_matrix.index,columns=tf_idf_matrix.columns)
    df_result = df_result/100
    return df_result

def clean_text(df,stopwords,tagger,column='testo'):
    """
    clean dataframe of letters
    :param df: dataframe with metadata
    :param column: column to clean
    :return: dataframe cleaned
    """
    cleaned_corpus = []
    for elem in df[column]:
        elem = removeNonAlpha(elem)
        elem = removeStopWords(elem,stopwords=stopwords)
        elem = lemmatize(elem,tagger)
        pass
        cleaned_corpus.append(elem)
    return cleaned_corpus

def calculate_tf_idf(corpus,rownames):
    cv = TfidfVectorizer(ngram_range=(1, 1), max_features=50000)
    X = cv.fit_transform(corpus)
    Y = X.toarray()
    count_vect_df = pd.DataFrame(Y, columns=cv.get_feature_names(),index=rownames)
    return count_vect_df,X

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    conf.read(main_path + '\configurations\configurations.ini')
    skip = False
    df = pd.read_excel(os.getcwd() + conf.get("INPUT", "metadati"), sheet_name=2)

    df = df[df['testo'].notna()]
    row_id = df['id_lettera'].values
    stopwords = get_stop_words('it')
    stopwords = add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    cleaned_corpus = clean_text(df,stopwords=stopwords,tagger=tagger,column='testo')
    df_tf_idf, raw_matrix = calculate_tf_idf(corpus=cleaned_corpus,rownames=row_id)
    final_result = avg_w2vec(df_tf_idf,model=ft)
    # calculate cosine similarity for the embedded vectors of the job positions
    cosine_sim = np.round(cosine_similarity(final_result, final_result),3)
    df_cosine = pd.DataFrame(cosine_sim,index=df.id_lettera, columns=df.id_lettera)
    df_cosine.to_excel(os.getcwd()+conf.get("OUTPUT","first_algorithm")+datetime.now().strftime("%d-%m-%y-%H-%M-%S")+".xlsx")

