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

conf = configparser.ConfigParser()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
conf.read(main_path+'\configurations\configurations.ini')
skip = False
df = pd.read_excel(os.getcwd()+conf.get("INPUT","metadati"),sheet_name=2)
#df = df[df.destinatario.str.contains("Capizucchi")]
df = df[df['testo'].notna()]
row_id = df['id_lettera'].values
stopwords = get_stop_words('it')
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
ft = fasttext.load_model(main_path+'/data/cc.it.300.bin')

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



def removeNonAlpha(text):
	import re
	text = re.sub("[^a-zA-Z0-9]+", " ",text)
	return text


def removeStopWords(text,stopwords=stopwords,remove_short_words=True):
    words = text.split()
    if remove_short_words:
        words = [i for i in words if len(i) > 2]
    words = [word for word in words if not word in set(stopwords)]
    text = ' '.join(words)
    return text

def lemmatize(text,tagger=tagger):
    tags = tagger.tag_text(text)
    tags = treetaggerwrapper.make_tags(tags)
    cleaned_text = []
    for elem in tags:
        lemma = elem.lemma
        cleaned_text.append(lemma)
    text = ' '.join(cleaned_text)
    return text

def calculate_tf_idf(corpus,rownames):
    cv = TfidfVectorizer(ngram_range=(1, 1), max_features=50000)
    X = cv.fit_transform(corpus)
    Y = X.toarray()
    count_vect_df = pd.DataFrame(Y, columns=cv.get_feature_names(),index=rownames)
    return count_vect_df,X

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



cleaned_corpus = []
for elem in df.testo:
    try:
        elem = removeNonAlpha(elem)
        elem = removeStopWords(elem)
        elem = lemmatize(elem)
    except:
        pass
    cleaned_corpus.append(elem)
df_tf_idf, raw_matrix = calculate_tf_idf(corpus=cleaned_corpus,rownames=row_id)
final_result = avg_w2vec(df_tf_idf,model=ft)
#df_cleaned_letters = pd.DataFrame(cleaned_corpus,columns=['id','testo'])
#df_cleaned_letters.to_csv("lemmatized_cleanded_letters.csv")



# calculate cosine similarity for the embedded vectors of the job positions
cosine_sim = np.round(cosine_similarity(final_result, final_result),3)
df_cosine = pd.DataFrame(cosine_sim,index=df.id_lettera, columns=df.id_lettera)
df_cosine.to_excel(os.getcwd()+conf.get("OUTPUT","first_algorithm")+datetime.now().strftime("%H-%M-%S")+".xlsx")

print(1)