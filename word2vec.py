# Importing necessary libraries
import pandas as pd
import numpy as np
import fasttext.util
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
from src import connection
from src import cleaner

def avg_w2vec(tf_idf_matrix,model):
    """
    calculates similarity results using w2vec average and tf idf matrix
    :param tf_idf_matrix: tf idf matrix
    :param model: fast text object
    :return: results of similarities between texts
    """
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



def save_lemmatized_text(df,cleaned_coprus,column_name='testo',save=True):
    """
    save lemmatized text in dataframe
    :param df: starting df with not lemmatized column
    :param cleaned_coprus: lemmatized text
    :param column_name: column name of lemmatized df
    :param save: save to excel
    :return:
    """
    del df[column_name]
    df[column_name] = cleaned_coprus
    if save:
        df.to_excel(main_path+'/data/df_lemmatized.xlsx',index=False)
    return df

def calculate_tf_idf(corpus,max_df=0.4): # removed rownames as index of matrix cause no id for now
    cv = TfidfVectorizer(ngram_range=(1, 1), max_features=50000,max_df=max_df)
    X = cv.fit_transform(corpus)
    Y = X.toarray()
    count_vect_df = pd.DataFrame(Y, columns=cv.get_feature_names())# removed index = rownames
    return count_vect_df,X


def graph_to_pandas(graph):
    list = graph.nodes.match("Letter").all()
    return pd.DataFrame(list)


if __name__ == '__main__':
    start = datetime.now()
    conf = connection.get_conf()
    graph = connection.connect(conf)
    df = graph_to_pandas(graph)
    testo = conf.get("ITEMS","testo")
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    
    df = df[df['transcription'].notna()]
    #row_id = df['id_lettera'].values
    stopwords = get_stop_words('it')
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    cleaned_corpus = clean_text(df,stopwords=stopwords,tagger=tagger,column=testo)
    df = save_lemmatized_text(df=df, cleaned_coprus=cleaned_corpus, column_name=testo, save=True)
    df_tf_idf, raw_matrix = calculate_tf_idf(corpus=cleaned_corpus) # rownames = row_id when switched to db
    final_result = avg_w2vec(df_tf_idf,model=ft)
    # calculate cosine similarity for the embedded vectors of the job positions
    cosine_sim = np.round(cosine_similarity(final_result, final_result),3)
    df_cosine = pd.DataFrame(cosine_sim)
    df_cosine.to_excel(os.getcwd()+conf.get("OUTPUT","first_algorithm")+datetime.now().strftime("%d-%m-%y-%H-%M-%S")+".xlsx")
    print(datetime.now() - start)
