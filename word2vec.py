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
import treetaggerwrapper
from src import cleaner
import converter
import corrector
from src import cleaner
from stop_words import get_stop_words




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
        df.to_excel(os.getcwd()+'/data/df_lemmatized.xlsx',index=False)
    return df

def calculate_tf_idf(corpus,rownames, max_df=0.4): # removed rownames as index of matrix cause no id for now
    cv = TfidfVectorizer(ngram_range=(1, 1), max_features=50000,max_df=max_df)
    X = cv.fit_transform(corpus)
    Y = X.toarray()
    count_vect_df = pd.DataFrame(Y, columns=cv.get_feature_names(),index=rownames)# removed index = rownames
    return count_vect_df,X


def graph_to_pandas(graph):
    list = graph.nodes.match("Letter").all()
    return pd.DataFrame(list)


if __name__ == '__main__':
    start = datetime.now()
    print ("algorithm started at {}".format(start))
    db = False
    paragaph = True
    conf = connection.get_conf()
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    if db:
        graph = connection.connect(conf)
        df = graph_to_pandas(graph)
        testo = conf.get("ITEMS","testo")
        df = df[df['transcription'].notna()]
        row_id = df['letter_id'].values
        df_read = pd.read_excel(main_path + conf.get("INPUT","lemmatized"))
        cleaned_corpus = df_read.transcription.values.astype('U')
    else:
        print("started reading metadati")
        df_read = pd.read_excel(os.getcwd() + conf.get("INPUT", "metadati"),sheet_name=2)
        if paragaph:
            df_read['testo'] = df_read['testo'].str.replace('\r', ' ').str.split('\n')
            df_read = df_read.explode('testo')
        #res = cleaner.split_paragraphs(df_read.loc[66,"testo"])
        row_id = df_read['id_lettera'].values
        #df_read = pd.read_excel(main_path + conf.get("INPUT", "lemmatized_old"))
        testo = []
        print("started stopwords")
        stopwords = get_stop_words('it')
        stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
        stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
        stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
        cleaned_corpus = cleaner.clean_text(df_read, stopwords=stopwords, tagger=tagger, column='testo')
        counter = 0
        """
        for row in cleaned_corpus:
            row = cleaner.removeNonAlpha(row)
            json = corrector.correct_letter(row,debug=False)
            testo.append(json)
            print(counter)
            counter = counter +1
            print(counter)
        #df_final = pd.DataFrame(testo,index=df_read['id_lettera'])
        """

    # @@@@@@@@@@ IF CHANGED TO DB = True CHANGE "testo" with "cleaned_corpus" !!!!
    print("started tf idf calc")
    df_tf_idf, raw_matrix = calculate_tf_idf(corpus=cleaned_corpus,rownames=row_id) # rownames = row_id when switched to db
    converted_df = converter.calculate_dataframe(df_tf_idf,model=ft)
    total_big_df = converter.total_w2vec(converted_df, ft, df_tf_idf)
    #final_result = avg_w2vec(df_tf_idf,model=ft)
    cosine_sim = np.round(cosine_similarity(total_big_df, total_big_df),3)
    df_cosine = pd.DataFrame(cosine_sim, index=row_id,columns=[row_id])
    df_cosine.to_excel(os.getcwd()+conf.get("OUTPUT","bigw2vec")+datetime.now().strftime("%d-%m-%y-%H-%M-%S")+".xlsx")
    a = df_cosine.to_numpy().flatten()
    b = [x for x in a if x < 0.85]
    b = np.array(b)
    sort_index_array = np.argsort(b)
    sorted_array = b[sort_index_array]
    rslt = sorted_array[-20:]
    rslt = rslt[1::2]
    dict_list = []
    for elem in rslt:
        row, column = np.where(df_cosine == elem)
        ind = list(zip(df_cosine.index[row], df_cosine.columns[column]))
        dict = {"letters":ind, "value":elem}
        dict_list.append(dict)

    best_df = pd.DataFrame(dict_list,columns=['letters','value'])
    best_df.to_excel(os.getcwd() + conf.get("OUTPUT", "best_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")

    print(datetime.now() - start)
