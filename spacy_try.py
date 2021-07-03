import spacy
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
import word2vec

nlp = spacy.load('it_core_news_lg')



if __name__ == '__main__':
    start = datetime.now()
    conf = connection.get_conf()
    graph = connection.connect(conf)
    df = word2vec.graph_to_pandas(graph)
    testo = conf.get("ITEMS", "testo")
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    df = df[df['transcription'].notna()]
    row_id = df['letter_id'].values
    stopwords = get_stop_words('it')
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    cleaned_corpus = cleaner.clean_text(df, stopwords=stopwords, tagger=tagger, column=testo)
    df = word2vec.save_lemmatized_text(df=df, cleaned_coprus=cleaned_corpus, column_name=testo, save=True)
