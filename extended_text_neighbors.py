import treetaggerwrapper
from stop_words import get_stop_words
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import configparser
from datetime import datetime
from collections import Counter
from datetime import datetime
import pandas as pd
import fasttext.util
import neighbors


start = datetime.now().strftime('%H-%M-%S')
conf = configparser.ConfigParser()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
conf.read(main_path+'\configurations\configurations.ini')
df = pd.read_excel(os.getcwd() + conf.get("INPUT", "lemmatized"))
ft = fasttext.load_model(main_path+'/data/cc.it.300.bin')


def extend_text(text):
    text = text.split()
    full_text = []
    for word in text:
        new_words = []
        full_text.append(new_words)
    flat_list = [item for sublist in full_text for item in sublist]
    new_text = ' '.join(flat_list)
    return new_text


new = extend_text(df.loc[0,'testo'])