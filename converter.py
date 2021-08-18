import fasttext.util
import os
import pandas as pd
import numpy as np

main_path = os.getcwd()
def calculate_dataframe(tf_idf_matrix,model):
    dict_list = []
    for word in tf_idf_matrix.columns:
        dict = {"word":word,"value":model.get_word_vector(word)}
        dict_list.append(dict)
    df = pd.DataFrame(dict_list)
    final_df = pd.DataFrame(df['value'].to_list(), index=df["word"])
    return final_df

def calculate_vec(converted_df,ft,tf_idf_matrix,column_df):
    dict_list = []
    names_col_list = []
    all_words = tf_idf_matrix.columns.to_list()
    for word in all_words:
        tf_value = tf_idf_matrix.loc[column_df,word]
        if tf_value > 0:
            dict = {"word":word, "value":ft.get_word_vector(word)*tf_value}
            dict_list.append(dict['value'])
        else:
            dict_list.append(np.zeros(converted_df.shape[1]))
        vector = list(np.arange(300))
        single_name_list = [word + str(s) for s in vector]
        names_col_list.append(single_name_list)
    flat_list = [item for sublist in dict_list for item in sublist]
    total_name_list = [item for sublist in names_col_list for item in sublist]
    df = pd.DataFrame(flat_list,index=total_name_list)
    return df


