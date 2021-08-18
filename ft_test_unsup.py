import fasttext.util
import os
import fasttext
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
mamma1 = ft.get_word_vector("mamma")
tasto1 = ft.get_word_vector("tasto")
model = fasttext.train_supervised(r"C:\Users\black\OneDrive\Desktop\Alex\valenti\pythonProject\algorithm\data\rubbish.txt")
mamma2 = model.get_word_vector("mamma")
tasto2 = model.get_word_vector("tasto")

print(1)
