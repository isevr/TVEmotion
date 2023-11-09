import pandas as pd
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
import re
from keras.preprocessing.text import Tokenizer

class Tokenization:
    def __init__(self, dataset_path: str, text_col: str, labels_col: str):
        self.df = pd.read_csv(dataset_path)
        self.text_col = text_col
        self.labels_col = labels_col

    def preprocess_utility(self, text: str):
        t = text.lower()
        t = re.sub(r'@[^ ]*',r'',t)
        t = re.sub(r'\W+',r' ',t)
        t = re.sub(r'(https|quot|http)', '', t)
        t = re.sub(r'\b(?!(?:ai)\b)\w{3}\b','', t)
        t = re.sub(r'http?://\S+|www\.\S+','',t)
        stopwords_list = stopwords.words('english')
        txt = ' '.join([word for word in t.split() if word not in stopwords_list])
        return txt


    def preprocess(self):
        self.lemmatizer = WordNetLemmatizer()
        self.df['preprocessed'] = [' '.join([self.lemmatizer.lemmatize(self.preprocess_utility(txt))])
                    .strip() for txt in self.df[self.text_col]]
        self.texts = self.df['preprocessed'].values
        return self.texts

    def tokenizer(self):
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.texts)
        return self.tokenizer

    def sequences(self):
        self.text_seq = self.tokenizer.texts_to_sequences(self.text_col)
        self.padded_seq = pad_sequences(self.text_seq, maxlen=50, padding='post')
        return self.padded_seq