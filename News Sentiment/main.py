# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:36:56 2020

@author: M Shoaib
"""


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing import sequence
import keras
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.neural_network import BernoulliRBM
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
all_data=pd.read_json('news_data.json',lines=True)
# print(all_data['category'].unique())
# print(all_data.isnull().sum())
print(all_data['category'].value_counts())
new_data=pd.DataFrame()
new_data[['labels','headlines','short_des']]=all_data[['category','headline','short_description']]
total_cate=len(all_data['category'].unique())
def clean1(txt):
    txt1=txt.lower()
    txt1=' '.join(re.sub("(@[A-Za-z]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",txt1).split())
    # txt1=re.sub('[0-9]','',txt1)
    return txt1
new_data['Cleaned_headlines']=new_data['headlines'].apply(clean1)
new_data['Cleaned_short_des']=new_data['short_des'].apply(clean1)
test=pd.DataFrame()
test['head']=new_data['Cleaned_headlines']
arr=np.array(test['head'])
tokenizer=Tokenizer(num_words=None,oov_token='<OOV>')

tokenizer.fit_on_texts(arr)
word_index=tokenizer.word_index
arr2=tokenizer.texts_to_sequences(arr)
le=LabelEncoder()
one_hot=OneHotEncoder()
test['category_to_int']=le.fit_transform(all_data['category'])
enc_df=pd.DataFrame(one_hot.fit_transform(test[['category_to_int']]).toarray())
test=test.join(enc_df)
X=new_data[['Cleaned_headlines','Cleaned_short_des']]
labels=enc_df
total_words=len(word_index)
labels=np.array(labels)
X_train,X_test,y_train,y_test=train_test_split(arr2,labels,test_size=.2,random_state=42)
max_len=100
batch_size=128
X_train=sequence.pad_sequences(X_train,max_len)
X_test=sequence.pad_sequences(X_test,max_len)
model=tf.keras.Sequential([
      tf.keras.layers.Embedding(total_words+1,100),
      
      # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
      tf.keras.layers.LSTM(100,dropout=0.2),
     
      # tf.keras.layers.Dense(64,activation='softmax'),
      tf.keras.layers.Dense(41,activation='softmax')      
])
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10,batch_size=100)
results = model.evaluate(X_test, y_test)
print(results)
# model.save('100_LSTM.h5')
def pred1(txt):
   
  words=word_index  
  token=txt
  tokens=clean1(token)
  tokens=tokens.split()
  tokens=[words[word]if word in words else 0 for word in tokens]
  padded=sequence.pad_sequences([tokens],max_len)[0]
  pred=np.zeros((1,100))
  pred[0]=padded
  res=model.predict(pred)
  class_names = ['CRIME', 'ENTERTAINMENT', 'WORLD NEWS', 'IMPACT', 'POLITICS',
       'WEIRD NEWS', 'BLACK VOICES', 'WOMEN', 'COMEDY', 'QUEER VOICES',
       'SPORTS', 'BUSINESS', 'TRAVEL', 'MEDIA', 'TECH', 'RELIGION',
       'SCIENCE', 'LATINO VOICES', 'EDUCATION', 'COLLEGE', 'PARENTS',
       'ARTS & CULTURE', 'STYLE', 'GREEN', 'TASTE', 'HEALTHY LIVING',
       'THE WORLDPOST', 'GOOD NEWS', 'WORLDPOST', 'FIFTY', 'ARTS',
       'WELLNESS', 'PARENTING', 'HOME & LIVING', 'STYLE & BEAUTY',
       'DIVORCE', 'WEDDINGS', 'FOOD & DRINK', 'MONEY', 'ENVIRONMENT',
       'CULTURE & ARTS']
  print(np.argmax(res))
  
  print(class_names[np.argmax(res)])
  # print(res[0][0][0])
  # if res[0][0][0]<=[0.5]:
  #   print('Negative Statement')
  # else:
  #   print('Positive Statement')
        
while True:
  a='Quit'
  t=input('Enter Statement: ')
  if t!=a.lower():
    pred1(t)
  else:
    break

