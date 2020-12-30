# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:48:50 2020

@author: M Shoaib
"""


import os
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
all_data=pd.read_csv('Shakespeare_data.csv')
data=pd.DataFrame()
data['Player']=all_data['Player']
data['lines']=all_data['PlayerLine']
data.dropna(subset = ["Player"], inplace=True)
def clean1(txt):
    token=txt.lower()
    token=' '.join(re.sub("(@[A-Za-z]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",token).split())
    # txt1=re.sub('[0-9]','',txt1)
    return token
data['lines']=data['lines'].apply(clean1)
data['Player']=data['Player'].apply(clean1)
lst1=data['lines']
lst1=lst1.tolist()
text=[]
for sent in lst1:
  for word in sent.split():
    text.append(word)
vocab=sorted(set(text))
char2idx={u:i for i,u in enumerate(vocab)}
idx2char=np.array(vocab)
def text_to_int(text):
    return np.array([char2idx[c]for c in text])
text_as_int=text_to_int(text)
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ' '.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))
seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target) 
for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape)
print(len(example_batch_predictions))
print(example_batch_predictions)
pred = example_batch_predictions[0]
print(len(pred))
print(pred)

time_pred = pred[0]
print(len(time_pred))
print(time_pred)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
# Directory where the checkpoints will be saved
model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))
