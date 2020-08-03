import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import parameters
import model
print(tf.__version__)

# Load data

train_df = pd.read_csv('train.csv').fillna(' ')
x = train_df['comment_text'].values
y = train_df['toxic'].values

#Tokenize

x_tokenizer = text.Tokenizer(parameters.max_features)
x_tokenizer.fit_on_texts(list(x))
x_tokenized = x_tokenizer.texts_to_sequences(x) #list of lists(containing numbers), so basically a list of sequences, not a numpy array
#pad_sequences:transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array of shape 
x_train_val = sequence.pad_sequences(x_tokenized, maxlen=parameters.max_text_length)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((parameters.max_features, parameters.embedding_dims))
for word, index in x_tokenizer.word_index.items():
  if index > max_features -1:
    break
  else:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[index] = embedding_vector

#train

model = model.classifier(parameters.output_classes)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.15, random_state=1)
model.fit(x_train, y_train,
          batch_size=parameters.batch_size,
          epochs=parameters.epochs,
          validation_data=(x_val, y_val))

#evaluate
model.evaluate(x_val, y_val, batch_size=128)
test_df = pd.read_csv('./test.csv')
x_test = test_df['comment_text'].values
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)
y_testing = model.predict(x_testing, verbose = 1, batch_size=32)

test_df['Toxic'] = ['not toxic' if x < .5 else 'toxic' for x in y_testing]
test_df[['comment_text', 'Toxic']].head(20)#.sample(20, random_state=1)