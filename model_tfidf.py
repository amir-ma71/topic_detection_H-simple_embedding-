from keras import Sequential, metrics
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout,Conv1D,Flatten, MaxPool1D
from sklearn.model_selection import train_test_split
import numpy as np
from pre_pros.pre_pro import stop_word_remover, tokenize
from pre_pros.vectorizer import *
from pre_pros.conf_matrix import *
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

print("reading Data . . .")
dataset = pd.read_csv("./data_compact/final.csv", encoding="utf-8", quoting=1, sep="~")

max_lenght = 250
label_list = ["news", "sport", "health", "entertainment", "economy", "tech"]

print("vectorizing . . .")
vectors = tf_idf(dataset, label_list=label_list)
x = np.array(vectors[0])
y = np.array(vectors[1])
print("vectorizing Done . . .")

from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, random_state=42)
print(8)

MAX_NB_WORDS = 100000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250

model = Sequential()

model.add(Embedding(MAX_NB_WORDS, 100, input_length=x.shape[1]))
# model.add(SpatialDropout1D(0.2))
model.add(Conv1D(256,3))
model.add(MaxPool1D())
model.add(Dropout(0.3))
model.add(Conv1D(256,3))
model.add(MaxPool1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

epochs = 10
batch_size = 128

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, verbose=0)

print("loss -->", loss)
print("accuracy -->", accuracy)
print("f1_score -->", f1_score)
print("precision -->", precision)
print("recall -->", recall)


y_pred = model.predict(X_test)

l_list_conf = ["news", "sport", "healt", "enter", "econ", "tech"]
print_matrix(conf(Y_test, y_pred, l_list_conf),l_list_conf)