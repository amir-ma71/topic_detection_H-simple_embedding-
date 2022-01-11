import pandas as pd
from pre_pros.pre_pro import stop_word_remover, tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_coding_vocab(corpos):
    words = ["new word"]
    counter = 1
    for row in corpos:
        try:
            row_words = stop_word_remover(tokenize(row), is_split=True, return_split=True)
            for word in row_words:
                if word not in words:
                    words.append(word)
        except:
            continue
        print(counter)
        counter += 1
    with open('./pre_pros/src/words_coding_vocab.pickle', 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("words codding list created....")


def build_count_vocab(corpos):
    vocab = {}
    counter = 1
    for row in corpos:
        try:
            row_words = stop_word_remover(tokenize(row), is_split=True, return_split=True)
            for word in row_words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        except:
            continue
        print(counter)
        counter += 1

    with open('./pre_pros/src/words_count_vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("words codding list created....")



def coding_vec(corpos, max_length, label_list):
    with open('./pre_pros/src/words_coding_vocab.pickle', 'rb') as handle:
        words = pickle.load(handle)
    text_vectors = []
    label_vectors = []
    text = list(corpos["text"])
    label = list(corpos["label"])
    for row in range(len(text)):
        try:
            row_words = stop_word_remover(tokenize(text[row]), is_split=True, return_split=True)
            label_vec = label_list.index(label[row])
            one_hot = np.zeros(len(label_list))
            one_hot[label_vec] = 1
        except:
            continue
        row_vector = []
        for word in row_words:
            try:
                row_vector.append(words.index(word))
            except:
                row_vector.append(0)

        if len(row_vector) >= max_length:
            row_vector = row_vector[0:max_length]
        else:
            for i in range(max_length - len(row_vector)):
                row_vector.append(-1)

        text_vectors.append(row_vector)
        label_vectors.append([one_hot])

    return text_vectors, label_vectors


def count_vec(corpos, max_length, label_list):
    with open('./pre_pros/src/words_count_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    text_vectors = []
    label_vectors = []
    text = list(corpos["text"])
    label = list(corpos["label"])
    for row in range(len(text)):
        try:
            row_words = stop_word_remover(tokenize(text[row]), is_split=True, return_split=True)
            label_vec = label_list.index(label[row])
            one_hot = np.zeros(len(label_list))
            one_hot[label_vec] = 1
        except:
            continue
        row_vector = []
        for word in row_words:
            try:
                row_vector.append(vocab[word])
            except:
                row_vector.append(0)

        if len(row_vector) >= max_length:
            row_vector = row_vector[0:max_length]
        else:
            for i in range(max_length - len(row_vector)):
                row_vector.append(0)

        text_vectors.append(row_vector)
        label_vectors.append(one_hot)

    return text_vectors, label_vectors


def tf_idf(corpos, label_list):
    tfidf = TfidfVectorizer(encoding='utf-8',
                            tokenizer=tokenize,
                            max_features= 250
                            )
    text_vectors = tfidf.fit_transform(corpos["text"]).toarray()
    label = list(corpos["label"])
    label_vectors = []
    text_vector = []

    for l in range(len(label)):
        if label[l] in label_list:
            label_vec = label_list.index(label[l])
            one_hot = np.zeros(len(label_list))
            one_hot[label_vec] = 1
            label_vectors.append(one_hot)
            text_vector.append(text_vectors[l])
        else:
            continue


    return text_vector, label_vectors

