# Importing required module
import numpy as np
import pandas as pd
from pre_pros.pre_pro import stop_word_remover, tokenize

# Example text corpus for our tutorial

dataset = pd.read_csv("D:/project/topic_detection/data_compact/final.csv", encoding="utf-8", quoting=1, sep="~")
text = list(dataset["text"])
# text = ['Topic sentences are similar to mini thesis statements.\
#         Like a thesis statement, a topic sentence has a specific \
#         main point. Whereas the thesis is the main point of the essay',
#         'the topic sentence is the main point of the paragraph.\
#         Like the thesis statement, a topic sentence has a unifying function. \
#         But a thesis statement or topic sentence alone doesn’t guarantee unity.',
#         'An essay is unified if all the paragraphs relate to the thesis,\
#         whereas a paragraph is unified if all the sentences relate to the topic sentence.']

# Preprocessing the text data
sentences = []
word_set = []

for sent in text:
    x = stop_word_remover(tokenize(sent), is_split=True, return_split=True)
    sentences.append(x)
    for word in x:
        if word not in word_set:
            word_set.append(word)

# Set of vocab
word_set = set(word_set)
# Total documents in our corpus
total_documents = len(sentences)

# Creating an index for each word in our vocab.
index_dict = {}  # Dictionary to store index for each word
i = 0
for word in word_set:
    index_dict[word] = i
    i += 1


# Create a count dictionary

def count_dict(sentences):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count


word_count = count_dict(sentences)

#Term Frequency
def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N


# Inverse Document Frequency

def inverse_doc_freq(word):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_documents / word_occurance)


def tf_idf(sentence):
    tf_idf_vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = termfreq(sentence, word)
        idf = inverse_doc_freq(word)

        value = tf * idf
        tf_idf_vec[index_dict[word]] = value
    return tf_idf_vec


# TF-IDF Encoded text corpus
vectors = []
for sent in sentences:
    vec = tf_idf(sent)
    vectors.append(vec)

print(vectors[0])