import os 
import nltk
import math
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn import metrics

pos_train_data = []
neg_train_data = []
pos_test_data = []
neg_test_data = []

pos_train_dir = "../../part1_data/train/pos/"
neg_train_dir = "../../part1_data/train/neg/"
pos_test_dir = "../../part1_data/test/pos/"
neg_test_dir = "../../part1_data/test/neg/"

num_train_pos = 0
num_train_neg = 0
num_test_pos = 0
num_test_neg = 0

pos_words_ = []
neg_words_ = []

alpha = 1
unigram_plus_trigram_pos_words = []
unigram_plus_trigram_neg_words = []

pos_bigram_unigram_word_counter = Counter()
neg_bigram_unigram_word_counter = Counter()
pos_trigram_unigram_word_counter = Counter()
neg_trigram_unigram_word_counter = Counter()

vocab_plus_bigram_size = 0
vocab_plus_trigram_size = 0
total_unigram_plus_bigram_pos_words = 0
total_unigram_plus_bigram_neg_words = 0
total_unigram_plus_trigram_pos_words = 0
total_unigram_plus_trigram_neg_words = 0

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 

# Returns a list of useful words for a sentence
def preprocess_text(sentence):
    processed_words = []
    tokenized_words = sentence.split(" ")
    for word in tokenized_words:
        if word.isalpha():
            processed_words.append(ps.stem(word.lower()))
        else:
            continue
    words = [word for word in processed_words if word not in stop_words]
    return words

def load_data():
    
    for _, _, files in os.walk(pos_train_dir):
        for filename in files:
            with open(pos_train_dir+filename) as f:
                sentence = f.readlines()[0]
                pos_train_data.append(sentence)
            f.close()
    
    for _, _, files in os.walk(neg_train_dir):
        for filename in files:
            with open(neg_train_dir+filename) as f:
                sentence = f.readlines()[0]
                neg_train_data.append(sentence)
            f.close()
    
    for _, _, files in os.walk(pos_test_dir):
        for filename in files:
            with open(pos_test_dir+filename) as f:
                sentence = f.readlines()[0]
                pos_test_data.append(sentence)
            f.close()       
    
    for _, _, files in os.walk(neg_test_dir):
        for filename in files:
            with open(neg_test_dir+filename) as f:
                sentence = f.readlines()[0]
                neg_test_data.append(sentence)
            f.close()  

def gen_bigrams(sentence):
    bigrams = []
    for i in range(len(sentence)-1):
        bigrams.append(sentence[i] + " " + sentence[i+1])
    return sentence+bigrams

def gen_trigrams(sentence):
    bigrams = []
    trigrams = []
    for i in range(len(sentence)-1):
        bigrams.append(sentence[i] + " " + sentence[i+1])
    for i in range(len(sentence)-2):
        trigrams.append(sentence[i] + " " + sentence[i+1] + " " + sentence[i+2])
    return sentence+bigrams+trigrams

def trigrammed_predict(sentence):
    words = gen_trigrams(preprocess_text(sentence))

    # We will be checking P(y = 0 | x) and P(y = 1 | x) which only
    # requires us to calculate P(y) and P(x | y)
    
    # For class 0
    p_y_0 = math.log(num_train_neg / (num_train_pos + num_train_neg))
    p_y_1 = math.log(num_train_pos / (num_train_pos + num_train_neg))
    p_x_y_0 = 0
    p_x_y_1 = 0
    for word in words:
        p_x_y_0 += math.log(neg_trigram_unigram_word_counter[word] + alpha) - math.log(total_unigram_plus_trigram_neg_words + alpha*vocab_plus_trigram_size)
        p_x_y_1 += math.log(pos_trigram_unigram_word_counter[word] + alpha) - math.log(total_unigram_plus_trigram_pos_words + alpha*vocab_plus_trigram_size)
    
    p_y_0_x = p_x_y_0 + p_y_0
    p_y_1_x = p_x_y_1 + p_y_1
    
    if p_y_0_x > p_y_1_x:
        return 0
    return 1

def bigrammed_predict(sentence):
    words = gen_bigrams(preprocess_text(sentence))
    # We will be checking P(y = 0 | x) and P(y = 1 | x) which only
    # requires us to calculate P(y) and P(x | y)
    
    # For class 0
    p_y_0 = math.log(num_train_neg / (num_train_pos + num_train_neg))
    p_y_1 = math.log(num_train_pos / (num_train_pos + num_train_neg))
    p_x_y_0 = 0
    p_x_y_1 = 0
    for word in words:
        p_x_y_0 += math.log(neg_bigram_unigram_word_counter[word] + alpha) - math.log(total_unigram_plus_bigram_neg_words + alpha*vocab_plus_bigram_size)
        p_x_y_1 += math.log(pos_bigram_unigram_word_counter[word] + alpha) - math.log(total_unigram_plus_bigram_pos_words + alpha*vocab_plus_bigram_size)
    
    p_y_0_x = p_x_y_0 + p_y_0
    p_y_1_x = p_x_y_1 + p_y_1
    
    if p_y_0_x > p_y_1_x:
        return 0
    return 1


load_data()
num_train_pos = len(pos_train_data)
num_train_neg = len(neg_train_data)
num_test_pos = len(pos_test_data)
num_test_neg = len(neg_test_data)

pos_words_ = list(map(preprocess_text, pos_train_data))
neg_words_ = list(map(preprocess_text, neg_train_data))

unigram_plus_bigram_pos_words_ = list(map(gen_bigrams, pos_words_))
unigram_plus_bigram_neg_words_ = list(map(gen_bigrams, neg_words_))

unigram_plus_bigram_pos_words = []
unigram_plus_bigram_neg_words = []
for words in unigram_plus_bigram_pos_words_:
    unigram_plus_bigram_pos_words += words

for words in unigram_plus_bigram_neg_words_:
    unigram_plus_bigram_neg_words += words

pos_bigram_unigram_word_counter = Counter(unigram_plus_bigram_pos_words)
neg_bigram_unigram_word_counter = Counter(unigram_plus_bigram_neg_words)

vocab_plus_bigram_size = len(set(pos_bigram_unigram_word_counter) | set(neg_bigram_unigram_word_counter))
total_unigram_plus_bigram_pos_words = len(unigram_plus_bigram_pos_words)
total_unigram_plus_bigram_neg_words = len(unigram_plus_bigram_neg_words)

pos_bigram_train_predictions = list(map(bigrammed_predict, pos_train_data))
neg_bigram_train_predictions = list(map(bigrammed_predict, neg_train_data))

total_correct_predictions = pos_bigram_train_predictions.count(1) + neg_bigram_train_predictions.count(0)
print("Accuracy on training data: ", total_correct_predictions / (num_train_pos + num_train_neg))

pos_bigram_test_predictions = list(map(bigrammed_predict, pos_test_data))
neg_bigram_test_predictions = list(map(bigrammed_predict, neg_test_data))

total_correct_predictions = pos_bigram_test_predictions.count(1) + neg_bigram_test_predictions.count(0)
print("Accuracy on test data: ", total_correct_predictions / (num_test_pos + num_test_neg))

# To use a trigram as a feature
# unigram_plus_trigram_pos_words_ = list(map(gen_trigrams, pos_words_))
# unigram_plus_trigram_neg_words_ = list(map(gen_trigrams, neg_words_))

# unigram_plus_trigram_pos_words = []
# unigram_plus_trigram_neg_words = []
# for words in unigram_plus_trigram_pos_words_:
#     unigram_plus_trigram_pos_words += words

# for words in unigram_plus_trigram_neg_words_:
#     unigram_plus_trigram_neg_words += words

# pos_trigram_unigram_word_counter = Counter(unigram_plus_trigram_pos_words)
# neg_trigram_unigram_word_counter = Counter(unigram_plus_trigram_neg_words)

# vocab_plus_trigram_size = len(set(pos_trigram_unigram_word_counter) | set(neg_trigram_unigram_word_counter))
# total_unigram_plus_trigram_pos_words = len(unigram_plus_trigram_pos_words)
# total_unigram_plus_trigram_neg_words = len(unigram_plus_trigram_neg_words)

# pos_trigram_train_predictions = list(map(trigrammed_predict, pos_train_data))
# neg_trigram_train_predictions = list(map(trigrammed_predict, neg_train_data))
# total_correct_predictions = pos_trigram_train_predictions.count(1) + neg_trigram_train_predictions.count(0)
# print("Accuracy on training data: ", total_correct_predictions / (num_train_pos + num_train_neg))

# pos_trigram_test_predictions = list(map(trigrammed_predict, pos_test_data))
# neg_trigram_test_predictions = list(map(trigrammed_predict, neg_test_data))
# total_correct_predictions = pos_trigram_test_predictions.count(1) + neg_trigram_test_predictions.count(0)
# print("Accuracy on test data: ", total_correct_predictions / (num_test_pos + num_test_neg))