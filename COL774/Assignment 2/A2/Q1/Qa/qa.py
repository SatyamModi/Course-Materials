import os 
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import sys

pos_train_dir = ""
neg_train_dir = ""
pos_test_dir = ""
neg_test_dir = ""

pos_sentences = []
neg_sentences = []
alpha = 1
positive_words = []
negative_words = []
num_positive = 0
num_negative = 0
positive_word_counter = Counter()
negative_word_counter = Counter()

vocab_size = 0
total_positive_words = 0
total_negative_words = 0

def get_words(pos_train_dir, neg_train_dir):
    positive_words = []
    negative_words = []

    num_positive = 0
    num_negative = 0
    for _, _, files in os.walk(pos_train_dir):
        for filename in files:
            num_positive += 1
            with open(pos_train_dir+filename) as f:
                sentence = []
                words = f.readlines()[0].split(" ")
                for word in words:
                    if word.isalpha():
                        sentence.append(word.lower())
                        positive_words.append(word.lower())
                    else:
                        continue
                pos_sentences.append(sentence)
            f.close()

    for _, _, files in os.walk(neg_train_dir):
        for filename in files:
            num_negative += 1
            with open(neg_train_dir+filename) as f:
                sentence = []
                words = f.readlines()[0].split(" ")
                for word in words:
                    if word.isalpha():
                        sentence.append(word.lower())
                        negative_words.append(word.lower())
                    else:
                        continue
                neg_sentences.append(sentence)
            f.close()
    
    return ((positive_words, num_positive), (negative_words, num_negative))

def predict_on_train(pos_sentences, neg_sentences):
    count_pos = len(pos_sentences)
    count_neg = len(neg_sentences)

    pred_pos_count = 0
    pred_neg_count = 0
    for i in range(count_pos):
        sentence = pos_sentences[i]
        result = predict(sentence)
        if result == 1:
            pred_pos_count += 1
        else:
            continue
    
    for i in range(count_neg):
        sentence = neg_sentences[i]
        result = predict(sentence)
        if result == 0:
            pred_neg_count += 1
        else:
            continue
    
    print("Accuracy on training: ", (pred_neg_count+pred_pos_count)/(count_pos+count_neg))

def predict_on_test(pos_test_dir, neg_test_dir):
    
    actual = []
    predicted = []

    test_positive_correct_count = 0
    test_positive_incorrect_count = 0
    test_negative_correct_count = 0
    test_negative_incorrect_count = 0

    for _, _, files in os.walk(pos_test_dir):
        for filename in files:
            with open(pos_test_dir+filename) as f:
                sentence = []
                tokenized_words = f.readlines()[0].split(" ")
                for word in tokenized_words:
                    if word.isalpha():
                        sentence.append(word.lower())
                    else:
                        continue
                
                result = predict(sentence)
                actual.append(1)
                predicted.append(result)
                if (result == 1):
                    test_positive_correct_count += 1
                else:
                    test_positive_incorrect_count += 1
                f.close()

    for _, _, files in os.walk(neg_test_dir):
        for filename in files:
            with open(neg_test_dir+filename) as f:
                sentence = []
                tokenized_words = f.readlines()[0].split(" ")
                for word in tokenized_words:
                    if word.isalpha():
                        sentence.append(word.lower())
                    else:
                        continue
                        
                actual.append(0)
                result = predict(sentence)
                predicted.append(result)
                if (result == 0):
                    test_negative_correct_count += 1
                else:
                    test_negative_incorrect_count += 1
                f.close()

    correct = test_positive_correct_count+test_negative_correct_count
    incorrect = test_positive_incorrect_count+test_negative_incorrect_count
    print("Accuracy on test: ", correct/(correct+incorrect))

    confusion_matrix_naive = metrics.confusion_matrix(np.array(actual), np.array(predicted))
    cm_display_naive = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_naive, display_labels = [0, 1])

    cm_display_naive.plot()
    plt.savefig('confusion_matrix.png')

def predict(sentence):
    
    # We will be checking P(y = 0 | x) and P(y = 1 | x) which only
    # requires us to calculate P(y) and P(x | y)
    
    # For class 0
    p_y_0 = math.log(num_negative / (num_negative + num_positive))
    p_y_1 = math.log(num_positive / (num_negative + num_positive))
    p_x_y_0 = 0
    p_x_y_1 = 0
    for word in sentence:
        p_x_y_0 += math.log(negative_word_counter[word] + alpha) - math.log(total_negative_words + alpha*vocab_size)
        p_x_y_1 += math.log(positive_word_counter[word] + alpha) - math.log(total_positive_words + alpha*vocab_size)
    
    p_y_0_x = p_x_y_0 + p_y_0
    p_y_1_x = p_x_y_1 + p_y_1
    
    if p_y_0_x > p_y_1_x:
        return 0
    return 1

train_dir = sys.argv[1]
test_dir = sys.argv[2]
pos_train_dir = train_dir + '/pos/'
neg_train_dir = train_dir + '/neg/'
pos_test_dir = test_dir + '/pos/'
neg_test_dir = test_dir + '/neg/'

# num_positive and num_negative is the 
# number of examples of that kind
((positive_words, num_positive), (negative_words, num_negative)) = get_words(pos_train_dir, neg_train_dir)

positive_word_counter = Counter(positive_words)
negative_word_counter = Counter(negative_words)

vocab_size = len(set(positive_word_counter) | set(negative_word_counter))
total_positive_words = len(set(positive_word_counter))
total_negative_words = len(set(negative_word_counter))

predict_on_test(pos_test_dir, neg_test_dir)
predict_on_train(pos_sentences, neg_sentences)


# Code to generate word cloud, to use you must comment
# the rest portion of the code
stopwords = set(STOPWORDS)
positive_word_cloud = WordCloud(background_color='white', max_words=2000, stopwords=stopwords)
positive_word_cloud.generate(" ".join(positive_words))
plt.imshow(positive_word_cloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('pos_word_cloud.png')

negative_word_cloud = WordCloud(background_color='white', max_words=2000, stopwords=stopwords)
negative_word_cloud.generate(" ".join(negative_words))
plt.imshow(negative_word_cloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('neg_word_cloud.png')