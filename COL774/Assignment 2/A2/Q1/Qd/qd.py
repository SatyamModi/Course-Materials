import os 
import nltk
import math
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import sys

pos_train_data = []
neg_train_data = []
pos_test_data = []
neg_test_data = []

pos_train_dir = ""
neg_train_dir = ""
pos_test_dir = ""
neg_test_dir = ""

num_train_pos = 0
num_train_neg = 0
num_test_pos = 0
num_test_neg = 0

alpha = 1
pos_stem_word_counter = Counter()
neg_stem_word_counter = Counter()

vocab_stem_size = 0
total_stem_neg_words = 0
total_stem_pos_words = 0

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

def stemmed_predict(sentence):
    words = preprocess_text(sentence)
    # We will be checking P(y = 0 | x) and P(y = 1 | x) which only
    # requires us to calculate P(y) and P(x | y)
    
    # For class 0
    p_y_0 = math.log(num_train_neg / (num_train_pos + num_train_neg))
    p_y_1 = math.log(num_train_pos / (num_train_pos + num_train_neg))
    p_x_y_0 = 0
    p_x_y_1 = 0
    for word in words:
        p_x_y_0 += math.log(neg_stem_word_counter[word] + alpha) - math.log(total_stem_neg_words + alpha*vocab_stem_size)
        p_x_y_1 += math.log(pos_stem_word_counter[word] + alpha) - math.log(total_stem_pos_words + alpha*vocab_stem_size)
    
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

load_data()

num_train_pos = len(pos_train_data)
num_train_neg = len(neg_train_data)
num_test_pos = len(pos_test_data)
num_test_neg = len(neg_test_data)

pos_words_ = list(map(preprocess_text, pos_train_data))
neg_words_ = list(map(preprocess_text, neg_train_data))
pos_words = []
neg_words = []
for words in pos_words_:
    pos_words += words

for words in neg_words_:
    neg_words += words

pos_stem_word_counter = Counter(pos_words)
neg_stem_word_counter = Counter(neg_words)

vocab_stem_size = len(set(pos_stem_word_counter) | set(neg_stem_word_counter))
total_stem_neg_words = len(neg_words)
total_stem_pos_words = len(pos_words)

pos_train_predictions = list(map(stemmed_predict, pos_train_data))
neg_train_predictions = list(map(stemmed_predict, neg_train_data))
total_correct_predictions = pos_train_predictions.count(1) + neg_train_predictions.count(0)
print("Accuracy on training data: ", total_correct_predictions / (num_train_pos + num_train_neg))

pos_test_predictions = list(map(stemmed_predict, pos_test_data))
neg_test_predictions = list(map(stemmed_predict, neg_test_data))
total_correct_predictions = pos_test_predictions.count(1) + neg_test_predictions.count(0)
print("Accuracy on test data: ", total_correct_predictions / (num_test_pos + num_test_neg))


pos_word_cloud_stem = WordCloud(background_color='white', max_words=2000)
pos_word_cloud_stem.generate(" ".join(pos_words))
plt.imshow(pos_word_cloud_stem, interpolation='bilinear')
plt.axis('off')
plt.savefig('pos_word_cloud_stem.png')

neg_word_cloud_stem = WordCloud(background_color='white', max_words=2000)
neg_word_cloud_stem.generate(" ".join(neg_words))
plt.imshow(neg_word_cloud_stem, interpolation='bilinear')
plt.axis('off')
plt.savefig('neg_word_cloud_stem.png')
