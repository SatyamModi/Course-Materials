import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import  metrics
import sys


def load_data_scikit(train_data, train_labels, test_data, test_labels):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    m_train = np.shape(train_data)[0]
    m_test = np.shape(test_data)[0]

    for i in range(m_train):
        label = train_labels[i][0]
        X_train.append(np.array(train_data[i]).flatten())
        Y_train.append([label])

    for i in range(m_test):
        label = test_labels[i][0]
        X_test.append(np.array(test_data[i]).flatten())
        Y_test.append([label])

    X_train = np.array(X_train)/255
    X_test = np.array(X_test)/255
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    return (X_train, Y_train, X_test, Y_test)


train_dir = sys.argv[1]
test_dir = sys.argv[2]
train_data_ = pd.read_pickle(train_dir + "/train_data.pickle")
test_data_ = pd.read_pickle(test_dir + '/test_data.pickle')

train_data = train_data_['data']
train_labels = train_data_['labels']

test_data = test_data_['data']
test_labels = test_data_['labels']

X_train, Y_train, X_test, Y_test = load_data_scikit(train_data, train_labels, test_data, test_labels)

svm_rbf = SVC(kernel='rbf', gamma=0.001, C=1, decision_function_shape='ovo').fit(X_train, Y_train.T[0])
svm_predictions = svm_rbf.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, svm_predictions)
print("Accuracy on test data: ", accuracy)
