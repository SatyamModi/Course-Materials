from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

C = 1.0
gamma = 0.001
C_vals = [1e-5, 1e-3, 1, 5, 10]

# Load function for scikit svm
def load_data_scikit(train_data, train_labels,  test_data, test_labels):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for i in range(len(train_data)):
        label = train_labels[i][0]
        X_train.append(np.array(train_data[i]).flatten())
        Y_train.append([label])

    for i in range(len(test_data)):
        label = test_labels[i][0]
        X_test.append(np.array(test_data[i]).flatten())
        Y_test.append([label])

    X_train = np.array(X_train)/255
    X_test = np.array(X_test)/255
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    return (X_train, Y_train, X_test, Y_test)

def get_k_fold_aacuracy(X_train, Y_train):
    k = 5
    kf = KFold(n_splits=k, shuffle = True)
    cv_acc = []
    for C in C_vals:
        accuracy = 0
        svm_rbf = SVC(kernel='rbf', gamma=gamma, C=C, decision_function_shape='ovo')
        for train_index, test_index in kf.split(X_train):
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index]

            svm_rbf.fit(x_train, y_train.T[0])
            y_pred = svm_rbf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            accuracy += acc
        cv_acc.append(accuracy/k)
    return cv_acc


def get_test_accuracy(X_train, Y_train, X_test, Y_test):
    test_acc = []
    for C in C_vals:
        svm_rbf = SVC(kernel='rbf', gamma=gamma, C=C, decision_function_shape='ovo')
        svm_rbf.fit(X_train, Y_train.T[0])
        y_pred = svm_rbf.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, y_pred)
        test_acc.append(accuracy)
    return test_acc

train_dir = sys.argv[1]
test_dir = sys.argv[2]
train_data_ = pd.read_pickle(train_dir + "/train_data.pickle")
test_data_ = pd.read_pickle(test_dir + '/test_data.pickle')

train_data = train_data_['data']
train_labels = train_data_['labels']

test_data = test_data_['data']
test_labels = test_data_['labels']

X_train, Y_train, X_test, Y_test = load_data_scikit(train_data, train_labels, test_data, test_labels)
cv_acc = get_k_fold_aacuracy(X_train, Y_train)
test_acc = get_test_accuracy(X_train, Y_train, X_test, Y_test)

log_C = [np.log(c) for c in C_vals]
plt.scatter(log_C, cv_acc, label='cross validation')
plt.scatter(log_C, test_acc, label='test')
plt.plot(log_C, cv_acc)
plt.plot(log_C, test_acc)
plt.title("Cross validation and test accuracy for different C")
plt.ylabel("Accuracy")
plt.xlabel("Log(C)")
plt.legend()
plt.show()