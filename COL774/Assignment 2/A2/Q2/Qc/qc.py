import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import time 
import sys 

d = 3
C = 1.0
gamma = 0.001
def load_data(train_data, train_labels, test_data, test_labels):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for i in range(len(train_data)):
        label = train_labels[i][0]
        if label == d or label == d+1:
            X_train.append(np.array(train_data[i]).flatten())
            Y_train.append([1 if label == d else -1])
        else:
            continue

    for i in range(len(test_data)):
        label = test_labels[i][0]
        if label == d or label == d+1:
            X_test.append(np.array(test_data[i]).flatten())
            Y_test.append([1 if label == d else -1])
        else:
            continue

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
X_train, Y_train, X_test, Y_test = load_data(train_data, train_labels, test_data, test_labels)


# Using a scikit SVM with linear kernel
start = time.time()
linear_svc = SVC(kernel='linear', C = C)
linear_svc.fit(X_train, Y_train.T[0])
print("time taken by linear kernel: ", time.time()-start)

Y_pred = linear_svc.predict(X_test)
acc = metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy on test data by linear kernel: ", acc)

linear_w = linear_svc.coef_[0]
linear_b = linear_svc.intercept_[0]

# linear_support_indices = np.array(linear_svc.support_)
# np.save('linear_support_indices', linear_support_indices)
# linear_support_vectors = X_train[linear_support_indices]
# np.save('linear_support_vector', linear_support_vectors)
print("Number of support vectors in linear kernel:", np.sum(linear_svc.n_support_))
print()

# Using a scikit SVM with gaussian kernel
start = time.time()
gauss_svc = SVC(kernel = 'rbf', gamma=gamma, C=C)
gauss_svc.fit(X_train, Y_train.T[0])
print("time taken by gaussian kernel: ", time.time()-start)

Y_pred = gauss_svc.predict(X_test)
acc = metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy on test data by gaussian kernel: ", acc)

# gaussian_support_indices = np.array(gauss_svc.support_)
# np.save('gaussian_support_indices', gaussian_support_indices)
# gaussian_support_vectors = gauss_svc.support_vectors_
# np.save('gaussian_support_vectors', gaussian_support_vectors)
print("Number of support vectors in gaussian kernel:", np.sum(gauss_svc.n_support_))