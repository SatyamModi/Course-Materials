import numpy as np
import pandas as pd
from PIL import Image
import time 
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
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

def get_kernel_matrix(x0, x1):
    return np.exp(-gamma * cdist(x0, x1) ** 2)

def get_P_matrix(X, Y, K = np.array([]), kernel = False):
    if (kernel == False):
        XY = X*Y
        p = np.dot(XY, XY.T)*1.0
        P = matrix(p, tc = 'd')
        return P
    else:
        p = Y*K*Y.T
        P = matrix(p, tc = 'd')
        return P

def get_Q_matrix(m):
    Q = -np.ones((m, 1))
    return matrix(Q , tc = 'd')

def get_G_matrix(m):
    g1 = np.identity(m)
    g2 = -g1
    G = np.vstack((g1, g2))
    return matrix(G, tc = 'd')

def get_H_matrix(m):
    h1 = C*np.ones((m, 1))
    h2 = 0 * h1
    H = np.vstack((h1,h2))
    return matrix(H , tc = 'd')

def get_A_matrix(Y):
    m = np.shape(Y)[0]
    A = Y.reshape(1,m)*1.
    return matrix(A , tc = 'd')

def get_B_matrix():
    return matrix([0.0], tc = 'd')
    
def get_alpha(P, q, G, h, A, b):
    sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    return np.array(sol['x'])

def get_gaussian_b(Y, wT_x):
    maxterm = -float('inf')
    minterm = float('inf')
    for i in range(m_train):
        if Y[i][0] == -1:
            maxterm = max(maxterm, wT_x[i])
        else:
            minterm = min(minterm, wT_x[i])
    b_gauss = -(minterm + maxterm) / 2
    return b_gauss

# Here, K is the kernel matrix considering 
# train data and test data
def predict_gauss_test(Y_train, alpha, K, b_gauss):
    wT_x = np.sum(K*((alpha*Y_train).T), axis = 1)
    Y_pred = wT_x + b_gauss
    Y_pred[Y_pred >= 0] = 1
    Y_pred[Y_pred < 0] = -1
    return Y_pred 

def accuracy(Y_actual, Y_pred):
    count = 0
    m = np.shape(Y_actual)[0]
    for i in range(m):
        if Y_actual[i][0] == Y_pred[i]:
            count += 1
        else:
            continue
    return count / m

train_dir = sys.argv[1]
test_dir = sys.argv[2]
train_data_ = pd.read_pickle(train_dir+'/train_data.pickle')
test_data_ = pd.read_pickle(test_dir+'/test_data.pickle')

train_data = train_data_['data']
train_labels = train_data_['labels']

test_data = test_data_['data']
test_labels = test_data_['labels']
X_train, Y_train, X_test, Y_test = load_data(train_data, train_labels, test_data, test_labels)
m_train = np.shape(X_train)[0]

K_train = get_kernel_matrix(X_train, X_train)

P = get_P_matrix(X_train, Y_train, K_train, True)
q = get_Q_matrix(m_train)
G = get_G_matrix(m_train)
h = get_H_matrix(m_train)
A = get_A_matrix(Y_train)
b = get_B_matrix()

start = time.time()
alpha = get_alpha(P, q, G, h, A, b)
print("Time taken : ", time.time()-start)

count = 0
for i in range(m_train):
    if alpha[i][0] > 1e-4:
        count = count + 1
    else:
        continue

print("Number of support vectors: ", count)

wT_x = np.sum((alpha*Y_train)*K_train, axis = 0)
b_gauss = get_gaussian_b(Y_train, wT_x)

K_test = get_kernel_matrix(X_test, X_train)
Y_pred = predict_gauss_test(Y_train, alpha, K_test, b_gauss)
acc = accuracy(Y_test, Y_pred)
print("Accuracy on test data: ", acc*100)

# Top 5 images 
alpha_ = alpha.reshape((1, -1))[0]
req_index = np.argsort(-alpha_)
support_index = np.array(req_index[:count])

# np.save('support_index', support_index)
req_image = []
for i in range(1, 6):
    req_image.append(train_data[req_index[-i]])

for i in range(0, 5):
    img = Image.fromarray(req_image[i], 'RGB')
    img.save('alpha_g_{}.png'.format(i+1))
