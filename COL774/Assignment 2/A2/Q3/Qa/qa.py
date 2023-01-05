import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time 
import sys
from PIL import Image

gamma = 0.001
C = 1.0
def load_data(train_data, train_labels, test_data, test_labels):
    X_train = {}
    Y_train = {}
    
    X_test = []
    Y_test = []
    
    possible_labels = np.unique(train_labels)
    for label in possible_labels:
        X_train[label] = []
        Y_train[label] = []
        
    for i in range(len(train_data)):
        label = train_labels[i][0]
        X_train[label].append(np.array(train_data[i]).flatten())
        Y_train[label].append(label)
    
    for i in range(len(test_data)):
        label = test_labels[i][0]
        X_test.append(np.array(test_data[i]).flatten())
        Y_test.append([label])
    
    for label in possible_labels:
        X_train[label] = np.array(X_train[label], dtype = float)/255
        Y_train[label] = np.array(Y_train[label], dtype = int)
        
    X_test = np.array(X_test, dtype = float)/255
    Y_test = np.array(Y_test, dtype = int)
    
    return (X_train, Y_train, X_test, Y_test)

def get_gauss_scores(Y_train, alpha, K, b_gauss):
    wT_x = np.sum(K*((alpha*Y_train).T), axis = 1)
    pred_score = wT_x + b_gauss
    return pred_score

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

# Calculation of the gaussian kernel matrix

def get_kernel_matrix(x0, x1):
    return np.exp(-gamma * cdist(x0, x1) ** 2)

def get_alpha(P, q, G, h, A, b):
    sol = solvers.qp(P, q, G, h, A, b, options={'show_progress':False})
    return np.array(sol['x'])

def get_gaussian_b(Y, wT_x):
    maxterm = -float('inf')
    minterm = float('inf')
    m = np.shape(wT_x)[0]
    for i in range(m):
        if Y[i][0] == -1:
            maxterm = max(maxterm, wT_x[i])
        else:
            minterm = min(minterm, wT_x[i])
    b_gauss = -(minterm + maxterm) / 2
    return b_gauss

def classify(X_train, Y_train):
    
    m_train = np.shape(X_train)[0]
    K_train = get_kernel_matrix(X_train, X_train)
    P = get_P_matrix(X_train, Y_train, K_train, True)
    q = get_Q_matrix(m_train)
    G = get_G_matrix(m_train)
    h = get_H_matrix(m_train)
    A = get_A_matrix(Y_train)
    b = get_B_matrix()
    
    alpha = get_alpha(P, q, G, h, A, b)
    wT_x = np.sum((alpha*Y_train)*K_train, axis = 0)
    b_gauss = get_gaussian_b(Y_train, wT_x)
    return (alpha, b_gauss)

def get_alpha_and_b(X_train_, Y_train_):
    alphas = []
    b_s = []
    for i in range(5):
        for j in range(i+1, 5):
            class1 = i
            class2 = j
            X_train = []
            Y_train = []
            X_train = np.vstack((X_train_[class1], X_train_[class2]))
            Y_train = np.vstack((Y_train_[class1], Y_train_[class2]))

            # class1 is labelled as 1, class2 is labelled as -1
            Y_train[Y_train == class2] = -1
            Y_train[Y_train == class1] = 1
            Y_train = Y_train.reshape((-1, 1))
            alpha, b = classify(X_train, Y_train)
            alphas.append(alpha)
            b_s.append(b)
    return (np.array(alphas), np.array(b_s))

def get_score_and_votes(X_train_, Y_train_, alphas, b_s, X_test):
    m_test = np.shape(X_test)[0]
    scores = np.zeros((5, m_test))
    votes = np.zeros((5, m_test))
    idx = 0
    for i in range(5):
        for j in range(i+1, 5):
            class1 = i
            class2 = j
            X_train = []
            Y_train = []
            X_train = np.vstack((X_train_[class1], X_train_[class2]))
            Y_train = np.vstack((Y_train_[class1], Y_train_[class2]))

            # class1 is labelled as 1, class2 is labelled as -1
            Y_train[Y_train == class2] = -1
            Y_train[Y_train == class1] = 1
            Y_train = Y_train.reshape((-1, 1))
            alpha = alphas[idx]
            b = b_s[idx]
            K_test = get_kernel_matrix(X_test, X_train)
            score = get_gauss_scores(Y_train, alpha, K_test, b)

            score_i = np.array(score)
            score_i[score_i < 0] = 0
            score_i = np.abs(score_i)

            score_j = np.array(score)
            score_j[score_j >= 0] = 0
            score_j = np.abs(score_j)

            pred_i = np.array(score)
            pred_i[pred_i >= 0] = 1
            pred_i[pred_i < 0] = 0

            pred_j = np.array(score)
            pred_j[pred_j >= 0] = 0
            pred_j[pred_j < 0] = 1

            votes[i] += pred_i
            votes[j] += pred_j
            scores[i] += score_i
            scores[j] += score_j
            
            idx += 1
    return (scores, votes)

# Returns the prediction by considering the score and votes
def get_predictions(scores, votes):

    # scores and votes is of shape 5 * m_test
    m_test = np.shape(scores)[1]
    score_tuples = []
    dtype = [('vote', int), ('score', float), ('label', int)]
    for i in range(5):
        tmp = []
        for j in range(m_test):
            tmp.append((votes[i][j], scores[i][j], i))
        tmp = np.array(list(tuple(map(tuple, tmp))), dtype = dtype)
        score_tuples.append(tmp)
    score_tuples = np.array(score_tuples)
    pred = []
    for i in range(m_test):
        col = np.sort(score_tuples[:, i], order = ['vote'], axis = 0)
        pred.append(col[-1][2])
    return pred

def get_accuracy(Y_actual, Y_pred):
    m = np.shape(Y_actual)[0]
    count = 0
    for i in range(m):
        if Y_actual[i][0] == Y_pred[i]:
            count += 1
        else:
            continue
    return count / m

train_dir = sys.argv[1]
test_dir = sys.argv[2]

train_data_ = pd.read_pickle(train_dir + "/train_data.pickle")
test_data_ = pd.read_pickle(test_dir + '/test_data.pickle')

train_data = train_data_['data']
train_labels = train_data_['labels']

test_data = test_data_['data']
test_labels = test_data_['labels']

X_train_, Y_train_, X_test, Y_test = load_data(train_data, train_labels, test_data, test_labels)
start = time.time()
alphas, b_s = get_alpha_and_b(X_train_, Y_train_)
print("Time taken: ", time.time()-start)

scores, votes = get_score_and_votes(X_train_, Y_train_, alphas, b_s, X_test)
Y_pred = get_predictions(scores, votes)

accuracy = get_accuracy(Y_test, Y_pred)*100
print("Accuracy on test data: ", accuracy)