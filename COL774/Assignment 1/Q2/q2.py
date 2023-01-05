from matplotlib import projections, pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import math
import sys

LEARNING_RATE = 0.001

M = 1000000


def sample_data():
    x_0 = np.ones((M, 1))
    x_1 = np.random.normal(3, 2, (M, 1))
    x_2 = np.random.normal(-1, 2, (M, 1))
    epsilons = np.random.normal(0, math.sqrt(2), (M, 1))
    y = 3 * x_0 + 1 * x_1 + 2 * x_2 + epsilons
    x = np.concatenate((x_0, x_1, x_2), axis=1)
    return (x, y)


def gradient_descent(X, Y, theta, m):
    gradient = np.dot(X.T, (Y - np.dot(X, theta)))
    return - gradient / m


def calc_loss(X, Y, theta):
    loss = np.square(Y - np.dot(X, theta)).mean()
    return loss / 2


def sgd(X, Y, b=100):
    X, Y = shuffle(X, Y, random_state=0)
    m, n = np.shape(X)
    theta = np.zeros((n, 1))

    EPSILON = 1e-5
    loss_values = []
    iter = []
    iteration = 0
    num_batches = m // b
    curr_loss = 0
    prev_loss = 100
    epoch = 1
    total_loss = 0

    MOD = 1000

    while True:

        # Iteration for 1 epoch
        for j in range(1, num_batches+1):
            start = (j-1)*b
            end = start + b
            theta = theta - LEARNING_RATE * \
                gradient_descent(X[start: end], Y[start: end], theta, b)
            loss = calc_loss(X[start: end], Y[start: end], theta)
            total_loss += loss
            iteration += 1
            if (iteration % MOD == 0):
                curr_loss = total_loss / MOD
                loss_values.append(curr_loss)
                iter.append(iteration)
                total_loss = 0
                if iteration > 200000:
                    return theta

                if abs(curr_loss - prev_loss) < EPSILON:
                    return theta

            prev_loss = curr_loss
        epoch += 1


def test(X, theta):
    y = np.dot(X, theta)
    np.savetxt("result_2.txt", y, '%.5f')


test_dir = sys.argv[1]
X_test = np.loadtxt(test_dir+"/X.csv", delimiter=',', dtype=float)
m_test = len(X_test)
X_test = np.insert(X_test, 0, np.ones(m_test), axis=1)

X_train, Y_train = sample_data()
theta = sgd(X_train, Y_train, 100)
test(X_test, theta)
