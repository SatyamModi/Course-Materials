from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import sys

def read_csv(train_path, test_path):
    X_train = np.loadtxt(train_path+"/X.csv", delimiter=',', dtype=float)
    X_test = np.loadtxt(test_path+"/X.csv",delimiter=',', dtype=float)
    X_train, X_test = normalise(X_train, X_test)

    m_train = len(X_train)
    m_test = len(X_test)
    n = len(X_train[0])
    X_train = X_train.reshape((m_train, n))
    X_test = X_test.reshape((m_test, n))

    X_train = np.insert(X_train, 0, np.ones(m_train), axis=1)
    X_test = np.insert(X_test, 0, np.ones(m_test), axis=1)

    Y_train = np.loadtxt(train_path+"/Y.csv", dtype=float)
    Y_train = Y_train.reshape((m_train, 1))

    return (X_train, Y_train, X_test)

def normalise(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    variance = np.var(x_train, axis=0)
    x_train = (x_train - mean) / (variance ** 0.5)
    x_test = (x_test - mean) / (variance ** 0.5)
    return (x_train, x_test)


def hessian(X, theta):
    m = len(X)
    n = len(X[0])
    H = np.zeros((n, n))

    for j in range(n):
        for k in range(n):
            for i in range(m):
                h_theta = (1 / (1 + np.exp(-np.dot(theta.T, X[i]))))[0]
                H[j][k] += -(X[i][j]*X[i][k])*h_theta*(1-h_theta)

    return H


def likelihood_gradient(X, Y, theta):
    h_theta_X = (1 / (1 + np.exp(-np.dot(X, theta))))
    gradient = np.dot(X.T, Y - h_theta_X)
    return gradient

def newtons(X, Y):
    n = len(X[0])
    theta = np.zeros((n, 1))
    prev_grad_norm = 100
    curr_grad_norm = 0
    EPSILON = 1e-10
    while (abs(curr_grad_norm - prev_grad_norm) > EPSILON):
        H_inverse = np.linalg.inv(hessian(X, theta))
        gradient = likelihood_gradient(X, Y, theta)
        theta = theta - np.dot(H_inverse, gradient)
        prev_grad_norm = curr_grad_norm
        curr_grad_norm = np.sum(np.square(gradient))

    return theta


def plot(X, Y, theta):
    values = [1 if l == 1 else 0 for l in Y]
    classes = ["Class 0", "Class 1"]
    colours = ListedColormap(['b', 'r'])
    scatter = plt.scatter(X[:, 1], X[:, 0], s=5, c=values, cmap=colours)
    plt.ylabel("X1")
    plt.xlabel("X2")
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)

    x = np.linspace(1, 10, 100)
    y = -(theta[2]*x + theta[0]) / theta[1]
    plt.plot(x, y, color='black', label="Decision Boundary")
    plt.show()

def test(X, theta):
    y = [1 if pred[0] > 0.5 else 0 for pred in (1 / (1 + np.exp(-np.dot(X, theta))))]
    np.savetxt("result_3.txt", y, '%.0f')

train_dir = sys.argv[1]
test_dir = sys.argv[2]

X_train, Y_train, X_test = read_csv(train_dir, test_dir)
theta = newtons(X_train, Y_train)
test(X_test, theta)
# plot(X, Y, theta)
