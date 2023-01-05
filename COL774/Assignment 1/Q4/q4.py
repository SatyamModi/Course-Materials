from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
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

    Y_train = np.array([1 if label == "Canada" else 0 for label in np.loadtxt(train_path+"/Y.csv", dtype=str)])
    Y_train = Y_train.reshape((m_train, 1))

    return (X_train, Y_train, X_test)

def normalise(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    variance = np.var(x_train, axis=0)
    x_train = (x_train - mean) / (variance ** 0.5)
    x_test = (x_test - mean) / (variance ** 0.5)
    return (x_train, x_test)

def gda(X, Y):

    m = len(X)
    n = len(X[0])

    # probability of an Alaskan solmon id phi
    phi = np.mean(Y, axis=0)
    mu0 = np.zeros((n, ))
    mu1 = np.zeros((n, ))
    c0 = 0
    c1 = 0
    for i in range(m):
        # Class => Alaska
        if Y[i][0] == 0:
            mu0 += X[i]
            c0 += 1
        # Class => Canada
        else:
            mu1 += X[i]
            c1 += 1

    mu0 /= c0
    mu1 /= c1

    # calculating covariance matrix
    sigma = np.zeros((n, n))
    for i in range(m):
        if Y[i][0] == 0:
            sigma += np.outer(X[i]-mu0, X[i]-mu0)
        else:
            sigma += np.outer(X[i]-mu1, X[i]-mu1)

    sigma /= m
    mu0 = mu0.reshape((n, 1))
    mu1 = mu1.reshape((n, 1))
    return (phi, mu0, mu1, sigma)


def gda_quad(X, Y):

    m = len(X)
    n = len(X[0])

    # probability of an Alaskan solmon is phi, i.e. label 1
    phi = np.mean(Y, axis=0)
    mu0 = np.zeros((n, ))
    mu1 = np.zeros((n, ))
    c0 = 0
    c1 = 0
    for i in range(m):
        # Class => Alaska
        if Y[i][0] == 0:
            mu0 += X[i]
            c0 += 1
        # Class => Canada
        else:
            mu1 += X[i]
            c1 += 1

    mu0 /= c0
    mu1 /= c1

    # calculating covariance matrix
    sigma0 = np.zeros((n, n))
    sigma1 = np.zeros((n, n))
    for i in range(m):
        if Y[i][0] == 0:
            sigma0 += np.outer(X[i]-mu0, X[i]-mu0)
        else:
            sigma1 += np.outer(X[i]-mu1, X[i]-mu1)

    sigma0 /= c0
    sigma1 /= c1
    mu0 = mu0.reshape((n, 1))
    mu1 = mu1.reshape((n, 1))

    return (phi, mu0, mu1, sigma0, sigma1)


def plot(phi, mu0, mu1, sigma, X, Y):

    classes = ["Alaska", "Canada"]
    colours = ListedColormap(['r', 'b'])
    scatter = plt.scatter(X[:, 0], X[:, 1], s=4, c=Y, cmap=colours)
    plt.ylabel("Feature X2")
    plt.xlabel("Feature X1")
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)

    sigma_inv = np.linalg.inv(sigma)
    mat = np.dot(sigma_inv, mu0-mu1)
    a0 = mat[0][0]
    a1 = mat[1][0]
    b0, b1 = np.dot((mu0-mu1).T, sigma_inv)[0]
    c = (np.dot(mu1.T, np.dot(sigma_inv, mu1)) -
         np.dot(mu0.T, np.dot(sigma_inv, mu0)))[0][0]
    c += math.log(phi / (1-phi))

    x = np.linspace(-3, 3, 100)
    y = (-(a0 + b0)*x - c) / (a1 + b1)
    print("Equation of linear boundary is: {}x + {}".format(-(a0+b0)/(a1+b1), -c/(a1+b1)))
    plt.plot(x, y, color='black', label="Decision Boundary")
    plt.show()

def plot_quad(phi, mu0, mu1, sigma, sigma0, sigma1, X, Y):

    classes = ["Alaska", "Canada"]
    colours = ListedColormap(['r', 'b'])
    scatter = plt.scatter(X[:, 0], X[:, 1], s=4, c=Y, cmap=colours)
    plt.ylabel("Feature X2")
    plt.xlabel("Feature X1")
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)

    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    sigma1_inv_mu1 = np.dot(sigma1_inv, mu1)
    sigma0_inv_mu0 = np.dot(sigma0_inv, mu0)
    mu0_T_sigma0_inv_mu0 = np.dot(mu0.T, sigma0_inv_mu0)
    mu1_T_sigma1_inv_mu1 = np.dot(mu1.T, sigma1_inv_mu1)

    # plotting code for the line graph
    sigma_inv = np.linalg.inv(sigma)
    mat = np.dot(sigma_inv, mu0-mu1)
    a0 = mat[0][0]
    a1 = mat[1][0]
    b0, b1 = np.dot((mu0-mu1).T, sigma_inv)[0]
    c = (np.dot(mu1.T, np.dot(sigma_inv, mu1)) - np.dot(mu0.T, np.dot(sigma_inv, mu0)))[0][0]
    c += math.log(phi / (1-phi))
    x_linear = np.linspace(-3, 3, 100)
    y_linear = (-(a0 + b0)*x_linear - c) / (a1 + b1)
    print("Equation of linear boundary is: {}x + {}".format(-(a0+b0)/(a1+b1), -c/(a1+b1)))
    plt.plot(x_linear, y_linear, color='black', label="Decision Boundary")

    # plotting code for quadratic graph
    a00 = (sigma1_inv - sigma0_inv)[0][0]
    a01 = (sigma1_inv - sigma0_inv)[0][1]
    a10 = (sigma1_inv - sigma0_inv)[1][0]
    a11 = (sigma1_inv - sigma0_inv)[1][1]
    b0 = (sigma0_inv_mu0 - sigma1_inv_mu1)[0][0]
    b1 = (sigma0_inv_mu0 - sigma1_inv_mu1)[1][0]
    c = math.log(phi/(1-phi)) + math.log(np.linalg.det(sigma1) / np.linalg.det(sigma0))/2 + (mu1_T_sigma1_inv_mu1 - mu0_T_sigma0_inv_mu0)[0][0]
    x_quad = np.linspace(-3, 4, 100)
    y_quad = np.linspace(-3, 4, 100)
    x_quad, y_quad = np.meshgrid(x_quad, y_quad)
    plt.contour(x_quad, y_quad, (a00*x_quad**2 + (a01+a10)*x_quad*y_quad + a11*y_quad**2 + 2*b0*x_quad + 2*b1*y_quad + c), [0], colors='green')
    plt.show()

def test_linear(X, sigma, phi, mu0, mu1):
    m = len(X)
    y = []

    det_sigma = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    for i in range(m):
        x = X[i]
        p0 = (1-phi)*(1/det_sigma) * (np.exp(-1/2 * np.dot((x-mu0.T), np.dot(sigma_inv, (x-mu0.T).T))))[0][0]
        p1 =  phi* (1/det_sigma) * (np.exp(-1/2 * np.dot((x-mu1.T), np.dot(sigma_inv, (x-mu1.T).T))))[0][0]
        if p0 > p1:
            y.append("Alaska")
        else:
            y.append("Canada")

    y = np.array(y)
    y = np.reshape(y, (m, 1))
    np.savetxt("result_4.txt", y, fmt='%s')

def test_quad(X, sigma0, sigma1, phi, mu0, mu1):
    m = len(X)
    y = []

    det_sigma0 = np.linalg.det(sigma0)
    det_sigma1 = np.linalg.det(sigma1)

    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)

    for i in range(m):
        x = X[i]
        p0 = (1-phi)*(1/det_sigma0) * (np.exp(-1/2 * np.dot((x-mu0.T), np.dot(sigma0_inv, (x-mu0.T).T))))[0][0]
        p1 = (phi)*(1/det_sigma1) * (np.exp(-1/2 * np.dot((x-mu1.T), np.dot(sigma1_inv, (x-mu1.T).T))))[0][0]
        if p0 > p1:
            y.append("Alaska")
        else:
            y.append("Canada")

    y = np.array(y)
    y = np.reshape(y, (m, 1))
    np.savetxt("result_4.txt", np.array(y), fmt='%s')

# Train and Test dir taken from input
train_path = sys.argv[1]
test_path = sys.argv[2]

# Canada is labelled as 1 and Alaska as 0
X_train, Y_train, X_test = read_csv(train_path, test_path)
phi, mu0, mu1, sigma0, sigma1 = gda_quad(X_train, Y_train)

# Testing on test data
test_quad(X_test, sigma0, sigma1, phi, mu0, mu1)
# Output in quadratic case

# Plotting code 
# plot_quad(phi, mu0, mu1, sigma, sigma0, sigma1, X_train, Y_train)