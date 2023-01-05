from matplotlib import  pyplot as plt
import numpy as np
import sys 

LEARNING_RATE = 0.001
EPSILON = 1e-10

def read_csv(train_path, test_path):
    X_train = np.loadtxt(train_path+"/X.csv", dtype=float)
    X_test = np.loadtxt(test_path+"/X.csv", dtype=float)
    X_train, X_test = normalise(X_train, X_test)

    m_train = len(X_train)
    m_test = len(X_test)

    X_train = X_train.reshape((m_train, 1))
    X_test = X_test.reshape((m_test, 1))

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


def gradient_descent(X, Y, theta, m):
    gradient = np.dot(X.T, (Y - np.dot(X, theta)))
    return - gradient / m

def calc_loss(X, Y, theta):
    loss = np.square(Y - np.dot(X, theta)).mean()
    return loss / 2

def fmt(x):
    return "Cost = {}".format(f"{x:.2f}")

def batch_gradient_descent_with_plot(X, Y):

    m, n = np.shape(X)
    theta = np.zeros((n, 1))
    a = np.linspace(-0.5, 1.5, 200)
    b = np.linspace(-1, 1, 200)

    x, y = np.meshgrid(a, b)
    z = np.zeros(np.shape(x))

    for i in range(len(x)):
        for j in range(len(x[0])):
            z[i][j] = calc_loss(X, Y, np.array([[x[i][j]], [y[i][j]]]))

    # axes for the MeshGrid plot of Loss function
    ax1 = plt.axes(projection='3d')
    ax1.plot_wireframe(x, y, z, color='green', linewidth=0.5)
    ax1.set_xlabel('Theta[0]')
    ax1.set_ylabel('Theta[1]')
    ax1.set_zlabel('Cost')

    # axes for the Contour plot of the Cost function
    fig, ax2 = plt.subplots(1, 1)
    cs = ax2.contour(y, x, z)
    ax2.set_xlabel('theta[1]')
    ax2.set_ylabel('theta[0]')
    ax2.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    prev_loss = 100
    curr_loss = calc_loss(X, Y, theta)
    i = 0
    while (abs(curr_loss - prev_loss) > EPSILON):
        prev_loss = curr_loss
        theta = theta - LEARNING_RATE * gradient_descent(X, Y, theta, m)
        curr_loss = calc_loss(X, Y, theta)
        if i % 20 == 0:
            ax1.scatter3D(theta[0], theta[1], curr_loss)
            ax2.scatter(theta[1], theta[0])
            plt.pause(0.2)
        i += 1
    plt.show()
    return theta

# For testing by TAs
def batch_gradient_descent(X, Y):

    m, n = np.shape(X)
    theta = np.zeros((n, 1))
    prev_loss = 100
    curr_loss = calc_loss(X, Y, theta)
    i = 0
    while (abs(curr_loss - prev_loss) > EPSILON):
        prev_loss = curr_loss
        theta = theta - LEARNING_RATE * gradient_descent(X, Y, theta, m)
        curr_loss = calc_loss(X, Y, theta)
    return theta

# Plotting code for the actual and predicted y's. 
def plot_predictions(X, Y, theta):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 1], Y[:, 0], s=8)
    ax.set_xlabel("Acidity of wine")
    ax.set_ylabel("Density of wine")
    x = np.linspace(-3, 5, 100)
    y = theta[0] + theta[1]*x
    plt.plot(x, y, color='green')
    plt.show()

def test(X, theta):
    y = np.dot(X, theta)
    np.savetxt("result_1.txt", y, '%.5f')

# Train and Test dir taken from input
train_path = sys.argv[1]
test_path = sys.argv[2]
X_train, Y_train, X_test = read_csv(train_path, test_path)
# theta = batch_gradient_descent(X_train, Y_train)
# test(X_test, theta)
theta = batch_gradient_descent_with_plot(X_train, Y_train)