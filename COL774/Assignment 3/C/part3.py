import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import time
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (np.ones_like(z)) * (z >= 0)

class Nnet:

    def __init__(self, sizes, l_rate, loss_type='MSE', epsilon=2e-5, decay=False, batch_size, activation, activation_prime):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = []
        self.biases = []
        self.l_rate = 0.1
        self.decay = decay
        self.batch_size = batch_size
        self.activation = activation
        self.activation_prime = activation_prime
        self.epsilon=epsilon
        self.loss_type = loss_type
        for i in range(self.num_layers-1):
            self.weights.append(np.random.randn(sizes[i+1], sizes[i])* (2 / sizes[i]) ** 0.5)
            self.biases.append(np.random.randn(sizes[i+1], 1)* (2 / sizes[i]) ** 0.5)

    def forward(self, X):
        X = X.T
        activation = X

        for i in range(len(self.biases)):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, activation) + b

            if i == len(self.biases)-1:
                activation = sigmoid(z)
            else:
                activation = self.activation(z)

        return activation.T

    def calc_loss(self, X, Y, epoch):
        m = len(X)
        predictions = self.forward(X)
        loss = 0
        if self.loss_type == 'MSE':
        	loss = np.sum(np.square(predictions-Y))
    	else:
    		I = np.ones_like(Y)
    		a = (2*Y - I )*predictions
    		b = -np.log(I-Y+a)
			loss = np.sum(b)
        return loss/(2*m)

    def SGD(self, X, Y):
        m = len(X)
        num_batches = m//self.batch_size

        total_loss = 0
        prev_loss = 100
        curr_loss = 0 

        epoch_count = 0
        epoch = 0
        
        while True:
            
            epoch += 1
            epoch_count += 1
            for j in range(1, num_batches+1):
                start = (j-1)*self.batch_size
                end = start + self.batch_size
                X1 = X[start:end]
                Y1 = Y[start:end]
                self.update_mini_batch(X1, Y1, epoch)

            curr_loss = self.calc_loss(X, Y, epoch)
            diff = abs(curr_loss - prev_loss)
            prev_loss = curr_loss

            if (diff < self.epsilon):
                break

    def update_mini_batch(self, X, Y, epoch):
        
        grad_b, grad_w = self.backprop(X, Y)
        for i in range(len(self.weights)):
        	if self.decay:
	            self.weights[i] = self.weights[i] - (self.l_rate*grad_w[i])/math.sqrt(epoch)
	            self.biases[i] = self.biases[i] - (self.l_rate*grad_b[i])/math.sqrt(epoch)
            else:
            	self.weights[i] = self.weights[i] - (self.l_rate*grad_w[i])
	            self.biases[i] = self.biases[i] - (self.l_rate*grad_b[i])

    def backprop(self, X, Y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        X = X.T
        Y = Y.T

        activation = X
        activations = [X]
        zs = []

        for i in range(len(self.biases)):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, activation) + b
            zs.append(z)

            if i == len(self.biases)-1:
                activation = sigmoid(z)
            else:
                activation = self.activation(z)

            activations.append(activation)

        I = np.ones_like(Y) 
        delta = []
        if self.loss_type=='MSE':
        	delta = -(Y-activations[-1]) * sigmoid_prime(zs[-1])
    	else:
    		delta = (1/(I-Y-activations[-1])) * sigmoid_prime(zs[-1])

        delta_b[-1] = np.sum(delta, axis = 1).reshape((-1, 1)) / self.batch_size
        delta_w[-1] = np.dot(delta, activations[-2].T) / self.batch_size

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * self.activation_prime(z)
            delta_b[-l] = np.sum(delta, axis = 1).reshape((-1, 1)) / self.batch_size
            delta_w[-l] = np.dot(delta, activations[-l-1].T) / self.batch_size

        return (delta_b, delta_w)

    def evaluate(self, X, Y, title="", output_path="", plot=False):
        predictions = self.forward(X)
        result = np.argmax(predictions, axis = 1)
        actual_val = np.argmax(Y, axis = 1)

        # Need to plot confusion matrix for test_data
        if plot:
        	cf = confusion_matrix(actual_val, result)
			cf_display = ConfusionMatrixDisplay(confusion_matrix = cf, display_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
			cf_display.plot()
			plt.savefig(output_path + 'cm_{}.png'.format(title))
        	return (sum([result[i] == actual_val[i] for i in range(len(result))]) / len(result)) * 100

    	else:
    		return (sum([result[i] == actual_val[i] for i in range(len(result))]) / len(result)) * 100


# Create a normal neural net
def parta(data, f):
	network = Nnet([784, 5, 10], l_rate=0.1, batch_size=100, activation=sigmoid, activation_prime=sigmoid_prime)

# Varying hidden units in the single hidden layer
def partb(data, f):

	train_data, test_data = data 
	X_train, Y_train = train_data
	X_test, Y_test = test_data

	hidden_layer_units = [5, 10, 15, 20, 25]
	time_taken = []
	train_acc = []
	test_acc = []
	for n_unit in hidden_layer_units:
		network = Nnet([784, n_unit, 10], loss_type='MSE', l_rate=0.1, batch_size=100, activation=sigmoid, activation_prime=sigmoid_prime)
		start = time.time()
		network.SGD(X_train, Y_train)
		end = time.time()

		train_acc_ = network.evaluate(X_train, Y_train)
		test_acc_ = network.evaluate(X_test, Y_test, title=str(n_unit), output_path=output_path, plot=True)
		time_taken.append(end-start)
		train_acc.append(train_acc_)
		test_acc.append(test_acc_)

		f.write('Training accuracy: {}, hidden_units: {}\n'.format(train_acc_, n_unit))
		f.write('Test accuracy: {}, hidden_units: {}\n'.format(test_acc_, n_unit))
	plot_acc(train_acc, test_acc, hidden_layer_units)
	plot_time(time_taken, hidden_layer_units)
	f.close()

# Varying hidden units in the single hidden layer using adaptive learning rate
def partc(data, f):

	train_data, test_data = data 
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	
	hidden_layer_units = [5, 10, 15, 20, 25]
	time_taken = []
	train_acc = []
	test_acc = []
	for n_unit in hidden_layer_units:
		network = Nnet([784, n_unit, 10], loss_type='MSE', l_rate=0.1, decay=True, batch_size=100, activation=sigmoid, activation_prime=sigmoid_prime)
		start = time.time()
		network.SGD(X_train, Y_train)
		end = time.time()

		train_acc_ = network.evaluate(X_train, Y_train)
		test_acc_ = network.evaluate(X_test, Y_test, title=str(n_unit), output_path=output_path, plot=True)
		time_taken.append(end-start)
		train_acc.append(train_acc_)
		test_acc.append(test_acc_)
		f.write('Training accuracy: {}, hidden_units: {}\n'.format(train_acc_, n_unit))
		f.write('Test accuracy: {}, hidden_units: {}\n'.format(test_acc_, n_unit))

	plot_acc(train_acc, test_acc, hidden_layer_units)
	plot_time(time_taken, hidden_layer_units)

	f.close()

# using Relu activation with 100*100 units in the hidden layer
def partd(data, f):

	train_data, test_data = data 
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	
	# For Relu activation
	network = Nnet([784, 100, 100, 10], loss_type='MSE', l_rate=0.1, decay=True, batch_size=100, activation=relu, activation_prime=relu_prime)
	network.SGD(X_train, Y_train)
	train_acc_ = network.evaluate(X_train, Y_train)
	test_acc_ = network.evaluate(X_test, Y_test, title='relu', output_path=output_path, plot=True)
	f.write("Training accuracy on relu: {}\n".format(train_acc_))
	f.write("Test accuracy on relu: {}\n".format(test_acc_))

	# For Sigmoid activation
	network = Nnet([784, 100, 100, 10],loss_type='MSE',  l_rate=0.1, decay=True, batch_size=100, activation=sigmoid, activation_prime=sigmoid_prime)
	network.SGD(X_train, Y_train)
	train_acc_ = network.evaluate(X_train, Y_train)
	test_acc_ = network.evaluate(X_test, Y_test, title='sigmoid', output_path=output_path, plot=True)
	f.write("Training accuracy on sigmoid: {}\n".format(train_acc_))
	f.write("Test accuracy on sigmoid: {}\n".format(test_acc_))
	f.close()

def plot_acc(train_acc, test_acc, n):
	plt.plot(n, test_acc, color='r', marker='.', label='Test accuracy')
	plt.plot(n, train_acc, color='g', marker = '.', label='Training accuracy')
	plt.title('Variation of accuracy with number of hidden layer/units')
	plt.xlabel('No. of hidden layers/units')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('Accuracy.png')

def plot_time(time_taken, n):
	plt.plot(n, time_taken, color='r', marker='.')
	plt.xlabel('No. of units in hidden layer')
	plt.ylabel('Training time(in sec)')
	plt.title('Variation of training time with units in hidden layer')
	plt.savefig('time_taken.png')

# Using multi layered neural network for relu and sigmoid activation
# keeping number of units to be 50 in each layer
def parte(data, f):

	train_data, test_data = data 
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	
	# For relu activation
	relu_train_acc = []
	relu_test_acc = []
	for i in range(2, 6):
		network = Nnet([784]+[50]*i+[10], loss_type='MSE', l_rate=0.1, batch_size=100, activation=relu, activation_prime=relu_prime)
		network.SGD(X_train, Y_train)
		train_acc_ = network.evaluate(X_train, Y_train)
		test_acc_ = network.evaluate(X_test, Y_test)
		relu_train_acc.append(train_acc_)
		relu_test_acc.append(test_acc_)
		f.write("Training accuracy with Relu : {}, hidden_layers: {}\n".format(train_acc_, i))
		f.write("Test accuracy with Relu: {}, hidden_layers: {}\n".format(test_acc_, i))
	
	# plotting code goes here 
	plot_acc(relu_train_acc, relu_test_acc, range(2, 6))

	# For sigmoid activation
	sigmoid_train_acc = []
	sigmoid_test_acc = []
	for i in range(2, 6):
		network = Nnet([784]+[50]*i+[10], loss_type='MSE',  l_rate=0.1, batch_size=100, activation=sigmoid, activation_prime=sigmoid_prime)
		network.SGD(X_train, Y_train)
		train_acc_ = network.evaluate(X_train, Y_train)
		test_acc_ = network.evaluate(X_test, Y_test)
		sigmoid_train_acc.append(train_acc_)
		sigmoid_test_acc.append(test_acc_)
		f.write("Training accuracy with Sigmoid : {}, hidden_layers: {}\n".format(train_acc_, i))
		f.write("Test accuracy with Sigmoid: {}, hidden_layers: {}\n".format(test_acc_, i))

	# plotting code goes here
	plot_acc(sigmoid_train_acc, sigmoid_test_acc, range(2, 6))
	f.close()

# Implementing the neural net with BCE loss
def partf(data, f):
	network = Nnet([784, 50, 50, 10], loss_type='BCE',  l_rate=0.1, epsilon=4e-6, batch_size=100, activation=relu, activation_prime=relu_prime)
	network.SGD(X_train, Y_train)
	train_acc_ = network.evaluate(X_train, Y_train)
	test_acc_ = network.evaluate(X_test, Y_test, title='relu', output_path=output_path, plot=True)
	f.write("Training accuracy on relu: {}\n".format(train_acc_))
	f.write("Test accuracy on relu: {}\n".format(test_acc_))

# Using MLP Classfier from scikit with relu and the hidden layers 
# are chosen from the previous part 
def partg(data, f):
	nnet = MLPClassifier([50, 50], random_state = 0, activation='relu', solver='sgd', verbose = True, batch_size = 100, learning_rate_init=0.1, max_iter=1000)
	nnet.fit(X_train, Y_train)
	train_acc = nnet.score(X_train, Y_train)
	test_acc = nnet.score(X_test, Y_test)
	f.write("Train accuracy: {}\n".format(train_acc))
	f.write("Test accuracy: {}\n".format(test_acc))
	f.close()

def get_data(train_data_path, test_data_path):
	train_data = pd.read_csv(train_data_path, delimiter=',').to_numpy()
	test_data = pd.read_csv(test_data_path, delimiter=',').to_numpy()

    m_train, n_train = np.shape(train_data)
    X_train = train_data[:, :n_train-1]/255

    Y_train = train_data[:, -1]
    n_values = np.max(Y_train) + 1
    Y_train = np.eye(n_values)[Y_train]

    m_test, n_test = np.shape(test_data)
    X_test = test_data[:, :n_test-1]/255
    Y_test = test_data[:, -1]
    n_values = np.max(Y_test) + 1
    Y_test = np.eye(n_values)[Y_test]

    return ((X_train, Y_train), (X_test, Y_test))


if __name__ == "__main__":

	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	output_folder_path = sys.argv[3]
	question_part = sys.argv[4]

	if question_part == 'a':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_a.txt", "w+")
		parta(data, f)

	elif question_part == 'b':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_b.txt", "w+")
		partb(data, f)

	elif question_part == 'c':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_c.txt", "w+")
		partc(data, f)

	elif question_part == 'd':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_d.txt", "w+")
		partd(data, f)

	elif question_part == 'e':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_e.txt", "w+")
		parte(data, f)

	elif question_part == 'f':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_f.txt", "w+")
		partf(data, f)

	elif question_part == 'g':
		data = get_data(train_data_path, test_data_path)
		f = open(output_folder_path+"3_g.txt", "w+")
		partg(data, f)