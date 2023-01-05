import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import csv
import pandas as pd
import numpy as np
import math
from enum import Enum
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

def get_best_params(clfs):
    param_results = pd.DataFrame.from_dict(clfs.cv_results_)
    best_param = clfs.best_params_
    best_index_ = clfs.best_index_
    row = param_results[best_index_:best_index_+1].to_dict(orient='records')[0]
    return best_param

def get_vectorizer(data):
	data = data[['condition', 'review', 'date']]
	data['condition'].fillna("", inplace=True)
	data['review'].fillna("", inplace=True)
	data['date'].fillna("", inplace=True)
	stopwords_ = stopwords.words('english')

	corpus = []
	for row in data.itertuples(index = True):
		sentence = getattr(row, 'condition') + getattr(row, 'review') + getattr(row, 'date')
		words = sentence.split(" ")
		new_sentence = ""
		for word in words:
			if word.isalpha():
				new_sentence += word + " "
			else:
				continue
		corpus.append(new_sentence)
	vectorizer = CountVectorizer(stop_words = stopwords_)
	vectorizer.fit_transform(corpus)
	return vectorizer

def preprocess(data, vectorizer):
	data = data[['condition', 'review', 'date']]
	data['condition'].fillna("", inplace=True)
	data['review'].fillna("", inplace=True)
	data['date'].fillna("", inplace=True)
	corpus = []
	for row in data.itertuples(index = True):
		sentence = getattr(row, 'condition') + getattr(row, 'review') + getattr(row, 'date')
		words = sentence.split(" ")
		new_sentence = ""
		for word in words:
			if word.isalpha():
				new_sentence += word + " "
			else:
				continue
		corpus.append(new_sentence)
	X = vectorizer.transform(corpus)
	Y = np.array(data['rating'])

	return (X, Y)

def get_data(train_data_path, test_data_path, val_data_path):

	train_data = pd.read_csv(train_data_path)
	test_data = pd.read_csv(test_data_path)
	val_data = pd.read_csv(val_data_path)

	vectorizer = get_vectorizer(train_data)
	train_data_t = preprocess(train_data, vectorizer)
	test_data_t = preprocess(test_data, vectorizer)
	val_data_t = preprocess(val_data, vectorizer)

	data = (train_data_t, test_data_t, val_data_t)
	return data 

# Normal decision tree construction
def parta(data, f):

	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	clf = DecisionTreeClassifier(random_state=0)
	clf.fit(X_train, Y_train)
	train_score = clf.score(X_train, Y_train)
	test_score = clf.score(X_test, Y_test)
	val_score = clf.score(X_val, Y_val)
	
	f.write("Training accuracy: {}\n".format(train_score*100))
	f.write("Test accuracy: {}\n".format(test_score*100))
	f.write("Validation accuracy: {}\n".format(val_score*100))
	f.close()

# Grid Search in decision tree feature space 
def partb(data, f):
	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	tree_params = {'max_depth': [200, 210, 220, 230, 240], 'min_samples_leaf': [1,2,3,4], 'min_samples_split': [5, 10]}
	clfs = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_params,return_train_score = True)
	clfs.fit(X_train, Y_train)

	best_param = get_best_params(clfs)
	
	clf = DecisionTreeClassifier(max_depth = best_param['max_depth'], min_samples_split=best_param['min_samples_split'], 
	                             min_samples_leaf=best_param['min_samples_leaf'], random_state=0)
	clf.fit(X_train, Y_train)
	train_score = clf.score(X_train, Y_train)
	test_score = clf.score(X_test, Y_test)
	val_score = clf.score(X_val, Y_val)

	f.write("Best params: {}\n".format(best_param))
	f.write("Training accuracy: {}\n".format(train_score*100))
	f.write("Test accuracy: {}\n".format(test_score*100))
	f.write("Validation accuracy: {}\n".format(val_score*100))
	f.close()	

# Cost complexity Prunning the decision tree 
def partc(data, f):

	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	clf = DecisionTreeClassifier(random_state=0)
	path = clf.cost_complexity_pruning_path(X_train, Y_train)
	ccp_alphas, impurities = path.ccp_alphas, path.impurities

	fig, ax = plt.subplots(figsize=(10,10))
	ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
	ax.set_xlabel("effective alpha")
	ax.set_ylabel("total impurity of leaves")
	ax.set_title("Total Impurity vs effective alpha for training set")

	# Training the classifers for different values of ccp_alpha
	# and storing them in the list 
	clfs = []
	for ccp_alpha in ccp_alphas:
	    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
	    clf.fit(X_train, Y_train)
	    clfs.append(clf)

	clfs = clfs[:-1]
	ccp_alphas = ccp_alphas[:-1]

	node_counts = [clf.tree_.node_count for clf in clfs]
	depth = [clf.tree_.max_depth for clf in clfs]
	fig, ax = plt.subplots(2, 1, figsize=(10,10))
	ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
	ax[0].set_xlabel("alpha")
	ax[0].set_ylabel("number of nodes")
	ax[0].set_title("Number of nodes vs alpha")
	ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
	ax[1].set_xlabel("alpha")
	ax[1].set_ylabel("depth of tree")
	ax[1].set_title("Depth vs alpha")
	fig.tight_layout()
	plt.savefig('depth_alpha_vs_node.png')

	# plotting the train_scores, test_scores, val_scores for values of 
	# ccp_alpha
	train_scores = [clf.score(X_train, Y_train) for clf in clfs]
	test_scores = [clf.score(X_test, Y_test) for clf in clfs]
	val_scores = [clf.score(X_val, Y_val) for clf in clfs]
	fig, ax = plt.subplots(figsize=(10,10))
	ax.set_xlabel("alpha")
	ax.set_ylabel("accuracy")
	ax.set_title("Accuracy vs alpha for training, test and vaidation sets")
	ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
	ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
	ax.plot(ccp_alphas, val_scores, marker="o", label="test", drawstyle="steps-post")
	ax.legend()
	plt.savefig('accuracy.png')

	idx = test_scores.index(max(test_scores))
	best_alpha = ccp_alphas[idx]
	train_acc = train_scores[idx]*100
	test_acc = test_scores[idx]*100
	val_acc = val_scores[idx]*100


	f.write("Best alpha: {}\n".format(best_alpha))
	f.write("Training accuracy: {}\n".format(train_acc))
	f.write("Test accuracy: {}\n".format(test_acc))
	f.write("Validation accuracy: {}\n".format(val_acc))
	f.close()


# Grid search with random forests
def partd(data, f):

	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	forest_params = {'max_features':[0.4, 0.5, 0.6, 0.7, 0.8],'n_estimators':[50, 100, 150, 200, 250, 300, 350, 400, 450],'min_samples_split':[2, 4, 6, 8, 10], 'max_depth': [40, 50, 60, 70]}
	clfs = GridSearchCV(RandomForestClassifier(random_state=0, oob_score=True), forest_params, cv=5)
	clfs.fit(X_train, Y_train.flatten())

	best_param = get_best_params(clfs)
	clf = RandomForestClassifier(n_estimators=best_param['n_estimators'],oob_score=True, min_samples_split=best_param['min_samples_split'], 
								max_depth=best_param['max_depth'], random_state=0)
	clf.fit(X_train, Y_train)
	train_score = clf.score(X_train, Y_train)
	test_score = clf.score(X_test, Y_test)
	val_score = clf.score(X_val, Y_val)
	oob_score = clf.oob_score_

	f.write("Best params: {}\n".format(best_param))
	f.write("Training accuracy: {}\n".format(train_score*100))
	f.write("Test accuracy: {}\n".format(test_score*100))
	f.write("Validation accuracy: {}\n".format(val_score*100))
	f.write("Oob score: {}\n".format(oob_score*100))
	f.close()

# Implemented XGBoost algorithm
def parte(data, f):
	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	n_estimators = range(50, 500, 50)
	subsample = [0.4, 0.5, 0.6, 0.7, 0.8]
	max_depth = range(40, 80, 10)
	param_grid = dict(n_estimators=n_estimators, subsample=subsample, random_state=[0], max_depth=max_depth, 
	                  objective=['binary:logistic'])

	model = XGBClassifier()
	clfs = GridSearchCV(model, param_grid,return_train_score = True)
	clfs.fit(X_train, Y_train)

	best_param = get_best_params(clfs)

	clf = XGBClassifier(n_estimators=best_param['n_estimators'], subsample = best_param['subsample'] ,
	                    max_depth = best_param['max_depth'], objective=best_param['objective'], random_state=0)

	clf.fit(X_train, Y_train)
	train_score = clf.score(X_train, Y_train)
	test_score = clf.score(X_test, Y_test)
	val_score = clf.score(X_val, Y_val)
		
	f.write("Best params: {}\n".format(best_param))
	f.write("Training accuracy: {}\n".format(train_score*100))
	f.write("Test accuracy: {}\n".format(test_score*100))
	f.write("Validation accuracy: {}\n".format(val_score*100))
	f.close()

def evaluate(bst, X, Y):
	Y_pred = np.argmax(bst.predict(X), axis=1)

	score = 0
	for i in range(len(Y)):
	  if Y_pred[i] == Y[i]:
	    score += 1
	  else:
	    continue
	return score/len(Y)

# Implemented Gradient boosted machines
def partf(data, f):
	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	train_data_t = lgb.Dataset(X_train, label=Y_train)
	val_data_t = lgb.Dataset(X_val, label=Y_val)
	param = {'num_leaves': 500, 'num_class': 10, 'objective': 'multiclass'}
	bst = lgb.train(param, train_data_t, valid_sets=val_data_t, num_boost_round=100)

	train_score = evaluate(bst, X_train, Y_train)
	test_score = evaluate(bst, X_test, Y_test)
	val_score = evaluate(bst, X_val, Y_val)

	f.write("Training accuracy: {}\n".format(train_score*100))
	f.write("Test accuracy: {}\n".format(test_score*100))
	f.write("Validation accuracy: {}\n".format(val_score*100))
	f.close()

# Training with varying amount of data
def partg(data, f):
	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	num_samples = range(20000, 180000, 20000)
	for n in num_samples:
		index = np.random.choice(X_train.shape[0], n)
		X_train_t = X_train[index]
		Y_train_t = Y_train[index]
		train_data_t = (X_train_t, Y_train_t)
		data_t = (train_data_t, test_data, val_data)
		parta(data_t, f)
		partb(data_t, f)
		partc(data_t, f)
		partd(data_t, f)
		parte(data_t, f)
		partf(data_t, f)


if __name__ == "__main__":

	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	val_data_path = sys.argv[3]
	output_folder_path = sys.argv[4]
	question_part = sys.argv[5]

	if question_part == 'a':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_a.txt", "w+")
		parta(data, f)

	elif question_part == 'b':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_b.txt", "w+")
		partb(data, f)

	elif question_part == 'c':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_c.txt", "w+")
		partc(data, f)

	elif question_part == 'd':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_d.txt", "w+")
		partd(data, f)

	elif question_part == 'e':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_e.txt", "w+")
		parte(data, f)

	elif question_part == 'f':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_f.txt", "w+")
		partf(data, f)

	elif question_part == 'g':
		data = get_data(train_data_path, test_data_path, val_data_path)
		f = open(output_folder_path+"2_g.txt", "w+")
		partg(data, f)