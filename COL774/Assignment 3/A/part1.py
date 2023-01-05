import csv
import pandas as pd
import numpy as np
import math
import sys 
from enum import Enum
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import xgboost as xgb
from xgboost import XGBClassifier

# The function loads the X_data and Y_data and removes
# the rows in which one of the label is not available
def get_data(filename):
    data = pd.read_csv(filename, delimiter=',', usecols=["Age" ,"Shape" ,"Margin" ,"Density", "Severity"])
    data = data[(data.Age != '?') & (data.Shape != '?') & (data.Margin != '?') & (data.Density != '?')]
    
    dtype_dict = {'Age': int, 'Shape': int, 'Margin': int, 'Density': int, 'Severity': int}
    data = data.astype(int)
    X_data = data[['Age', 'Shape', 'Margin', 'Density']].to_numpy()
    Y_data = data[['Severity']].to_numpy().T[0]
    return (X_data, Y_data)

def get_data_for_xgboost(filename):
    data = pd.read_csv(filename, delimiter=',', usecols=["Age" ,"Shape" ,"Margin" ,"Density", "Severity"])
    data = data.replace('?', np.nan)
    data = data.astype('float')

    X_data = data[['Age', 'Shape', 'Margin', 'Density']].to_numpy()
    Y_data = data[['Severity']].to_numpy().T[0]
    return (X_data, Y_data)

def get_best_params(clfs):
    param_results = pd.DataFrame.from_dict(clfs.cv_results_)
    best_param = clfs.best_params_
    best_index_ = clfs.best_index_
    row = param_results[best_index_:best_index_+1].to_dict(orient='records')[0]
    return best_param

def get_imputated_data(filename, median = True):
    data = pd.read_csv(filename, delimiter=',', usecols=["Age" ,"Shape" ,"Margin" ,"Density", "Severity"])
    data = data.replace('?', 0)
    
    dtype_dict = {'Age': int, 'Shape': int, 'Margin': int, 'Density': int, 'Severity': int}
    data = data.astype(dtype_dict)
    
    if median:
        for col in ['Age', 'Shape', 'Margin', 'Density']:
            data[col] = data[col].replace(0, int(data[col].median()))
    else:
        for col in ['Age', 'Shape', 'Margin', 'Density']:
            data[col] = data[col].replace(0, int(data[col].mode()))

    X_data = data[['Age', 'Shape', 'Margin', 'Density']].to_numpy()
    Y_data = data[['Severity']].to_numpy().T[0]
    return (X_data, Y_data)

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

	# Code for plotting the decision tree
	fig = plt.figure(figsize=(25,20))
	_ = tree.plot_tree(clf, feature_names=['Age', 'Shape', 'Density', 'Margin'], class_names=['0','1'], filled=True, max_depth=4, fontsize=10)
	plt.savefig("dtree_a.png")

def partb(data, f):
	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	tree_params = {'criterion':["gini", "entropy"], 'max_depth': [1, 2, 4, 5, 6, 7, 8, 9, 10], 'max_features':[1,2,3] ,'min_samples_leaf': [1,2,3,4,5], 'min_samples_split': [2, 3, 4, 5, 6, 7]}
	clfs = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_params,return_train_score = True)
	clfs.fit(X_train, Y_train)

	best_param = get_best_params(clfs)
	
	clf = DecisionTreeClassifier(max_depth = best_param['max_depth'], min_samples_split=best_param['min_samples_split'], 
	                             min_samples_leaf=best_param['min_samples_leaf'], max_features=best_param['max_features'], 
	                             criterion=best_param['criterion'], random_state=0)
	clf.fit(X_train, Y_train)
	train_score = clf.score(X_train, Y_train)
	test_score = clf.score(X_test, Y_test)
	val_score = clf.score(X_val, Y_val)

	f.write("Best params: {}\n".format(best_param))
	f.write("Training accuracy: {}\n".format(train_score*100))
	f.write("Test accuracy: {}\n".format(test_score*100))
	f.write("Validation accuracy: {}\n".format(val_score*100))
	f.close()

	# Code for plotting the decision tree
	fig = plt.figure(figsize=(25,20))
	_ = tree.plot_tree(clf, feature_names=['Age', 'Shape', 'Density', 'Margin'], class_names=['0','1'], filled=True, max_depth=4, fontsize=10)
	plt.savefig("dtree_b.png")

def partc(data, f):

	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	clf = DecisionTreeClassifier(random_state=0)
	path = clf.cost_complexity_pruning_path(X_train, Y_train)
	ccp_alphas, impurities = path.ccp_alphas, path.impurities

	# Higher the ccp_alpha, more is the prunning
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
	ax.plot(ccp_alphas, val_scores, marker="o", label="val", drawstyle="steps-post")
	ax.legend()
	plt.savefig('accuracy.png')

	idx = val_scores.index(max(val_scores))
	best_alpha = ccp_alphas[idx]
	train_acc = train_scores[idx]*100
	test_acc = test_scores[idx]*100
	val_acc = val_scores[idx]*100

	f.write("Best alpha: {}\n".format(best_alpha))
	f.write("Training accuracy: {}\n".format(train_acc))
	f.write("Test accuracy: {}\n".format(test_acc))
	f.write("Validation accuracy: {}\n".format(val_acc))
	f.close()

def partd(data, f):

	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	forest_params = {'criterion':['gini','entropy'],'max_features':[1,2,3,4],'n_estimators':[50,60,70,80,90,100],'min_samples_split':[10,20,30,40,50,60,70,80,90,100]}
	clfs = GridSearchCV(RandomForestClassifier(random_state=0, oob_score=True), forest_params, cv=5)
	clfs.fit(X_train, Y_train.flatten())

	best_param = get_best_params(clfs)
	clf = RandomForestClassifier(n_estimators=best_param['n_estimators'],oob_score=True, min_samples_split=best_param['min_samples_split'], 
								max_features=best_param['max_features'], criterion=best_param['criterion'], random_state=0)
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

def parte(data, f):

	parta(data, f)
	partb(data, f)
	partc(data, f)
	partd(data, f)

def partf(data, f):
	train_data, test_data, val_data = data
	X_train, Y_train = train_data
	X_test, Y_test = test_data
	X_val, Y_val = val_data

	n_estimators = range(10, 60, 10)
	subsample = [0.1, 0.2, 0.3, 0.4, 0.5]
	max_depth = range(4, 11, 1)
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

if __name__ == "__main__":

	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	val_data_path = sys.argv[3]
	output_folder_path = sys.argv[4]
	question_part = sys.argv[5]

	if question_part == 'a':
		train_data = get_data(train_data_path)
		test_data = get_data(test_data_path)
		val_data = get_data(val_data_path)

		f = open(output_folder_path+"1_a.txt", "w+")
		data = (train_data, test_data, val_data)
		parta(data, f)

	elif question_part == 'b':
		train_data = get_data(train_data_path)
		test_data = get_data(test_data_path)
		val_data = get_data(val_data_path)

		f = open(output_folder_path+"1_b.txt", "w+")
		data = (train_data, test_data, val_data)
		partb(data, f)

	elif question_part == 'c':
		train_data = get_data(train_data_path)
		test_data = get_data(test_data_path)
		val_data = get_data(val_data_path)

		f = open(output_folder_path+"1_c.txt", "w+")
		data = (train_data, test_data, val_data)
		partc(data, f)

	elif question_part == 'd':
		train_data = get_data(train_data_path)
		test_data = get_data(test_data_path)
		val_data = get_data(val_data_path)

		f = open(output_folder_path+"1_d.txt", "w+")
		data = (train_data, test_data, val_data)
		partd(data, f)

	elif question_part == 'e':
		# when imputation done by median
		train_data = get_imputated_data(train_data_path, median=True)
		test_data = get_imputated_data(test_data_path, median=True)
		val_data = get_imputated_data(val_data_path, median=True)

		f = open(output_folder_path+"1_e.txt", "w+")
		f.write("Data for Median inputation\n")
		data = (train_data, test_data, val_data)
		parte(data, f)

		# when imputation done by mode
		f.write("Data for Mode inputation\n")
		train_data = get_imputated_data(train_data_path, median=False)
		test_data = get_imputated_data(test_data_path, median=False)
		val_data = get_imputated_data(val_data_path, median=False)

		data = (train_data, test_data, val_data)
		parte(data, f)

	# elif question_part == 'f':
	# 	train_data = get_data_for_xgboost(train_data_path)
	# 	test_data = get_data_for_xgboost(test_data_path)
	# 	val_data = get_data_for_xgboost(val_data_path)

	# 	f = open("1_f.txt")
	# 	data = (train_data, test_data, val_data)
	# 	partf(data, f)