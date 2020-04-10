import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import json
import os.path
from sklearn import linear_model, neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, '/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/cse881-data-mining/')
from classifier import tfidf

# Load Original unstructured data
dataset_address = '/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/cse881-data-mining/dataset/'
LDA_address = 'LDA/'
a = np.array(pd.read_csv(dataset_address + 'Training_Label.txt', sep=',', header=None, engine='python'))
b = np.array(pd.read_csv(dataset_address + 'Training.txt', sep=' ', header=None, engine='python'))

data = np.concatenate((b, a), axis=1)
np.random.shuffle(data)

# Define TFIDF dictionary
if os.path.exists(dataset_address + LDA_address + 'p_train.csv') and os.path.exists(dataset_address + LDA_address + 'y_train.csv')\
and os.path.exists(dataset_address + LDA_address + 'x_train_col.csv'):
	print('train_data exists')
	p_train = pd.read_csv(dataset_address + LDA_address + 'p_train.csv', index_col=0)
	y_train = pd.read_csv(dataset_address + LDA_address + 'y_train.csv', index_col=0)
	x_train_col = pd.read_csv(dataset_address + LDA_address + 'x_train_col.csv', index_col=0)
else:
	train_matrix = tfidf.tfidf(data[5000:len(data)-1])
	print('train_data generated')
	x_train = train_matrix.get_tfidf()
	y_train = train_matrix.get_label()

	# apply LinearDiscriminantAnalysis to transform the sparse matrix
	lda1 = LinearDiscriminantAnalysis()
	lda1.fit(x_train, y_train)
	p_train = lda1.transform(x_train)
	print('train_data size after LinearDiscriminantAnalysis: ', p_train.shape)
	x_train_col = pd.DataFrame(list(x_train.columns), columns = [0])
	x_train_col.to_csv(dataset_address + LDA_address + 'x_train_col.csv')

	pd.DataFrame(p_train).to_csv(dataset_address + 'p_train.csv')
	y_train.to_csv(dataset_address + LDA_address + 'y_train.csv')



if os.path.exists(dataset_address + LDA_address + 'p_test.csv') and os.path.exists(dataset_address + LDA_address + 'y_test.csv'):
	print('test_data exists')
	p_test = pd.read_csv(dataset_address + LDA_address + 'p_test.csv', index_col=0)
	y_test = pd.read_csv(dataset_address + LDA_address + 'y_test.csv', index_col=0)
else:
	test_matrix = tfidf.tfidf(data[0:5000])
	print('test_data generated')
	print(x_train_col)
	x_test = test_matrix.get_tfidf(x_train_col[0].tolist())
	y_test = test_matrix.get_label()

	print('begin lda 2')
	lda2 = LinearDiscriminantAnalysis()
	lda2.fit(x_test, y_test)
	p_test = lda2.transform(x_test)
	print('test_data size after lda: ', p_test.shape)

	pd.DataFrame(p_test).to_csv(dataset_address + LDA_address + 'p_test.csv')
	y_test.to_csv(dataset_address + LDA_address + 'y_test.csv')

print('data ready, prepare to fit models')

def accuracy(pred, test_labels):
	ac = 0
	for i in range(len(pred)):
		if pred[i] == test_labels.iloc[i,0]:
			ac+=1
	ac /= len(pred)
	return ac

n_neighbors = 5000
# knn classification
knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knn.fit(p_train, y_train)
pred = knn.predict(p_test)
print('pred:',pred)
print('y_test:',y_test)
print('KNN accuracy is: ', accuracy(pred, y_test))

