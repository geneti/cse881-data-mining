import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import json
import os.path
from sklearn.decomposition import PCA

from classifier import tfidf

# Load Original unstructured data
dataset_address = '~/dataset/'
PCA_address = 'PCA/'
a = np.array(pd.read_csv(dataset_address + 'Training_Label.txt', sep=',', header=None, engine='python'))
b = np.array(pd.read_csv(dataset_address + 'Training.txt', sep=' ', header=None, engine='python'))

data = np.concatenate((b, a), axis=1)
np.random.shuffle(data)

# Define TFIDF dictionary
if os.path.exists(dataset_address + PCA_address + 'p_train.csv') and os.path.exists(dataset_address + PCA_address+ 'y_train.csv')\
and os.path.exists(dataset_address + PCA_address+ 'x_train_col.csv'):
	print('train_data exists')
	p_train = pd.read_csv(dataset_address + PCA_address+ 'p_train.csv', index_col=0)
	y_train = pd.read_csv(dataset_address + PCA_address+ 'y_train.csv', index_col=0)
	x_train_col = pd.read_csv(dataset_address + PCA_address+ 'x_train_col.csv', index_col=0)
else:
	train_matrix = tfidf.tfidf(data[5000:len(data)-1])
	print('train_data generated')
	x_train = train_matrix.get_tfidf()
	y_train = train_matrix.get_label()

	# apply PCA to transform the sparse matrix
	pca1 = PCA()
	pca1.fit(x_train, y_train)
	p_train = pca1.transform(x_train)
	print('train_data size after PCA: ', p_train.shape)
	x_train_col = pd.DataFrame(list(x_train.columns), columns = [0])
	x_train_col.to_csv(dataset_address+ PCA_address + 'x_train_col.csv')

	pd.DataFrame(p_train).to_csv(dataset_address + PCA_address+ 'p_train.csv')
	y_train.to_csv(dataset_address+ PCA_address + 'y_train.csv')



if os.path.exists(dataset_address+ PCA_address + 'p_test.csv') and os.path.exists(dataset_address+ PCA_address + 'y_test.csv'):
	print('test_data exists')
	p_test = pd.read_csv(dataset_address+ PCA_address + 'p_test.csv', index_col=0)
	y_test = pd.read_csv(dataset_address+ PCA_address + 'y_test.csv', index_col=0)
else:
	test_matrix = tfidf.tfidf(data[0:5000])
	print('test_data generated')
	print(x_train_col)
	x_test = test_matrix.get_tfidf(x_train_col[0].tolist())
	y_test = test_matrix.get_label()

	print('begin PCA 2')
	pca2 = PCA()
	pca2.fit(x_test,y_test)
	p_test = pca2.transform(x_test)
	print('test_data size after PCA: ', p_test.shape)

	pd.DataFrame(p_test).to_csv(dataset_address+ PCA_address + 'p_test.csv')
	y_test.to_csv(dataset_address+ PCA_address + 'y_test.csv')

print('data ready, prepare to fit models')

def accuracy(pred, test_labels):
	ac = 0
	for i in range(len(pred)):
		if pred[i] == test_labels.iloc[i,0]:
			ac+=1
	ac /= len(pred)
	return ac

n_neighbors = 200
# knn classification
knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knn.fit(p_train, y_train)
pred = knn.predict(p_test)
print('pred:',pred)
print('y_test:',y_test)
print('KNN accuracy is: ', accuracy(pred, y_test))

