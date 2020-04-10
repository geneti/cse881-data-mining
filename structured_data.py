import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import json
sys.path.insert(0, '/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/cse881-data-mining/')
from classifier import Bayes_Classifier

# Load Original unstructured data
dataset_address = '/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/cse881-data-mining/dataset/'
train_label = np.array(pd.read_csv(dataset_address + 'Training_Label.txt', sep=',', header=None, engine='python'))
train_data = np.array(pd.read_csv(dataset_address + 'Training.txt', sep=' ', header=None, engine='python'))
test_data = np.array(pd.read_csv(dataset_address + 'Test.txt', sep=' ', header=None, engine='python'))
# Get Sample info

trd = train_data[1000:len(train_data)-1]
trl = train_label[1000:len(train_label)-1]
ted = train_label[0:1000]

N = len(train_data)

classifier = Bayes_Classifier.bayes_classifier(train_label)
classifier.train(train_data)
print('---------Train Complete----------')
print('evidence:', len(classifier.evidence))
print('likelihood:', len(classifier.likelihood[1]))
print('prior:', len(classifier.prior))
# print('total_samples_count:', classifier.total_samples_count)
# print('total_words_count:', classifier.total_words_count)
# print('testdata:', classifier.likelihood[18])
pred = classifier.test(test_data)
print('pred:', pred)
np.savetxt(dataset_address+'predict.txt', pred, fmt='%d')
np.save(dataset_address+'likelihood.npy', classifier.likelihood)
np.save(dataset_address+'prior.npy', classifier.prior)
np.save(dataset_address+'evidence.npy', classifier.evidence)