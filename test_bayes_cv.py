import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import json
sys.path.insert(0, '/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/cse881-data-mining/')
from classifier import Bayes_Classifier, Bayes_Classifier2

# Load Original unstructured data
dataset_address = '/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/cse881-data-mining/dataset/'
a = np.array(pd.read_csv(dataset_address + 'Training_Label.txt', sep=',', header=None, engine='python'))
b = np.array(pd.read_csv(dataset_address + 'Training.txt', sep=' ', header=None, engine='python'))

data = np.concatenate((b, a), axis=1)
np.random.shuffle(data)


classifier = Bayes_Classifier2.bayes_classifier(data[1000:len(data)-1])
classifier.train()
result = classifier.test(data[0:1000])
print('unbiased covariance: ', result)

