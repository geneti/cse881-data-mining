import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Original unstructured data
train_label = pd.read_csv('/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/Training_Label.txt',\
							 sep=',', header=None, engine='python')
train_data = pd.read_csv('/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/Training.txt',\
							 sep=' ', header=None, engine='python')
test_data = pd.read_csv('/Users/liuchang/Desktop/msu_all/homework/data_mining/cse881/project/Test.txt',\
							 sep=' ', header=None, engine='python')
# Get Sample info
N = len(train_data)

