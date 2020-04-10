import sys
import pandas as pd
import numpy as np
import operator
from math import *
import pickle

class tfidf(object):
	def __init__(self, data):
		self.data = data
		self.total_samples = len(data)
		self.label = np.array([])
		self.vocab = set() # all vocabulary
		self.tf = dict.fromkeys(range(self.total_samples), 0) # tf score
		for i in self.tf:
			self.tf[i] = dict()
		self.idf = dict() # idf score
		self.TFIDF = dict.fromkeys(range(self.total_samples), 0) # tf score
		for i in self.TFIDF:
			self.TFIDF[i] = dict()
		for it, s in enumerate(data):
			sample = np.fromstring(s[0],  dtype = int, sep=',')
			self.label = np.append(self.label, s[1])
			for word in sample:
				self.vocab.add(word)
				if word in self.tf[it].keys():
					self.tf[it][word] += 1/len(sample)
				else:
					self.tf[it][word] = 1/len(sample)
			for word in np.unique(sample):
				if word in self.idf.keys():
					self.idf[word] += 1
				else:
					self.idf[word] = 1
		for item in self.idf.values():
			item = log(self.total_samples/item)
		for ss in self.TFIDF.keys():
			for sw in self.tf[ss].keys():
				self.TFIDF[ss][sw] = self.tf[ss][sw] * self.idf[sw]

	def get_tfidf(self, vocab_list = None):
		# get tfidf dataframe
		res = pd.DataFrame.from_dict(self.TFIDF, orient = 'index')
		if vocab_list != None:
			nc = [i for i in vocab_list if i not in res.columns]
			rc = [i for i in res.columns if i not in vocab_list]
			print('nc: ', len(nc))
			res = pd.concat([res, pd.DataFrame(columns = nc)])
			print('rc: ', len(rc))
			res.drop(rc, axis = 1)
		res = res.fillna(0)
		res = res.sort_index(axis=1)
		print('Get a TFIDF matrix with size: ', res.shape)
		return res

	def get_label(self):
		# get corresponding labels
		print('Get a TFIDF label with size: ', len(self.label))
		return pd.DataFrame(self.label)

