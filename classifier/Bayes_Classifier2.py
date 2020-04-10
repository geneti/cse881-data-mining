import os
import sys
import operator
from math import *
import pickle
import numpy as np
import pandas as pd

'''
P(class|words) --> posterior
P(words)       --> evidence
P(words|class) --> likelihood
P(class)	   --> prior
'''
class bayes_classifier(object):
	def __init__(self, data, eps = 1):
		# require labels format: numpy array
		self.dimension = 20
		self.data = data
		self.total_samples_count = len(data)

		self.likelihood = dict.fromkeys(range(1,self.dimension+1))
		for key in self.likelihood:
			self.likelihood[key] = dict()

		self.prior = dict.fromkeys(range(1,self.dimension+1), 0)
		for s in self.data:
			self.prior[s[1]] += 1

		for i in range(1,self.dimension+1):
			self.prior[i] /= self.total_samples_count

		self.eps = eps
		self.vocab = set()
		self.unknown = dict.fromkeys(range(1,self.dimension+1), 0)

	def train(self):
		total_word = dict.fromkeys(range(1,self.dimension+1), 0)
		# require data format: numpy array
		for iterator, s in enumerate(self.data):
			current_label = s[1]
			sample = np.fromstring(s[0],  dtype = int, sep=',')
			total_word[current_label] += len(sample)
			for word in sample:
				self.vocab.add(word)
				if word not in self.likelihood[current_label].keys():
					self.likelihood[current_label][word] = 1
				else:
					self.likelihood[current_label][word] += 1
		for c in self.likelihood.keys():
			total = total_word[c] + len(self.likelihood[c].keys()) * self.eps
			self.prior[c] /= len(self.data)
			self.unknown[c] = self.eps/total
			for word in self.likelihood[c].keys():
				self.likelihood[c][word] = (self.likelihood[c][word] + self.eps) / total



	def test(self, data):
		pred = np.array([])
		test_labels = np.array([])
		# get the vocabulary list of all training and test data
		for s in data:
			sample = np.fromstring(s[0],  dtype = int, sep=',')
			for word in sample:
				self.vocab.add(word)
		print('total vocab: ', len(self.vocab))

		for iterator, s in enumerate(data):
			sample = np.fromstring(s[0], dtype = int, sep=',')
			current_posterior = np.array([])
			for current_class in range(1, self.dimension):
				likelihood = 0
				for word in sample:
					if word in self.likelihood[current_class]:
						likelihood += log(self.likelihood[current_class][word])
					else:
						likelihood += log(self.unknown[current_class])
				
				prior = log(self.prior[current_class])
				# print('likelihood:', likelihood)
				# print('prior:', prior)
				current_posterior = np.append(current_posterior, likelihood+prior)
			print('current_posterior: ', current_posterior)
			pred = np.append(pred, int(np.argmax(current_posterior)+1))
			#print('pred:',pred)
			test_labels = np.append(test_labels, s[1])

		print(pred)
		print(test_labels)
		ac = 0
		for i in range(len(pred)):
			if pred[i] == test_labels[i]:
				ac+=1
		ac /= len(pred)
		print('accurate is: ', ac)
		
		cov_unbias = np.mean(pred * test_labels) - np.mean(pred) * np.mean(test_labels)
		return cov_unbias

