import os
import sys
import operator
import math
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
	def __init__(self, labels, eps = 0.01, sigma = 0.00001):
		# require labels format: numpy array
		self.dimension = labels[len(labels)-1][0]
		self.labels = labels
		self.likelihood = dict.fromkeys(range(1,self.dimension+1))
		for key in self.likelihood:
			self.likelihood[key] = dict()
		self.prior = dict.fromkeys(range(1,self.dimension+1), 0)
		for s in self.labels:
			self.prior[s[0]] += 1
		self.evidence = dict()
		self.eps = eps
		self.sigma = sigma
		self.vocab = set()
		self.total_samples_count = len(labels)
		self.total_words_count = 0
		##self.count = dict.fromkeys(range(1,self.dimension+1), 0)

	def train(self, data):
		# require data format: numpy array
		for iterator, s in enumerate(data):
			current_label = self.labels[iterator][0]
			##self.count[current_label] += len(sample)
			sample = np.fromstring(s[0],  dtype = int, sep=',')
			#print('sample:', sample)
			self.total_words_count += len(sample)
			for word in sample:
				self.vocab.add(word)
				if word not in self.evidence.keys():
					self.evidence[word] = 1
				else:
					self.evidence[word] += 1
				if word not in self.likelihood[current_label].keys():
					self.likelihood[current_label][word] = 1
				else:
					self.likelihood[current_label][word] += 1

		# Normalize evidence, likelihood and prior
		for i in range(1,self.dimension+1):
			self.prior[i] /= self.total_samples_count

		for word in self.evidence.keys():
			self.evidence[word] /= self.total_words_count

		for label_iterator in range(1,self.dimension+1):
			class_sum = sum(self.likelihood[label_iterator].values()) + len(self.likelihood[label_iterator]) * self.eps
			for key, dividend in self.likelihood[label_iterator].items():
				self.likelihood[label_iterator][key] = (dividend + self.eps)/class_sum

	def test(self, data):
		pred = np.array([])
		for iterator, s in enumerate(data):
			sample = np.fromstring(s[0], dtype = int, sep=',')
			evidence = 1

			average_evidence = sum(self.evidence.values())/self.total_words_count
			for word in sample:
				if word not in self.evidence:
					evidence *= average_evidence
				else:
					evidence *= self.evidence[word]
			current_posterior = np.array([])
			for current_class in range(1,self.dimension+1):
				likelihood = 1

				average_likelihood = sum(self.likelihood[current_class].values())/len(self.likelihood[current_class])
				for word in sample:
					if word not in self.likelihood[current_class]:
						likelihood *= average_likelihood
					else:
						likelihood *= self.likelihood[current_class][word]
				prior = self.prior[current_class]
				if likelihood==0 or evidence==0:
					current_posterior = np.append(current_posterior, 0)
				else:
					current_posterior = np.append(current_posterior, likelihood*prior/evidence)
			#print('max:', current_posterior)
			pred = np.append(pred, int(np.argmax(current_posterior)+1))
		return pred

