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
	def __init__(self, labels, eps = 0.01):
		# require labels format: numpy array
		self.dimension = labels[len(labels)-1]
		self.labels = labels
		self.likelihood = dict.fromkeys(range(self.dimension))
		for key in likelihood:
			self.likelihood[key] = dict()
		self.prior = dict.fromkeys(range(self.dimension), 0)
		for s in labels:
			self.prior[labels[s]] += 1
		self.evidence = dict()
		self.eps = eps
		self.vocab = set()
		self.total_samples_count = len(labels)
		self.total_words_count = 0
		##self.count = dict.fromkeys(range(self.dimension), 0)

	def train(self, data):
		# require data format: numpy array
		for iterator, sample in enumerate(data):
			current_label = self.labels[iterator]
			##self.count[current_label] += len(sample)
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
		self.prior = self.prior/self.total_samples_count
		self.evidence = self.evidence/self.total_words_count
		for label_iterator in range(self.dimension):
			class_sum = sum(evidence[label_iterator].values()) + len(evidence[label_iterator]) * self.eps
			for key, dividend in self.evidence[label_iterator].items():
				self.evidence[label_iterator][key] = (dividend + self.eps)/class_sum

	def test(self, data):
		pred = np.array()
		for iterator, sample in enumerate(data):
			evidence = 1
			for word in sample:
				evidence *= self.evidence[word]
			current_posterior = np.array()
			for current_class in range(self.dimension):
				likelihood = 1
				for word in sample:
					likelihood *= self.likelihood[current_class][word]
				prior = self.prior[current_class]
				current_posterior.append(likelihood*prior/evidence)
			pred.append(max(current_posterior))
		return pred

