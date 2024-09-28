# -*- coding: utf-8 -*-
import os
import pprint
import random
from collections import Counter, defaultdict
from itertools import product
import scipy.stats as stats
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from statsmodels.formula.api import ols

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau, pearsonr, spearmanr, beta, norm, pointbiserialr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import researchpy as rp
import seaborn as sns
from collections import defaultdict
from pynini import Weight, shortestdistance
from plotnine import *
# from wynini import config as wfst_config
# from wynini.wfst import *
from pynini import Weight, shortestdistance, Fst, Arc
from learner_wfst import *
import scipy.stats as stats
import math
import functools

pp = pprint.PrettyPrinter(indent=4)
pprint.sorted = lambda x, key=None: x


def memoize(func):
	cache = func.cache = {}
	@functools.wraps(func)
	def memoized_func(*args, **kwargs):
		# Convert any sets in args to frozensets, lists to tuples, and dicts to tuple of tuples
		args = tuple(frozenset(arg) if isinstance(arg, set) 
					 else tuple(arg) if isinstance(arg, list) 
					 else tuple(arg.items()) if isinstance(arg, dict) 
					 else arg for arg in args)
		if args in cache:
			return cache[args]
		else:
			result = func(*args, **kwargs)
			cache[args] = result
			return result
	return memoized_func


class Phonotactics:

	def __init__(self, N=2):
		self.language = "turkish"
		self.N = N
		self.hypothesized_grammar = {}
		self.previous_grammar = {}
		self.updated_training_sample = []
		self.O = {}
		self.E = {}
		self.counter = 0
		self.con = set()
		self.tier = []
		self.phone2ix = {}
		self.parameters = {}
		self.threshold = 0.5
		self.confidence = 0.975
		self.penalty_weight = 3.0
		self.memo = {}
		self.max_length = 0
		self.model = 'filtering'
		self.sample_size = 0
		self.use_cache = True
		self.observed_smooth = 0
		self.filter = True
		# self.alpha = 0.00625 # Danis (2019)
		self.padding = False
		self.structure = "nonlocal"
		self.derived_environment = True

	def process_features(self, file_path):
		alphabet = []
		feature_dict = {}
		file = open(file_path, 'r', encoding='utf-8')
		header = file.readline()
		for line in file:
			line = line.rstrip("\n").split("\t")
			alphabet += [line[0]]
			# line = line.split(',')
			feature_dict[line[0]] = [x for x in line[1:]]
			
			feature_dict[line[0]] += [0, 0]

		num_feats = len(feature_dict[line[0]])

		# feature_dict['<s>'] = [0  for x in range(num_feats-2)] + ['+', '-']
		# feature_dict['<e>'] = [0 for x in range(num_feats-2)] + ['-', '+']

		feat = [feat for feat in header.rstrip("\n").split("\t")]
		feat.pop(0)
		# feat.extend([ '<s>','<e>'])

		feat2ix = {f: ix for (ix, f) in enumerate(feat)}
		ix2feat = {ix: f for (ix, f) in enumerate(feat)}
		

		# feature_table = np.chararray((len(alphabet), num_feats))
		# for i in range(inv_size):
		# 	feature_table[i] = feature_dict[ix2phone[i]]
		return feature_dict, feat2ix

	def get_corpus_data(self, data=None, filename=None):
		# If data is provided, extract "segmentation" from each dictionary in the list
		if data:
			raw_data = [item['Segmentation'] if 'Segmentation' in item else item['segmentation'] for item in data]
		# If a filename is provided, read from the file and split lines by tab to get the third item (i.e. Segmentation)
		elif filename:
			with open(filename, 'r', encoding='utf-8') as file:
				raw_data = [line.rstrip().split('\t')[2] for line in file if line.strip()]
				raw_data = raw_data[1:]  # Skip the header if the data is from a file with a header
		else:
			raise ValueError("Either data or filename must be provided")

		random.shuffle(raw_data)
		return raw_data

	def vectorize_length(self, data):
		'''write the number to the list every time you see a word with certain length'''
		m = 0
		for w in data:
			if len(w) >= m:
				m = len(w)
		l =  [0]*m
		for w in data:
			l[len(w)-1] += 1
		
		# for i in range(m):
			# length of the string is divided by the possible combination in expected sample
			# l[i] = l[i]/2**(i+1)

		# alphabet = list(set(phone for w in raw_data for phone in w))
		# max_chars = max([len(x) for x in raw_data])
		return m, l


	def make_count_table(self, grammar):
		'''
		Print how often each segment in a pair was observed and how often it was expected
		'''
		# Ensure boundary symbols are in the tier list
		tier = self.tier
		if '<s>' in tier:
			tier.remove('<s>')
		if '<e>' in tier:
			tier.remove('<e>')

		if self.language == 'turkish':
			order = ['i', 'e', 'y', 'ø', 'ɯ', 'ɑ', 'u', 'o']
			tier = [x for x in order if x in tier]  # sort tier based on order

		if self.language == 'finnish':
			order = ['y', 'ö', 'ä', 'u', 'o', 'a']
			tier = [x for x in order if x in tier]  # sort tier based on order

		sl2_possible = [tuple(comb) for comb in product(tier, repeat=2) if '<s>' not in comb if '<p>' not in comb if '<e>' not in comb]
		header = ''.join(tier)
		rows = [header]
		pairsdic = {}
		for bigram in sl2_possible:
			if bigram in grammar:
				# if self.bigram_freqs[bigram] != 0:
				# 	print(bigram)
				pairsdic[bigram] = 0 
			else: 
				pairsdic[bigram] = 1

		for seg in tier:
			row = [seg]
			for otherseg in tier:
				pair = (str(seg),str(otherseg))
				row.append(str(pairsdic.get(pair, '')))
			outrow = ''.join(row)
			rows.append(outrow)
		
		# Convert ARPAbet to IPA symbols
		ARPAbet_to_IPA = {
			'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', '<s>': '<s>', '<e>': '<e>', '<p>': '<p>',
		}
		if self.language == 'english':
			rows = [[ARPAbet_to_IPA.get(cell, cell) for cell in row.split('\t')] for row in rows]

		return (rows)


	def match(self, ngrams, grammar):
		# if any(ngram in grammar.keys() for ngram in ngrams):
		# 	print(ngrams, grammar, any(ngram in grammar for ngram in ngrams))
		
		return any(ngram in grammar for ngram in ngrams)


	def ngramize_item(self, string):
		N = self.N
		return [tuple(string[i:i+N]) for i in range(len(string) - (N - 1))] if len(string) >= N else []


	def count_bigram_frequencies(self, data):
		bigram_freqs = defaultdict(int)
		for sequence in data:
			for i in range(len(sequence) - 1):
				bigram = (sequence[i], sequence[i + 1])
				bigram_freqs[bigram] += 1
		return dict(bigram_freqs)


	def penalty(self, wfst, src, arc, grammar):
			symbol = (wfst.state_label(src)[0], wfst.ilabel(arc))
			if symbol in grammar:
				return Weight('log', self.penalty_weight) # Add a penalty term based on the number of constraints in the grammar
			# the more constraints in the grammar, the lower the expected frequency, and the 
			# lesser constraints to be added
				# return Weight('log', 3.0 + 0.1 * len(grammar)) # Add a penalty term based on the number of constraints in the grammar
			else:
				return Weight('log', 0.0)


	def Z(self, wfst, use_cache):
		if use_cache and wfst in self.memo:
			return self.memo[wfst]

		beta = shortestdistance(wfst, reverse=True)
		beta = np.array([float(w) for w in beta])
		result = np.exp(-beta[0])

		if use_cache:
			self.memo[wfst] = result

		return result
	
	
	def observed_dictionary(self):
		observed = {i: 0 for i in self.con}
		for string in self.updated_training_sample:
			unique_ngrams = set(self.ngramize_item(string))
			for ngram in unique_ngrams:
				if ngram in observed:
					observed[ngram] += 1
		return observed


	def expected_dictionary(self):

		use_cache = self.use_cache
		con = self.con
		hypothesized_grammar = list(self.hypothesized_grammar.keys())
		self.tier = [symbol for symbol in self.tier if symbol not in ['<s>', '<e>']]

		if self.padding == True:

			max_length = self.max_length # - 2
			E = {constraint: 0 for constraint in con}
			M_previous = ngram(context='left', length=1, arc_type='log')
			M_previous.assign_weights(hypothesized_grammar, self.penalty)
			M_updated = ngram(context='left', length=1, arc_type='log')

			A = braid(max_length, arc_type='log')
			S_previous = compose(A, M_previous)	# Z_S_previous = Z(S_previous)  # Store the value of Z(S_previous) to avoid recomputing it inside the loop
			Z_S_previous = self.Z(S_previous, use_cache)  # or False to disable caching
		
			# os.system('dot -Tpdf plot/S_previous.dot -o plot/S_previous.pdf')
			for constraint in con:
				if constraint not in hypothesized_grammar:
					hypothesized_grammar.append(constraint)
					M_updated.assign_weights(hypothesized_grammar, self.penalty)
					S_updated = compose(A, M_updated)
					E[constraint] += (1.0 - (self.Z(S_updated, use_cache) / Z_S_previous)) * self.sample_size
					hypothesized_grammar.remove(constraint)
					# max_length = self.max_length - 2
		else:
			# len_vector = self.len_vector[2:]
			len_vector = self.len_vector
			print(len_vector)
			E = {constraint: 0 for constraint in con}
			M_previous = ngram(context='left', length=1, arc_type='log')
			M_previous.assign_weights(hypothesized_grammar, self.penalty)

			M_updated = ngram(context='left', length=1, arc_type='log')
			for n in range(len(len_vector)):
				A = braid(n+1, arc_type='log') # length 10
				S_previous = compose(A, M_previous)	# Z_S_previous = Z(S_previous)  # Store the value of Z(S_previous) to avoid recomputing it inside the loop
				Z_S_previous = self.Z(S_previous, use_cache)  # or False to disable caching
			
				# os.system('dot -Tpdf plot/S_previous.dot -o plot/S_previous.pdf')
				for constraint in con:
					if constraint not in hypothesized_grammar:
						hypothesized_grammar.append(constraint)
						M_updated.assign_weights(hypothesized_grammar, self.penalty)
						S_updated = compose(A, M_updated)
						E[constraint] += (1.0 - (self.Z(S_updated, use_cache) / Z_S_previous)) * len_vector[n]
						hypothesized_grammar.remove(constraint)
		return E


	def upper_confidence_limit_wald(self, O, E):
		confidence = self.confidence

		if E == 0:
			print(f"Upper confidence limit is infinity because E = {E}")
			return float('inf')
		else:
			p = O / E
			z = stats.norm.ppf(confidence)
			denom = 1 + z**2 / E
			center = p + z**2 / (2 * E)
			radius = z * ((p * (1 - p) / E + z**2 / (4 * E**2))**0.5)
			upper_limit = (center + radius) / denom
			if np.isnan(upper_limit) and O <= E:
				print(f"NaN value encountered: O = {O}, E = {E}, confidence = {confidence}, p = {p}, z = {z}, denom = {denom}, center = {center}, radius = {radius}")
			return upper_limit
		
		
	def calculate_OE_and_upper_confidence(self, observed, expected):
		alpha = self.confidence
		n = self.sample_size
		t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)  # t-value for two-tailed test
		O = observed + self.observed_smooth
		E = expected

		# Calculate OE ratio
		OE_ratio = float('inf') if E == 0 else O / E

		# Calculate upper confidence limit
		upper_limit = 1.0
		if O < E and E != 0.0:
			p = OE_ratio
			std_err = (p * (1 - p)) / n
			pi_U = p + math.sqrt(std_err) * t_value
			upper_limit = pi_U if not np.isnan(pi_U) else upper_limit

			if np.isnan(pi_U):
				print(f"NaN value encountered: O = {O}, E = {E}, confidence = {alpha}, p = {p}, t_value = {t_value}, std_err = {std_err}, lower_limit = {pi_U}")

		return OE_ratio, upper_limit
	

	def iterate_and_update(self):
		'''HW algo'''
		# self.tier.remove('<s>')
		# self.tier.remove('<e>')
		# config_wfst = {'sigma': self.tier}
		# config.init(config_wfst)

		# Define a list of O/E ratio thresholds for the stepwise rising accuracy scale
		max_threshold = self.threshold
		thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		filtered_thresholds = [threshold for threshold in thresholds if threshold <= max_threshold]
		for threshold in filtered_thresholds:
			if self.filter == True:
				self.updated_training_sample = [
				string for string in self.updated_training_sample if not self.match(self.ngramize_item(string), self.hypothesized_grammar)]
				self.con = set(self.con) - set(self.hypothesized_grammar)
				print(len(self.updated_training_sample))
			self.max_length, self.len_vector = self.vectorize_length(self.updated_training_sample)
			self.sample_size = len(self.updated_training_sample)
			updated = False
			self.counter += 1
			print(f"\nIteration {self.counter}: ")
			self.O = self.observed_dictionary()
			self.E = self.expected_dictionary()
			# update E here and try new threshold
			for constraint in self.con:
				O = self.O.get(constraint, 0)
				E = self.E.get(constraint, 0)
				OE_ratio, upper_limit = self.calculate_OE_and_upper_confidence(O, E)
				# If the upper confidence limit is less than the threshold, add the constraint		
				if upper_limit <= threshold:
					print("constraint "+ str(constraint) + " OE_ratio "+ str(OE_ratio) + " Upper limit "+str(upper_limit))
					self.hypothesized_grammar[constraint] = (O, E, upper_limit)
					updated = True

		# If no new constraints are being added, break out of the loop
		if not updated:
			return self.hypothesized_grammar
		
		return self.hypothesized_grammar
	
		
	def scan_and_judge_categorical(self, input_filename, out_filename, neg_grammar, pos_grammar):
		tier = self.tier
		if '<s>' not in tier:
			tier.append('<s>')
		if '<e>' not in tier:
			tier.append('<e>')
		inp_file = open(input_filename, 'r', encoding='UTF-8')
		out_file = open(out_filename, 'w', encoding='UTF-8')

		data = []
		as_strings = []

		for line in inp_file:
			line = line.rstrip()
			as_strings.append(line)
			line = line.split()
			data.append([i for i in line if i in self.tier])

		# # Compute the maximum count
		# max_count = 0
		# min_count = float('inf')  # start with infinity so any count will be less
		# all_counts = []

		# for string in data:
		# 	ng = self.ngramize_item(string)
		# 	count = sum(neg_grammar[ngram] for ngram in ng if ngram in neg_grammar)
		# 	all_counts.append(count)
		# 	max_count = max(max_count, count)
		# 	min_count = min(min_count, count)

		# # Now, calculate mu (mean) and sigma (standard deviation)
		# mu = np.mean(all_counts)
		# sigma = np.std(all_counts)

		# # scale_factor = 1 / np.log(max_count + 1)  # to ensure that the probability stays between 0 and 1
		
		# def z_to_prob(z):
		# 	return stats.norm.cdf(z)
		
		# Scale and shift the probability to center it around 1.0
	
		for i, string in enumerate(data):
			curr_string = as_strings[i]
			ngrams = self.ngramize_item(string)

			if all(ngram in pos_grammar for ngram in ngrams):
				# breakpoint()
				prob = 1.0 # very good
			else:
				prob = 0.0
				# print(neg_grammar) #(bigrarm):(O, E, O/E)
				# counts = [neg_grammar[ngram][0] for ngram in ng if ngram in neg_grammar]
				# if any(count == 0 for count in counts):
				# 	prob = 0.0
				# else:
				# 	count = sum(counts)
				# 	z = (count - mu) / sigma if sigma > 0 else 0.0
				# 	prob = z_to_prob(z)

			# scaling techniques 1
			# prob = (count - 0/ max_count - 0) if max_count > 0 else 0.0
			# scaling techniques 2: MinMax Scaling
			# prob = (count - min_count) / (max_count - min_count) if max_count > min_count else 0.0
			# scaling techniques 3: Standard Scaling (Z-score normalization)
				# z = (count - mu) / sigma if sigma > 0 else 0.0
				# prob = z_to_prob(z)
			# scaling techniques 4: 

			# prob = np.exp(scale_factor * count) if max_count > 0 else 0.0
			out_file.write(curr_string.rstrip() + '\t' + str(prob) + "\n")

		inp_file.close()
		out_file.close()


	def scan_and_judge(self, input_filename, out_filename, pos_grammar,neg_grammar):
		tier = self.tier
		if '<s>' not in tier:
			tier.append('<s>')
		if '<e>' not in tier:
			tier.append('<e>')
		inp_file = open(input_filename, 'r', encoding='UTF-8')
		out_file = open(out_filename, 'w', encoding='UTF-8')

		data = []
		as_strings = []

		for line in inp_file:
			line = line.rstrip()
			as_strings.append(line)
			line = line.split()
			data.append([i for i in line if i in self.tier])

		# for i, string in enumerate(data):
		# 	curr_string = as_strings[i]
		# 	ngrams = self.ngramize_item(string)
		# 	probability = 1
		# 	for ngram in ngrams:
		# 	# If the bigram is in the grammar, multiply the probability by its relative frequency
		# 		if ngram in grammar:
		# 			probability *= grammar[ngram]
		# 		# If the bigram is not in the grammar, return 0 because the sequence is not valid
		# 		else:
		# 			probability = 0
		for i, string in enumerate(data):
			curr_string = as_strings[i]
			ngrams = self.ngramize_item(string)
			probability = 1
			for ngram in ngrams:
				# If the bigram is in the grammar, multiply the probability by its relative frequency
				# Then also multiply by the high prior probability
				if ngram in pos_grammar:
					probability *= pos_grammar[ngram] * 0.9
				# If the bigram is not in the grammar, multiply the current probability by the low prior probability
				elif ngram in neg_grammar:
					probability *= neg_grammar[ngram] * 0.1

			out_file.write(curr_string + '\t' + str(probability) + "\n")

		inp_file.close()
		out_file.close()

	def evaluate_kendalltau(self, filepath):
		# Read the data
		data = pd.read_csv(filepath, sep='\t', header=None)
		data = data[[0, 1, data.columns[-1]]]
		data.columns = ['word', 'judgment', 'score']
		data['score'] = data['score'].astype(float)

		
		# Convert 'grammatical' to 1 and 'ungrammatical' to 0
		# data['judgment'] = data['judgment'].map({'grammatical': 1, 'ungrammatical': 0})
		
		# Calculate Kendall's tau
		tau, p_value = kendalltau(data['judgment'], data['score'])
		
		print(f"Kendall's tau: {tau}")
		print(f"P-value: {p_value}")
		
		# Filter and print incorrect predictions
		incorrect_predictions = data[data['judgment'] != round(data['score'])]
		print(f"Incorrect predictions:\n{incorrect_predictions}")

		return tau

	def evaluate_fscore(self, filepath):
		# Read the data
		data = pd.read_csv(filepath, sep='\t', header=None, names=['word', 'judgment', 'score'])

		# Convert 'grammatical' to 1 and 'ungrammatical' to 0
		data['judgment'] = data['judgment'].map({'grammatical': 1, 'ungrammatical': 0})

		# Calculate F1 score

		precision = precision_score(data['judgment'], data['score'])
		recall = recall_score(data['judgment'], data['score'])
		f1 = f1_score(data['judgment'], data['score'])

		print('Precision: ', precision)
		print('Recall: ', recall)
		print('F1 Score: ', f1)

		# Calculate overall accuracy
		data['correct_prediction'] = (data['judgment'] == data['score']).astype(int)
		overall_accuracy = data['correct_prediction'].mean()
		print(f"Overall accuracy: {overall_accuracy}")

		# Calculate accuracy grouped by 'likert_rating_binary'
		grouped_accuracy = data.groupby('judgment')['correct_prediction'].mean()
		print(f"Grouped accuracy:\n{grouped_accuracy}")

		# Filter and print incorrect predictions
		incorrect_predictions = data[data['correct_prediction'] == 0]
		print(f"Incorrect predictions:\n{incorrect_predictions}")

		return f1, overall_accuracy
	

	def main(self,FeatureFile,JudgmentFile,TestingFile,MatrixFile,train_data=None):
		# raw_training_sample = self.get_corpus_data(TrainingFile)
		raw_training_sample = self.get_corpus_data(data=train_data)
		alphabet = list(set(phone for w in raw_training_sample for phone in w))
		boundary_list = ['<e>', '<s>', '<p>']
		# print(alphabet)
		if self.language == "turkish" or self.language == "korean":
			feature_dict, feat2ix = self.process_features(FeatureFile)

			vowel = [x for x in feature_dict if feature_dict[x][feat2ix['syll']] == "+" if feature_dict[x][feat2ix['long']] != "+" if feature_dict[x][feat2ix['abstract']] == "-"] #
			self.tier = vowel

		elif self.language == "chong":
			feature_dict, feat2ix = self.process_features(FeatureFile)

			vowel = [x for x in feature_dict if feature_dict[x][feat2ix['syll']] == "+"  if feature_dict[x][feat2ix['abstract']] == "-"] #
			self.tier = vowel

		elif self.language == "hungarian":
			feature_dict, feat2ix = self.process_features(FeatureFile)

			vowel = [x for x in feature_dict if feature_dict[x][feat2ix['syll']] == "+" if feature_dict[x][feat2ix['abstract']] != "+"] #
			neutral = []
			# neutral = ['i', 'é', 'í']
			neutral = ['i', 'é', 'í', 'e']

			self.tier =[v for v in vowel if v not in neutral] 

		elif self.language == "finnish":
			feature_dict, feat2ix = self.process_features(FeatureFile)

			vowel = [x for x in feature_dict if feature_dict[x][feat2ix['syll']] == "+" if feature_dict[x][feat2ix['abstract']] != "+"] #
			neutral = []
			neutral = ['i', 'e']
			self.tier =[v for v in vowel if v not in neutral] 	
		else:
			# feature_dict, feat2ix = self.process_features(FeatureFile)	
			# breakpoint()
			self.tier = alphabet 
		
		self.tier = [item for item in self.tier if item not in boundary_list]

		print(self.tier)
		# breakpoint()

		self.phoneme_to_idx = {p: ix for (ix, p) in enumerate(self.tier)}

		con = [tuple(comb) for comb in product(self.tier, repeat=2)]
		con = [constraint for constraint in con if constraint not in product(boundary_list, repeat=2)]
		self.con = [constraint for constraint in con if not (constraint[1] == '<e>' and constraint[0] in alphabet or constraint[0] == '<p>' and constraint[1] in alphabet or constraint[0] == '<e>' and constraint[1] in alphabet or constraint[0] in alphabet and constraint[1] == '<s>' or constraint[1] in alphabet and constraint[0] == '<s>' or constraint[0] in alphabet and constraint[1] == '<p>')]
		if self.derived_environment:
			# For derived environment, extract segments on the tier directly adjacent to a hyphen.
			def filter_segments(word):
				# Step 1: Keep only segments on the tier or hyphens
				print(word)
				word = ''.join([seg for seg in word if seg in self.tier or seg == '-'])

				# If there's no hyphen in the word, return an empty list (or you can skip it altogether)
				if '-' not in word:
					return []

				# Step 2: Keep only segments linked by the hyphen
				derived_environment = word.split('-')
				for i, p in enumerate(derived_environment):
					if not p:  # Skip empty strings (not on the tier)
						continue
					
					# Split the first item by each character and keep only the last segment
				first_part = list(derived_environment[0])
				if first_part:
					derived_environment[0] = first_part[-1]

				# Remove any empty derived_environment
				derived_environment = [p for p in derived_environment if p]

				# print(word)
				# print(derived_environment)
				# breakpoint()
				return derived_environment
			# Modify your training sample using the new function
			self.updated_training_sample = [filter_segments(word) for word in raw_training_sample if filter_segments(word)]

		else:
			# Otherwise, consider all segments on the tier.
			self.updated_training_sample = [filtered_string for filtered_string in ([i for i in string if i in self.tier] for string in raw_training_sample) if filtered_string]


		if self.padding == True:
			self.max_length, _ = self.vectorize_length(self.updated_training_sample)

			processed_data = []
			seen = set()  # Set for keeping track of seen lines
			for line in self.updated_training_sample:
				str_line = str(line)  # Convert list to string to make it hashable for set
				if str_line not in seen:
					# seen.add(str_line)
					if len(line) < self.max_length:
						line = ['<s>']  + line + ['<p>'] * (self.max_length - len(line)) + ['<e>']
					else:
						line = ['<s>'] + line + ['<e>']
					processed_data.append(line)
			self.updated_training_sample = processed_data

		# print(self.updated_training_sample)
		# print(self.con)
		# breakpoint()
		
		self.bigram_freqs = self.observed_dictionary()
		pp.pprint(self.bigram_freqs)
		# breakpoint()
		self.tier = [symbol for symbol in self.tier if symbol not in ['<s>', '<e>']]

		config_wfst = {'sigma': self.tier}
		config.init(config_wfst)

		if self.model == 'gross':
			converged_neg_grammar = {c:0 for c in self.bigram_freqs if self.bigram_freqs[c]==0}
		else:
			converged_neg_grammar = self.iterate_and_update()
		# pp.pprint(converged_neg_grammar)

		total_freq = sum(self.bigram_freqs.values())
		penalty_factor = 0.5  # adjust this value as needed

		neg_grammar = {c:(self.bigram_freqs[c]/total_freq)*penalty_factor for c in converged_neg_grammar}
		pos_grammar = {c:self.bigram_freqs[c]/total_freq for c in self.con if c not in converged_neg_grammar}
		converged_grammar = {**pos_grammar, **neg_grammar}
		# self.scan_and_judge(TestingFile, JudgmentFile, pos_grammar,neg_grammar)

		self.scan_and_judge_categorical(TestingFile, JudgmentFile, neg_grammar, pos_grammar)

		# put constraints in a matrix
		table = self.make_count_table(converged_neg_grammar)
		with open(MatrixFile, 'w') as f:
			f.writelines('\t'.join(row) + '\n' for row in table)

		return converged_neg_grammar
	

def hyperparameter_tuning(phonotactics, TrainingFile, FeatureFile, JudgementFile, TestingFile, MatrixFile, humanJudgement, language, train_data):
	# Define ranges for hyperparameters
	thresholds = np.linspace(0.001, 1, num=10)
	confidences = np.linspace(0.975, 0.995, num=5)
	penalties = np.linspace(10, 20, num=5)

	best_objective = -1
	best_params = []

	all_objectives = []
	all_thresholds = []

	# Loop over all combinations of hyperparameters
	for threshold in thresholds:
		phonotactics.threshold = threshold

		phonotactics.main(TrainingFile, FeatureFile, JudgementFile, TestingFile, MatrixFile, train_data)
		# if language == 'turkish' or language == 'korean' or language == 'hungarian':
		fscore, accuracy = phonotactics.evaluate_fscore(JudgementFile)
		all_objectives.append(accuracy)  # Store fscore in the list

		# If the F-score is better than the best so far, update best_params and best_objective
		if accuracy > best_objective:
			best_objective = accuracy
			best_params = [threshold]

		all_thresholds.append(threshold)  # Append threshold for each iteration

	# After the grid search, print the best parameters
	print("Best objectives: ", best_objective)
	print("Best parameters - Threshold: {}".format(*best_params))

	mpl.rc('font',family='Times New Roman')

	# Plot objectives vs thresholds
	fig, ax = plt.subplots()
	ax.plot(all_thresholds, all_objectives, marker='o')  # Plot all objectives and thresholds
	ax.set_xlabel('Threshold', fontname='Times New Roman', fontsize=12)
	ax.set_ylabel('Objective', fontname='Times New Roman', fontsize=12)
	plt.grid(True)

	# Save the figure
	plt.savefig('tuning.png', dpi=400)
