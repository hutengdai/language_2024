import unicodedata
import csv
from phonotactics import *
import copy  # You'll need the copy module
import random

def hyperparameter_alternation(phonotactics_hypothesis,threshold_values,biases,split_ratios,language, run_experiment):
	
	for bias in biases:
		for threshold in threshold_values:
			accuracies = [run_experiment(phonotactics_hypothesis, threshold, bias, split_ratio, language) for split_ratio in split_ratios]
			results = pd.DataFrame({
				'Training Data Size': split_ratios,
				'Accuracy': accuracies,
				'Bias': [bias] * len(split_ratios),
			})

			output_filename = f"result/{language}/{language}_{threshold}_{bias}.txt"
			with open(output_filename, 'w') as file:
				results.to_csv(file, sep="\t", index=False)

def read_ur_file(filename):
	try:
		# Attempt to read as dictionary using eval
		with open(filename, 'r') as file:
			content = file.read()
			# Using eval to safely convert string representation of dictionary into an actual dictionary
			# Note: Be cautious while using eval with untrusted data.
			return eval('{' + content + '}')
	except:
		# If the above failed, try reading the file in the second format
		output_dict = {}
		with open(filename, 'r', encoding='utf-8') as file:
			lines = file.readlines()
			for line in lines:
				key, value = line.strip().split(':')
				output_dict[key.strip()] = value.strip()
		return output_dict

def phonotactic_learner(train_data,FeatureFile, TestingFile, split_ratio, language, structure,filter,padding,confidence,penalty_weight,threshold,model):
	phonotactics = Phonotactics()
	phonotactics.language = language
	phonotactics.structure = structure
	phonotactics.filter = filter
	phonotactics.padding = padding
	phonotactics.confidence = confidence
	phonotactics.penalty_weight = penalty_weight # 10
	phonotactics.threshold = threshold # 0.5  for best accuracy
	phonotactics.model = model #filtering
	phonotactics.derived_environment = False # whether or not specified the search domain to derived environment

	JudgementFile = ("result/%s/judgment/judgment_%s_%s_flt-%s_pad-%s_conf-%s_pen-%s_split-%s_thr-%s.txt" % 
		(
		phonotactics.language,
		phonotactics.model,
		phonotactics.structure, 
		'T' if phonotactics.filter else 'F', 
		'T' if phonotactics.padding else 'F', 
		str(phonotactics.confidence), 
		str(phonotactics.penalty_weight),
		str(split_ratio),
		str(phonotactics.threshold)
		)
	)
	MatrixFile = ("result/%s/matrix/matrix_%s_%s_flt-%s_pad-%s_conf-%s_pen-%s_split-%s_thr-%s.txt" % 
		(
		phonotactics.language,
		phonotactics.model,
		phonotactics.structure, 
		'T' if phonotactics.filter else 'F', 
		'T' if phonotactics.padding else 'F', 
		str(phonotactics.confidence), 
		str(phonotactics.penalty_weight),
		str(split_ratio),
		str(phonotactics.threshold)
		)
	)
	learned_phonotactics = phonotactics.main(FeatureFile, JudgementFile, TestingFile, MatrixFile,train_data)
	# if language != 'turkish':
	f1, overall_accuracy = phonotactics.evaluate_fscore(JudgementFile)
	print("F1 "+str(f1))
	print("overall accuracy "+str(overall_accuracy))
	# else:
	# 	tau = phonotactics.evaluate_kendalltau(JudgementFile)
	# 	print("tau "+str(tau))

	tier = [v for v in phonotactics.tier if v not in ["<s>", "<e>"]]
	return set(learned_phonotactics.keys()), tier

class DataSplitter:
	def __init__(self, train_file, split=0.8, test_file=None):
		self.train_data = self._read_data(train_file)
		
		if test_file:
			self.train_data, self.dev_data = self._split_data(self.train_data, split)

			self.test_data = self._read_data(test_file)
		else:
			self.train_data, self.test_data = self._split_data(self.train_data, split)

	def _read_data(self, path):
		with open(path, 'r', encoding="utf-8") as file:
			reader = csv.DictReader(file, delimiter="\t")
			data = []
			for row in reader:
				entry = {
					'sf': row['SF'],
					'freq': row['Freq'],
					'analysis': row['Analysis'],
					'segmentation': row['Segmentation']
				}
				data.append(entry)
			return data

	def _split_data(self, data, split, seed=42):
		random.seed(seed)  # set the seed for reproducibility

		# Calculate total frequency to normalize the probabilities
		total_freq = sum(int(x['freq']) for x in data)
		probabilities = [int(x['freq']) / total_freq for x in data]

		# Sample without replacement for the train data
		train_size = int(split * len(data))
		train_indices = random.choices(range(len(data)), weights=probabilities, k=train_size)
		train_data = [data[i] for i in train_indices]

		# Remove duplicates from train_indices to ensure dev_data does not contain duplicates
		unique_train_indices = set(train_indices)

		# The rest of the data will be for development
		dev_data = [item for index, item in enumerate(data) if index not in unique_train_indices]

		return train_data, dev_data
	
	# def _split_data(self, data, split, seed=42):
	# 	# for reproducing learned phonotactic grammars
	# 	train_data = data
	# 	dev_data = []
	# 	return train_data, dev_data

	def create_train_dev_datasets(self):
		# Method 2: Create a list of all words and corresponding weights (frequencies)
		weights = [int(item['freq']) for item in self.word_data_list]

		# Determine the split index
		split_idx = int(0.8 * len(self.word_data_list))

		# Obtain the training data randomly, weighted by the token frequency
		train_indices = np.random.choice(len(self.word_data_list), size=split_idx, replace=False, p=np.array(weights)/sum(weights))
		train_data = [self.word_data_list[i] for i in train_indices]

		# For the dev data, you can take the remaining items that aren't in train_data
		dev_data = [item for i, item in enumerate(self.word_data_list) if i not in train_indices]

		return train_data, dev_data
	
class RuleLearner(Phonotactics):

	def __init__(self, UR_data, train_data, FeatureFile, tier, phonotactics, naturalness = True):
		super().__init__()
		self.ur = UR_data
		self.train_data = train_data
		self.tier = tier
		self.phonotactics = phonotactics

		self.feature_dict, self.feat2ix = self.process_features(FeatureFile)
		self.abstract_segments = [x for x in self.feature_dict if self.feature_dict[x][self.feat2ix['abstract']] == "+"] #
		self.feature_encoding = self.feature_reshape()
		self.redundant_features = self.get_redundant_features()
		self.entailment_pairs = self.generate_entailment_pairs()
		self.bias = "maximal"
		self.parental_stem_lexicon = {}
		self.parental_affixes_lexicon = {}
		self.exceptions = []
		self.naturalness = naturalness

	def entail_logic(self, precedent, consequence):
		for feature, p_val in precedent.items():
			if feature == 'abstract':  # Ignore the 'abstract' feature
				continue
			
			c_val = consequence.get(feature)

			# If the value in precedent is not '0' and doesn't match the value in consequence, return False
			if p_val != '0' and p_val != c_val:
				return False

		return True

	def generate_entailment_pairs(self):
		abstract_segments = {k: v for k, v in self.feature_encoding.items() if v.get('abstract') == '+'}
		
		entailment_relations = {a_key: [] for a_key in abstract_segments}
		
		for a_key, a_features in abstract_segments.items():
			for other_key, other_features in self.feature_encoding.items():
				if self.entail_logic(a_features, other_features):
					entailment_relations[a_key].append(other_key)

		return entailment_relations

	def feature_reshape(self):
		result = {}
		for segment, features in self.feature_dict.items():
			segment_features = {}
			for index, value in enumerate(features):
				# We only want to map features that have an associated index
				if index in self.feat2ix.values():
					feature_name = [f for f, i in self.feat2ix.items() if i == index][0]
					segment_features[feature_name] = value
			result[segment] = segment_features
		return result

	def get_redundant_features(self):
		"""Returns features shared by every segment in the tier."""
		all_segments = self.tier
		universally_shared = {}
		
		for feature, ix in self.feat2ix.items():
			if all(self.feature_encoding[seg].get(feature) == self.feature_encoding[all_segments[0]].get(feature) for seg in all_segments):
				universally_shared[feature] = self.feature_encoding[all_segments[0]].get(feature)

		return universally_shared

	def get_env_shared_features(self, segments, redundant_features):
		"""Returns the features shared by the provided segments excluding universally shared features."""
		segment_specific_shared_features = {}
		for feature in self.feature_encoding[segments[0]]:
			if feature not in redundant_features:
				if all(self.feature_encoding[segment].get(feature) == self.feature_encoding[segments[0]].get(feature) for segment in segments):
					segment_specific_shared_features[feature] = self.feature_encoding[segments[0]].get(feature)
		return segment_specific_shared_features
	
	def get_changed_features(self, ur_segment, sr_segment):
		ur_features = self.feature_encoding.get(ur_segment, {})
		sr_features = self.feature_encoding.get(sr_segment, {})
		
		changed_features = {}
		for feature, ur_value in ur_features.items():
			if feature == "abstract":  # Skip the abstract feature
				continue
			sr_value = sr_features.get(feature)
			if sr_value is not None and ur_value != sr_value:
				changed_features[feature] = sr_value
				
		return changed_features

	def extract_stem_environment(self, stem):
		# Return the last segment from self.tier in the stem by searching from the right to left. If no tier segments, return empty string
		return next((char for char in reversed(stem) if char in self.tier), "nontier")

	def extract_data(self,word_data, key):
		"""Extract specific data from the given key in word_data."""
		segments = word_data[key].split("-")
		if len(segments) > 1:  # If there's at least one hyphen.
			return segments[1:]
		return []

	def generate_phonological_rules(self):
		tier = self.tier + self.abstract_segments
		segment_rules = []
		for word_data in self.train_data:
			
			stem_sf = word_data['segmentation'].split("-")[0]
			stem_environment = self.extract_stem_environment(stem_sf)

			# Skip processing if there's no hyphen in the analysis.
			if "-" not in word_data['analysis']:
				continue

			left_env = stem_environment
			meanings = self.extract_data(word_data, 'analysis')
			surface_forms = word_data['segmentation'].split("-")[1:]
			underlying_forms = [self.ur["-" + affix] for affix in meanings]
			derived_environment = stem_environment + ''.join(char for char in ''.join(surface_forms) if char in tier)
			# if "ű" in surface_forms or 'ő' in surface_forms:
			# 	print(f"Found 'ű' or 'ő' in raw data: {word_data['segmentation']}")
			# 	breakpoint()
			# breakpoint()

			# Skip any nontier stem data.
			if stem_environment not in tier:
				# You cannot learn default neutral vowel
				continue

			# Check against phonotactic constraints.
			ngrams = self.ngramize_item(derived_environment)
			if self.phonotactics and self.match(ngrams, self.phonotactics):
				self.exceptions.append({'segmentation': word_data['segmentation'], 'analysis': word_data['analysis']})
				continue

			normalized_ur = unicodedata.normalize('NFC', ''.join(underlying_forms))
			normalized_sr = unicodedata.normalize('NFC', ''.join(surface_forms))

			# if "ű" in normalized_sr or "ő" in normalized_sr:
			# 	print(f"Found 'ű' or 'ő' in SR: {normalized_sr}")
			# 	breakpoint()
			ur_tier = ''.join([char for char in normalized_ur if char in tier])
			sr_tier = ''.join([char for char in normalized_sr if char in tier])
			# After the filtering process:
			# if "ű" in ur_tier or "ő" in ur_tier:
			# 	print(f"Found 'ű' or 'ő' in ur_tier: {ur_tier}")
			# if "ű" in sr_tier or "ő" in sr_tier:
			# 	print(f"Found 'ű' or 'ő' in sr_tier: {sr_tier}")

			# Report if SR is empty.
			if not sr_tier:
				# print(f"'{word_data['analysis']}':'{word_data['segmentation']}' not on tier")
				continue

			# Ensure UR and SR have equal lengths.
			if len(ur_tier) != len(sr_tier):
				print(f"Ignore UR '{ur_tier}' length mismatch with SR '{sr_tier}'")
				continue

			# Add differing UR-SR pairs to segment rules.
			for ur, sr in zip(ur_tier, sr_tier):
				# Within the loop where UR and SR pairs are compared:
				# if ur == "ű" or sr == "ű" or ur == "ő" or sr == "ő":
				# 	print(f"Processing segment rule with 'ű' or 'ő': UR={ur}, SR={sr}, Env={left_env}")
				# if ur == "a" and sr == "o":
				# 	print(word_data)
				# 	breakpoint()
				if ur != sr and (ur, sr, left_env) not in segment_rules:
					segment_rules.append((ur, sr, left_env))
				left_env = sr
		
		# Combine and update segment-based rules.
		merged_rules = {}
		for ur, sr, env in segment_rules:
			key = (ur, sr)
			merged_rules[key] = list(set(merged_rules.get(key, []) + list(env)))
		# print("segment-based mapping")
		# print(merged_rules)
		
		
		# breakpoint()
		return merged_rules

	def remove_conflicting_rules(self, seg_rules):
		seg_rules_copy = copy.deepcopy(seg_rules)  # Create a deep copy of seg_rules
		
		# Identify conflicting environments
		conflicts = {}
		for (ur, sr), env_list in seg_rules_copy.items():
			for env in env_list:
				if (ur, env) in conflicts and conflicts[(ur, env)] != sr:
					# If SR is different for the same (UR, env) combination
					conflicts[(ur, env)] = None  # mark as conflict
				else:
					conflicts[(ur, env)] = sr

		# Remove the conflicting environments
		for (ur, env), sr in conflicts.items():
			if sr is None:  # If it's a conflicting environment
				for rule_key in seg_rules_copy:
					if rule_key[0] == ur and env in seg_rules_copy[rule_key]:
						seg_rules_copy[rule_key].remove(env)

		# Delete UR-SR pairs with empty environments
		to_delete = [key for key, value in seg_rules_copy.items() if len(value) == 0]
		for key in to_delete:
			del seg_rules_copy[key]

		return seg_rules_copy

	def generate_feature_based_rules(self, merged_rules):
		feature_rules = set()
		examined_combinations = set()
		unmatched_rules = {(ur, sr): env_segments for (ur, sr), env_segments in merged_rules.items()}
		entailment_pairs = self.entailment_pairs
		redundant_features = self.get_redundant_features()  # Moved outside the loop for efficiency.

		for (ur, sr), env_segments in merged_rules.items():
			changed_features = self.get_changed_features(ur, sr)
			left_env_shared_features = self.get_env_shared_features(env_segments, redundant_features)


			if self.bias == "minimal":
				feature_rules.update(self.minimal_generalization(ur, changed_features, left_env_shared_features, merged_rules, entailment_pairs))
			elif self.bias == "maximal":
				feature_rules.update(self.maximal_generalization(ur, changed_features, left_env_shared_features, merged_rules, examined_combinations, entailment_pairs))

		ordered_rules = self.order_rules_by_entailment(feature_rules)
		return ordered_rules
	
	def minimal_generalization(self, ur, changed_features, left_env_shared_features, merged_rules, entailment_pairs):
		feature_rules = set()  # Initialized inside the function

		for change_feature, change_value in changed_features.items():
			ur_features = self.feature_encoding[ur].copy()
			ur_features[change_feature] = change_value
			changed_feature_segments = {segment for segment, features in self.feature_encoding.items() if self.entail_logic(ur_features, features)}

			for env_feature, env_value in left_env_shared_features.items():
				env_feature_segments = {seg for seg in self.tier if self.feature_encoding[seg].get(env_feature) == env_value}
				valid_rule = self.check_rule_validity(merged_rules, ur, changed_feature_segments, env_feature_segments, entailment_pairs)
				if valid_rule:

					rule_features = ','.join([f"{v}{f}" for f, v in changed_features.items()])
					env_features = ','.join([f"{v}{f}" for f, v in left_env_shared_features.items()])
					rule = f"{ur} -> [{rule_features}] / [{env_features}] _"
					feature_rules.add(rule)
		return feature_rules

	def maximal_generalization(self, ur, changed_features, left_env_shared_features, merged_rules, examined_combinations, entailment_pairs):
		feature_rules = set()  # Initialized inside the function

		for change_feature, change_value in changed_features.items():
			ur_features = self.feature_encoding[ur].copy()
			ur_features[change_feature] = change_value
			changed_feature_segments = {segment for segment, features in self.feature_encoding.items() if self.entail_logic(ur_features, features)}

			for env_feature, env_value in left_env_shared_features.items():

				if self.naturalness == True and change_feature != env_feature:
					continue

				combination = f"{ur} -> {change_value}{change_feature}/{env_value}{env_feature}_"

				if combination in examined_combinations:
					continue

				examined_combinations.add(combination)
				env_feature_segments = {seg for seg in self.tier if self.feature_encoding[seg].get(env_feature) == env_value}

				valid_rule = self.check_rule_validity(merged_rules, ur, changed_feature_segments, env_feature_segments, entailment_pairs)
				# if combination == 'A -> +back/+back_':
				# 	breakpoint()
				if valid_rule:
					rule = f"{ur} -> [{change_value}{change_feature}] / [{env_value}{env_feature}] _"
					feature_rules.add(rule)
		
		return feature_rules  # Return the modified feature_rules

	def check_rule_validity(self, merged_rules, ur, changed_feature_segments, env_feature_segments, entailment_pairs):

		for key, env_list in merged_rules.items():
			# ur match or is entailed by the ur of the segment-based rule here
			ur_condition = key[0] == ur # or key[0] in entailment_pairs.get(ur, []) (note: only if entailed UR also constitute as counterevidence---not realistic; e.g. /a/ -> o in finnish vs. /A/ -> a)
			if ur_condition:
				for env_segment in env_list:
					env_condition = env_segment in env_feature_segments
					sr_condition = key[1] in changed_feature_segments
					if env_condition and not sr_condition:
						# if ur == "A":
						# 	breakpoint()
						return False
					# if sr_condition and not env_condition:
					# 	return False
					
		return True

		# feature_rules = self.remove_entailed_rules(feature_rules)
		# if unmatched_rules:
		# 	print(unmatched_rules)
		# 	breakpoint()
		# for (ur, sr), env_list in unmatched_rules.items():
		# 	for i in env_list:
		# 		rule = f"{ur} -> {sr} / {i}"
		# 		ordered_rules.append(rule)
				# breakpoint()

	def order_rules_by_entailment(self, feature_rules):
		# Temporary list to hold the rules before finalizing the order
		temp_rules = []
		# Iterate over feature_rules to finalize the order
		for rule in feature_rules:
			index = len(temp_rules)
			for i in range(len(temp_rules) - 1, -1, -1):
				if self.is_entailed(rule, temp_rules[i]):
					index = i
			temp_rules.insert(index, rule)
		return temp_rules
	
	def is_entailed(self, new_rule, existing_rule):
		ur_new = new_rule.split(' -> ')[0].strip()
		ur_existing = existing_rule.split(' -> ')[0]

		# sr_new, env_new = new_rule.split(' -> ')[1].split('/')
		# sr_new = sr_new.strip().replace("[","").replace("]","")
		# env_new = env_new.split('_')[0].strip().replace("[","").replace("]","")

		# sr_existing, env_existing = existing_rule.split(' -> ')[1].split('/')
		# sr_existing = sr_existing.strip().replace("[","").replace("]","")
		# env_existing = env_existing.split('_')[0].strip().replace("[","").replace("]","")

		# Check for feature entailment in SR (left of '/')
		# sr_new_features = set(sr_new.split(','))
		# sr_existing_features = set(sr_existing.split(','))

		# if not sr_new_features.issubset(sr_existing_features):
		# 	return False

		# # Check for feature entailment in ENV (right of '/')
		# env_new_features = set(env_new.split(','))
		# env_existing_features = set(env_existing.split(','))
		
		# if not env_new_features.issuperset(env_existing_features):
		# 	return False

		# Check UR entailment
		return ur_existing in self.entailment_pairs.get(ur_new, [])

	def remove_entailed_rules(self, feature_rules):
		to_remove = []

		for rule in feature_rules:
			ur1 = rule.split(' -> ')[0]
			sr_env1 = rule.split(' -> ')[1]

			for other_rule in feature_rules:
				if rule == other_rule:
					continue  # don't compare with itself

				ur2 = other_rule.split(' -> ')[0]
				sr_env2 = other_rule.split(' -> ')[1]

				if ur2 in self.entailment_pairs.get(ur1, []) and sr_env1 == sr_env2:
					to_remove.append(other_rule)

		# Remove entailed rules
		feature_rules = {rule for rule in feature_rules if rule not in to_remove}

		return feature_rules
	

	def alpha_rules(self, feature_rules):
		consolidated_rules = []
		rules_processed = set()
		
		for rule in feature_rules:
			if rule in rules_processed:
				continue

			ur, _, features_part = rule.partition(" -> ")

			# Note: Now 'ur' is in the feature representation due to changes in 'generate_feature_based_rules'
			changes, environment = features_part.split(" / ")
			change_feature = changes[2:-1]
			value = changes[1]
			opposite_value = '+' if value == '-' else '-'

			positive_rule_same_env = f"{ur} -> [+{change_feature}] / [+{change_feature}] _"
			negative_rule_same_env = f"{ur} -> [-{change_feature}] / [-{change_feature}] _"
			positive_rule_opposite_env = f"{ur} -> [+{change_feature}] / [-{change_feature}] _"
			negative_rule_opposite_env = f"{ur} -> [-{change_feature}] / [+{change_feature}] _"
			
			if positive_rule_same_env in feature_rules and negative_rule_same_env in feature_rules:
				general_rule = f"{ur} -> [α{change_feature}] / [α{change_feature}] _"
				if general_rule not in consolidated_rules:
					consolidated_rules.append(general_rule)
				rules_processed.add(positive_rule_same_env)
				rules_processed.add(negative_rule_same_env)
				
			elif positive_rule_opposite_env in feature_rules and negative_rule_opposite_env in feature_rules:
				inverse_general_rule = f"{ur} -> [α{change_feature}] / [-α{change_feature}] _"
				if inverse_general_rule not in consolidated_rules:
					consolidated_rules.append(inverse_general_rule)
				rules_processed.add(positive_rule_opposite_env)
				rules_processed.add(negative_rule_opposite_env)
				
			else:
				if rule not in rules_processed:
					consolidated_rules.append(rule)
					rules_processed.add(rule)

		return consolidated_rules

class RuleUtilities:
	
	def initialize_UR_SR(self):
		analysis = self.data_item['analysis']
		# If the analysis does not contain a hyphen, treat it as 'Stem'
		if '-' not in analysis:
			self.data_item['ur'] = self.data_item['segmentation']
		else:
			morphemes = analysis.split('-')
			segmentation_parts = self.data_item['segmentation'].split('-')

			for idx, morpheme in enumerate(morphemes):
				if idx == 0:
					morphemes[idx] = segmentation_parts[idx]
				elif '-' + morpheme in self.UR_data:
					morphemes[idx] = self.UR_data['-' + morpheme]
				else:
					morphemes[idx] = morpheme  # Use the given morpheme if no match is found in UR_data

			self.data_item['ur'] = ''.join(morphemes)

		# Initially, set the surface representation (SR) to be the same as UR
		self.data_item['predicted_sr'] = self.data_item['ur']


	def create_tier_based_affix(self, affix):
		"""Generate a tier-based affix and index map."""
		tier_affix = []
		idx_map = []
		for idx, seg in enumerate(affix):
			if seg in self.tier:
				tier_affix.append(seg)
				idx_map.append(idx)
		return tier_affix, idx_map

	def parse_rule(self, rule):
		parts = rule.split(" -> ")
		ur = parts[0]
		change, environment = parts[1].split(" / ")
		
		# Removing underscores and spaces from the environment
		environment = environment.replace("_", "").replace(" ", "")
		
		return ur, change, environment

	def entail(self, general, specific):
		# Gate 1: Directly checking the "abstract" feature. Concrete cannot entail Abstract.
		if "abstract" in general and "abstract" in specific:
			if specific["abstract"] == "+" and general["abstract"] == "-":
				return False

		# Gate 2: Evaluating other features. For nonabstract shared features, if value in the precedent is not 0, and is not identical to the consequence, the entailment is false. 
		for feature, value in general.items():
			# We've already evaluated the abstract feature, so skip it here.
			if feature == "abstract":
				continue
			
			# If the feature is missing from the specific set.
			if feature not in specific:
				return False
			
			# If the general value is not '0' and doesn't match the specific value.
			if value != '0' and specific[feature] != value:
				return False

		# If none of the above conditions were met, return True.
		return True

	def convert_string_to_list(self,input_string):
		# Remove opening and closing brackets and split by comma
		items = input_string[1:-1].split(',')
		
		# Strip whitespace from each item and return as list
		return [item.strip() for item in items]

	def get_derived_segment(self, alternating_segment_features):
		perfect_match = None
		abstract_match = None

		for segment, features in self.feature_encoding.items():
			if all(features[key] == alternating_segment_features.get(key) for key in features) and not perfect_match:
				perfect_match = segment
			elif all(features[key] == alternating_segment_features.get(key) for key in features if key != "abstract") and not abstract_match:
				abstract_match = segment

			if perfect_match:
				break

		if not perfect_match and not abstract_match:

			raise ValueError(f"No segment found for features: {alternating_segment_features}")

		return perfect_match or abstract_match


	def create_tier_based_affix(self, affix):
		"""Generate a tier-based affix and index map."""
		tier_affix = []
		idx_map = []
		for idx, seg in enumerate(affix):
			if seg in self.tier:
				tier_affix.append(seg)
				idx_map.append(idx)
		return tier_affix, idx_map

	def parse_rule(self, rule):
		parts = rule.split(" -> ")
		ur = parts[0]
		change, environment = parts[1].split(" / ")
		
		# Removing underscores and spaces from the environment
		environment = environment.replace("_", "").replace(" ", "")
		
		return ur, change, environment

	def entail(self, general, specific):
		# Gate 1: Directly checking the "abstract" feature.
		if "abstract" in general and "abstract" in specific:
			if specific["abstract"] == "+" and general["abstract"] == "-":
				return False

		# Gate 2: Evaluating other features.
		for feature, value in general.items():
			# We've already evaluated the abstract feature, so skip it here.
			if feature == "abstract":
				continue
			
			# If the feature is missing from the specific set.
			if feature not in specific:
				return False
			
			# If the general value is not '0' and doesn't match the specific value.
			if value != '0' and specific[feature] != value:
				return False

		# If none of the above conditions were met, return True.
		return True

	def convert_string_to_list(self,input_string):
		# Remove opening and closing brackets and split by comma
		items = input_string[1:-1].split(',')
		
		# Strip whitespace from each item and return as list
		return [item.strip() for item in items]

	def get_derived_segment(self, alternating_segment_features):
		perfect_match = None
		abstract_match = None

		for segment, features in self.feature_encoding.items():
			if all(features[key] == alternating_segment_features.get(key) for key in features) and not perfect_match:
				perfect_match = segment
			elif all(features[key] == alternating_segment_features.get(key) for key in features if key != "abstract") and not abstract_match:
				abstract_match = segment

			if perfect_match:
				break

		if not perfect_match and not abstract_match:
			raise ValueError(f"No segment found for features: {alternating_segment_features}")

		return perfect_match or abstract_match

def filter_tier(sequence, tier_segments):
	"""
	Filter a sequence to only include characters that are in tier_segments.
	"""
	return ''.join([char for char in sequence if char in tier_segments])
