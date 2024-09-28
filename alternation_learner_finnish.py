import pprint
pp = pprint.PrettyPrinter(indent=4)
from core_learner import *
import re
class RuleApplication(RuleUtilities):

	def __init__(self, bias, data_item, phonotactics, tier, UR_data, grammar_alt, abstract_segments, feature_encoding):
		self.grammar_alt = grammar_alt
		self.bias = bias
		self.data_item = data_item
		self.UR_data = UR_data
		self.initialize_UR_SR()
		self.phonotactics = phonotactics
		self.tier = tier
		self.abstract_segments = abstract_segments
		self.feature_encoding = feature_encoding

	def extract_stem_environment(self, stem):
		"""Extracts the phonological environment (left) from the stem."""
		# a = next((char for char in reversed(stem) if char in self.tier), "nontier")
		# print(a)
		# breakpoint()
		return next((char for char in reversed(stem) if char in self.tier), "nontier")
	
	def default_rules(self, combined_sr):
		# Replacing all characters in abstract_segments_long and abstract_segments_short
		for char in ['A']:
			combined_sr = combined_sr.replace(char, 'ä')
		for char in ['U']:
			combined_sr = combined_sr.replace(char, 'y')	

		# Replace "-V" with "-" followed by the symbol immediately preceding "-"
		i = 0
		while i < len(combined_sr) - 1:
			if combined_sr[i] == '-' and combined_sr[i + 1] == 'V':
				if i > 0:
					preceding_symbol = combined_sr[i - 1]
					combined_sr = combined_sr[:i] + '-' + preceding_symbol + combined_sr[i + 2:]
				else:
					# Handle edge case where '-' is the first character
					i += 1
					continue
			i += 1

		return combined_sr

	
	def apply_rules(self):
		sr = self.data_item['predicted_sr']
		stem, *affixes = sr.split('-')
		new_sr = [stem]  # Initialize with the original stem
		previous_environment = self.extract_stem_environment(stem)
		# breakpoint()
		for idx_affix, affix in enumerate(affixes):
			modified_affix = list(affix)

			for idx_seg, seg in enumerate(affix):
				if previous_environment == "nontier" and idx_affix == 0:
					previous_environment = seg
					continue

				if self.bias == "segmental":
					previous_environment, modified_affix = self.apply_segmental_rule(seg, previous_environment, modified_affix, idx_seg)
				else:
					previous_environment, modified_affix = self.apply_feature_based_rule(seg, previous_environment, modified_affix, idx_seg)

			new_sr.append(''.join(modified_affix))

		self.data_item['predicted_sr'] = self.default_rules('-'.join(new_sr))

	def apply_segmental_rule(self, seg, previous_environment, modified_affix, idx_seg):
		tier_and_abstract = self.tier + self.abstract_segments

		if seg in tier_and_abstract:
			for rule, replacements in self.grammar_alt.items():
				from_seg, to_seg = rule
				if seg == from_seg and previous_environment in replacements:
					modified_affix[idx_seg] = to_seg
					return to_seg, modified_affix
		return previous_environment, modified_affix

	def apply_feature_based_rule(self, seg, previous_environment, modified_affix, idx_seg):
		if seg in self.abstract_segments:
			alternating_segment_features = dict(self.feature_encoding[seg])

			for rule in self.grammar_alt:
				ur_in_rule, change, rule_environment = self.parse_rule(rule)
				change = self.convert_string_to_list(change)
				ur_in_rule_features = self.feature_encoding[ur_in_rule]
				previous_environment_features = self.feature_encoding[previous_environment]
				
				if rule_environment != "[]":  # special treatment for empty feature bundle
					rule_environment_features = {feature_value_pair[1:]: feature_value_pair[0] for feature_value_pair in self.convert_string_to_list(rule_environment)}
				else:
					rule_environment_features = {}

				if self.entail(ur_in_rule_features, alternating_segment_features) and self.entail(rule_environment_features, previous_environment_features):
					for feature_spec in change:
						feature_spec = feature_spec.strip()
						sign = feature_spec[0]
						feature = feature_spec[1:]
						alternating_segment_features[feature] = sign

			derived_segment = self.get_derived_segment(alternating_segment_features)
			modified_affix[idx_seg] = derived_segment
			return derived_segment, modified_affix # derived_segment will be the new previous_environment
		
		elif seg in self.tier:
			previous_environment = seg
			# comment: if just concrete vowel, the SR stays unchanged, but update the previous environment
		return previous_environment, modified_affix

def matching_strings(str1, str2, feature_encoding):
	if str1 == str2:
		return True
	
	# Check if lengths are the same
	if len(str1) != len(str2):
		return False

	# Define characters to ignore differences between long and short
	# ignored_pairs = {("a", "á"), ("á", "a"), ("e", "é"), ("é", "e")}
	ignored_pairs = {}
	# Iterate through segments
	for i in range(len(str1)):
		char1, char2 = str1[i], str2[i]

		# If characters are in ignored_pairs, continue to the next iteration
		if (char1, char2) in ignored_pairs:
			continue

		features1 = feature_encoding[char1]
		features2 = feature_encoding[char2]

		# Check if features are same, except for "long"
		for feature, value in features1.items():
			if feature != "long" and value != features2.get(feature, None):
				# print(str1, str2)
				# breakpoint()
				return False
				
	# If the loop finished without returning False
	return True



# Test the function
if __name__ == "__main__":
	'''Step 1: learn phonotactics'''
	# Hyperparameters and configurations
	language = "finnish"

	def run_experiment(phonotactics_hypothesis,threshold, 
					bias, split_ratio, 
					language=language, structure='nonlocal', filter=True, 
					padding=False, confidence=0.975, 
					penalty_weight=10, model='filtering'):
		if language == 'finnish':
			# TrainingFile = 'data/finnish/childes_train.txt'
			# TrainingFile = 'data/finnish/morpho_train.txt'
			TrainingFile = 'data/finnish/morpho-fin-updated.txt'
			# TrainingFile = 'data/finnish/childes_adult_train.txt'
			FeatureFile = 'data/finnish/features.txt'
			# TestingFile = 'data/finnish/blick-test-Zimmer.txt'
			TestingFile = 'data/finnish/blick_test.txt'
			WugFile = None # if task 1
			URFile = "data/finnish/finnish_ur_manual.txt"

		splitter = DataSplitter(TrainingFile, split=split_ratio, test_file=WugFile)
		train_data, test_data = splitter.train_data, splitter.test_data

		learned_phonotactics, tier = phonotactic_learner(train_data,FeatureFile,TestingFile, split_ratio, language, structure,filter,padding,confidence,penalty_weight,threshold,model)
		if not phonotactics_hypothesis:
			learned_phonotactics = None
		# breakpoint()

		# learned_phonotactics = {('ɑ', 'e'),('ɑ', 'i'), ('i', 'ɑ'), ('ø', 'i'), ('o', 'e'), ('ɑ', 'ø'), ('i', 'y'), ('ɯ', 'o'), ('y', 'ɯ'), ('o', 'ɯ'), ('ɯ', 'ø'), ('ɑ', 'y'), ('ø', 'u'), ('e', 'u'), ('y', 'ɑ'), ('u', 'y'), ('e', 'ɑ'), ('ɯ', 'y'), ('e', 'ɯ'), ('ø', 'ɑ'), ('ø', 'ɯ'), ('u', 'i'), ('i', 'u'), ('ɑ', 'o'), ('ɑ', 'u'), ('y', 'o'), ('o', 'o'), ('y', 'ø'), ('o', 'ø'), ('ɯ', 'i'), ('ɯ', 'u'), ('i', 'ɯ'), ('e', 'o'), ('e', 'ø'), ('ø', 'o'), ('u', 'e'), ('o', 'y'), ('u', 'ɯ'), ('ø', 'ø'), ('ɯ', 'e'), ('e', 'y'), ('y', 'i'), ('o', 'i'), ('i', 'o'), ('y', 'u'), ('i', 'ø'), ('u', 'o'), ('u', 'ø')}
		# tier = ['ɑ', 'e', 'o', 'ø', 'u', 'y', 'i', 'ɯ']
		# Tune the hyperparameters, and plot how it changes the F-score as we change the MaxThreshold
		# hyperparameter_tuning(phonotactics, TrainingFile, FeatureFile, JudgementFile, TestingFile, MatrixFile, humanJudgement, language)
		'''Step 2: Use phonotactics from Finnish to facilitate alternation learning'''
		UR_data = read_ur_file(URFile)
		rg = RuleLearner(UR_data, train_data, FeatureFile, tier, learned_phonotactics, naturalness = True)
		rg.bias = bias # or maximal  minimal if miniaml generalization (tighest fit in feature bundles)
		UR_data, abstract_segments,feature_encoding  = rg.ur, rg.abstract_segments, rg.feature_encoding
		seg_rules_raw = rg.generate_phonological_rules()
		print("seg_rules_raw:")
		pp.pprint(seg_rules_raw)

		if bias != "segmental":
			grammar_alt = rg.generate_feature_based_rules(seg_rules_raw)
		else:
			grammar_alt = rg.remove_conflicting_rules(seg_rules_raw)
		pp.pprint("bias: " + str(bias))

		pp.pprint(grammar_alt)


		# Define the output file name format
		output_filename_template = "errors/{language}/{language}_{threshold}_{bias}.txt"

		# Before the loop, define the output file name based on current settings
		output_filename = output_filename_template.format(language=language, threshold=threshold, bias=bias)

		# Ensure the directory exists before trying to write to the file
		os.makedirs(os.path.dirname(output_filename), exist_ok=True)
		# Open the file in write mode to clear it before starting the loop
		with open(output_filename, 'w') as file:
			pass  # Simply opening in write mode truncates the file

		tier += abstract_segments

		correct_predictions = 0
		valid_data_count = 0  # Initialize counter for valid data items
		skip_tier_based = 0
		skip_normalized = 0
		# skip_sf = 0
		skip_predicted = 0


		# test_item = {'sf': 'ylhäältäpäin', 'freq': '1', 'analysis': 'ylhältä_ADV-ABL-päin_ADV', 'segmentation': 'ylhä-ltä-päin', 'ur': 'ylhä-ltA-päin', 'predicted_sr': 'ylhä-ltä-päin'}		
		# rule_applicator = RuleApplication(bias, test_item, learned_phonotactics, tier, UR_data, grammar_alt, abstract_segments, feature_encoding)
		# rule_applicator.apply_rules()
		# breakpoint()		# rule_applicator = RuleApplication(bias, test_item, learned_phonotactics, tier, UR_data, grammar_alt, abstract_segments, feature_encoding)
		# rule_applicator.apply_rules()

		for item in test_data:
			# breakpoint()
			# Instantiate the class for the current item
			rule_applicator = RuleApplication(bias, item, learned_phonotactics, tier, UR_data, grammar_alt, abstract_segments, feature_encoding)

			rule_applicator.initialize_UR_SR()
			# Apply rules to the item
			normalized_ur = unicodedata.normalize('NFC', rule_applicator.data_item['ur'])
			normalized_segmentation = unicodedata.normalize('NFC', rule_applicator.data_item['segmentation'])
			
			tier_temp = tier + ["e","i"] # dirty fix for measuring the length in rule application
			tier_based_ur = filter_tier(normalized_ur, tier_temp)
			tier_based_segmentation = filter_tier(normalized_segmentation, tier_temp)

			if len(rule_applicator.data_item["analysis"].split("-")) < 2:
				continue

			if (len(tier_based_ur) != len(tier_based_segmentation) or 
				len(normalized_ur) != len(normalized_segmentation)):
				skip_tier_based += 1
				# print(item)
				# breakpoint()
				continue
			# if len(item["sf"]) != len(normalized_segmentation) - 1:  # Added the new condition here
			# 	skip_sf += 1
			# 	continue
			# if "@" in rule_applicator.data_item['predicted_sr']:
			# 	skip_predicted += 1
			# 	continue
			# else:

			valid_data_count += 1  # Increment valid data counter
		
			rule_applicator.apply_rules()

			predicted = rule_applicator.data_item["predicted_sr"]
			actual = rule_applicator.data_item["segmentation"]
			normalized_predicted = unicodedata.normalize('NFC', predicted)
			normalized_actual = unicodedata.normalize('NFC', actual)
		
			# Filter for the tier before comparison
			tier_filtered_predicted = filter_tier(normalized_predicted, tier)  # Assuming tier is a list of characters you want to keep
			tier_filtered_actual = filter_tier(normalized_actual, tier)

			if matching_strings(tier_filtered_predicted, tier_filtered_actual, feature_encoding):
				correct_predictions += 1
				# print(rule_applicator.data_item)
				# breakpoint()
			else:
				print(rule_applicator.data_item)

				with open(output_filename, 'a') as file:
					file.write(str(rule_applicator.data_item) + "\n")

		accuracy = (correct_predictions / valid_data_count) * 100  # Use valid_data_count instead of len(test_data)
		print(f"Accuracy: {accuracy:.2f}%")
		print(f"Correct/Total: {correct_predictions}/{valid_data_count}")
		print(f"Skipped due to tier-based/normalized mismatch: {skip_tier_based}")

		return accuracy
	
	split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# split_ratios = [1/13, 2/13, 3/13, 4/13, 5/13, 6/13, 7/13, 8/13, 9/13, 10/13,11/13,12/13]
	threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	split_ratios = [0.9]
	# threshold_values = [0.1]

	# Specify the combinations of bias and segmental you want to run
	biases = [
		"maximal",
		# "minimal",
		# "segmental",  # If you prefer minimal for segmental=True, or change to maximal if needed.
	]
	phonotactics_hypothesis = True

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
	hyperparameter_alternation(phonotactics_hypothesis, threshold_values, biases, split_ratios, language, run_experiment)
