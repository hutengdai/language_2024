import pprint
pp = pprint.PrettyPrinter(indent=4)
from core_learner import *

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
		return next((char for char in reversed(stem) if char in self.tier), "nontier")

	def apply_rules(self):
		sr = self.data_item['predicted_sr']
		stem, *affixes = sr.split('-')
		new_sr = [stem]  # Initialize with the original stem
		previous_environment = self.extract_stem_environment(stem)
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

		self.data_item['predicted_sr'] = '-'.join(new_sr)

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


# Test the function
if __name__ == "__main__":
	'''Step 1: learn phonotactics'''
	# Hyperparameters and configurations
	language = "turkish"

	def run_experiment(phonotactics_hypothesis,threshold, 
					bias, split_ratio, 
					language=language, structure='nonlocal', filter=True, 
					padding=False, confidence=0.975, 
					penalty_weight=10, model='filtering'):
		if language == 'turkish':
			# TrainingFile = 'data/turkish/childes_train.txt'
			# TrainingFile = 'data/turkish/morpho_train.txt'
			TrainingFile = 'data/turkish/morpho.txt'
			# TrainingFile = 'data/turkish/childes.txt'
			# TrainingFile = 'data/turkish/childes_adult_train.txt'
			FeatureFile = 'data/turkish/features.txt'
			# TestingFile = 'data/turkish/blick-test-Zimmer.txt'
			TestingFile = 'data/turkish/blick-test.txt'
			# WugFile = 'data/turkish/childes_test.txt'
			# WugFile = 'data/turkish/childes_child_test_noadult.txt'
			# WugFile = 'data/turkish/morpho_test.txt'
			WugFile = None # if task 1
			URFile = "data/turkish/ur.txt"

		splitter = DataSplitter(TrainingFile, split=split_ratio, test_file=WugFile)
		train_data, test_data = splitter.train_data, splitter.test_data

		learned_phonotactics, tier = phonotactic_learner(train_data,FeatureFile,TestingFile, split_ratio, language, structure,filter,padding,confidence,penalty_weight,threshold,model)
		if not phonotactics_hypothesis:
			learned_phonotactics = None
		# learned_phonotactics = {('ɑ', 'e'),('ɑ', 'i'), ('i', 'ɑ'), ('ø', 'i'), ('o', 'e'), ('ɑ', 'ø'), ('i', 'y'), ('ɯ', 'o'), ('y', 'ɯ'), ('o', 'ɯ'), ('ɯ', 'ø'), ('ɑ', 'y'), ('ø', 'u'), ('e', 'u'), ('y', 'ɑ'), ('u', 'y'), ('e', 'ɑ'), ('ɯ', 'y'), ('e', 'ɯ'), ('ø', 'ɑ'), ('ø', 'ɯ'), ('u', 'i'), ('i', 'u'), ('ɑ', 'o'), ('ɑ', 'u'), ('y', 'o'), ('o', 'o'), ('y', 'ø'), ('o', 'ø'), ('ɯ', 'i'), ('ɯ', 'u'), ('i', 'ɯ'), ('e', 'o'), ('e', 'ø'), ('ø', 'o'), ('u', 'e'), ('o', 'y'), ('u', 'ɯ'), ('ø', 'ø'), ('ɯ', 'e'), ('e', 'y'), ('y', 'i'), ('o', 'i'), ('i', 'o'), ('y', 'u'), ('i', 'ø'), ('u', 'o'), ('u', 'ø')}
		# tier = ['ɑ', 'e', 'o', 'ø', 'u', 'y', 'i', 'ɯ']
		# Tune the hyperparameters, and plot how it changes the F-score as we change the MaxThreshold
		# hyperparameter_tuning(phonotactics, TrainingFile, FeatureFile, JudgementFile, TestingFile, MatrixFile, humanJudgement, language)
		# breakpoint()
		'''Step 2: Use phonotactics from Turkish to facilitate alternation learning'''
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

		# test_item = {'sf': 'ɑʃɑɰɯdɑkilerden', 'freq': '103', 'analysis': 'Stem-loc-ki-pl-abl', 'segmentation': 'ɑʃɑɰɯ-dɑ-ki-ler-den', 'ur': 'ɑʃɑɰɯ-dA-ki-lAr-dAn', 'predicted_sr': 'ɑʃɑɰɯ-dɑ-ki-lɑr-dɑn'}
		# rule_applicator = RuleApplication(bias, test_item, learned_phonotactics, tier, UR_data, grammar_alt, abstract_segments, feature_encoding)
		# rule_applicator.apply_rules()

		correct_predictions = 0
		total_items = 0
		for item in test_data:

			# Instantiate the class for the current item
			rule_applicator = RuleApplication(bias, item, learned_phonotactics, tier, UR_data, grammar_alt, abstract_segments, feature_encoding)
			
			# Skip items where analysis is "Stem"
			if len(rule_applicator.data_item["analysis"].split("-")) < 2:
				continue

			# Apply rules to the item
			rule_applicator.apply_rules()
			
			predicted = rule_applicator.data_item["predicted_sr"].replace('t', 'd')
			actual = rule_applicator.data_item["segmentation"].replace('t', 'd')
			
			if predicted == actual:
				correct_predictions += 1
				# print(rule_applicator.data_item)
				# breakpoint()
			else:
				print(rule_applicator.data_item)
				# If the prediction is wrong, write the data_item to the file
				with open(output_filename, 'a') as file:
					file.write(str(rule_applicator.data_item) + "\n")
			# Increase total items count
			total_items += 1


		if total_items == 0:
			breakpoint()
		accuracy = (correct_predictions / total_items) * 100
		print(f"Accuracy: {accuracy:.2f}%")
		print(f"Correct/Total: {correct_predictions}/{total_items}")
		return accuracy
	
	split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# split_ratios = [1/13, 2/13, 3/13, 4/13, 5/13, 6/13, 7/13, 8/13, 9/13, 10/13,11/13,12/13]
	threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	# split_ratios = [1]
	threshold_values = [0.0]

	# Specify the combinations of bias and segmental you want to run
	biases = [
		"maximal",
		"minimal",
		"segmental",  # If you prefer minimal for segmental=True, or change to maximal if needed.
	]
	phonotactics_hypothesis = True # use phonotactics filter (True) or not (False)

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
