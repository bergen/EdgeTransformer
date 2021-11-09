#@title Imports

# Licensed under the Apache License, Version 2.0
# Adapted from Ontanon et al. (2021), Making Transformers Solve Compositional Tasks

import json
import io
import random
import time
import numpy as np
import statistics
from torch.nn.utils.rnn import pad_sequence as pad

import torch
from types import SimpleNamespace


#@title "COGS" tagging dataset generation

DATA_DIR = './data/'  # Directory containing sequence tagged json files.
DATASET_FILES = [DATA_DIR + ifile for ifile in [
				"train_seqtag.jsonl", "test_seqtag.jsonl", "dev_seqtag.jsonl","gen_seqtag.jsonl"]
]

PAD_TOKEN = "[PAD]"
PAD_PARENT = 99999

# Possible parent encodings:
# The type of parent encoding is used already in dataset generation.
PARENT_ABSOLUTE = "parent_absolute_encoding"
PARENT_RELATIVE = "parent_relative_encoding"
PARENT_ATTENTION = "parent_attention_encoding"


def decode(seq, vocab):
	out = ""
	for tok in seq:
		if tok == 0:
			return out
		out += str(vocab[tok]) + ", "
	return out


def read_cogs_datafile(filename):
	data = []
	print(filename)
	with open(filename, "r") as f:
		for line in f:
			data.append(json.loads(line))
	print(f"loaded {filename}: {len(data)} instances.")
	return data


def split_set_by_distribution(dataset):
	"""Create multiple splits based on the generalization type.

	Only COGS geneneralization dataset is annotated by the generalization type.
	"""
	distributions = []
	split = {}
	for example in dataset:
		distribution = example["distribution"]
		if distribution in split:
			split[distribution].append(example)
		else:
			distributions.append(distribution)
			split[distribution] = [example]
	for distribution in split:
		print(f"{distribution}: {len(split[distribution])}")
	return split, distributions

def pad_to_len(tensor,batch_size,padded_len,padding_value):
	padded = torch.ones(batch_size,padded_len).long()*padding_value
	padded[:,:tensor.size(1)] = tensor

	return padded

def pad_tensors(data):
	tokens,parent,role,category,noun_type,verb = [d['tokens'] for d in data],[d['parent'] for d in data], [d['role'] for d in data],\
		[d['category'] for d in data],[d['noun_type'] for d in data], [d['verb'] for d in data]


	sen_lens = [t.shape[0] for t in tokens]
	max_len = max(sen_lens)
	sen_lens = torch.tensor(sen_lens)

	mask = torch.arange(max_len)[None, :] >= sen_lens[:, None]

	batch_size = len(tokens)

	tokens = pad(tokens, batch_first=True, padding_value=0)
	tokens = pad_to_len(tokens,batch_size,max_len,padding_value=0)
	parent = pad(parent, batch_first=True, padding_value=PAD_PARENT)
	parent = pad_to_len(parent,batch_size,max_len,padding_value=PAD_PARENT)
	role = pad(role, batch_first=True, padding_value=0)
	role = pad_to_len(role,batch_size,max_len,padding_value=0)
	category = pad(category, batch_first=True, padding_value=0)
	category = pad_to_len(category,batch_size,max_len,padding_value=0)
	noun_type = pad(noun_type, batch_first=True, padding_value=0)
	noun_type = pad_to_len(noun_type,batch_size,max_len,padding_value=0)
	verb = pad(verb, batch_first=True, padding_value=0)
	verb = pad_to_len(verb,batch_size,max_len,padding_value=0)



	return tokens, parent, role, category, noun_type,verb, mask

def create_dataset_feature_tensor(dataset, feature, vocab, max_len, parent_encoding=None):
	"""Read the selected feature from the examples.

	Be carfeful about the parent encoding since we comnpare the indices to the
	attention matrix.
	"""
	feature_tensor = []
	for example in dataset:
		if feature == "parent":
			if parent_encoding == PARENT_ABSOLUTE:
				assert vocab[0] == PAD_PARENT  # padding
				assert vocab[1] == -1  # -1 means no parent
				assert vocab[2] == 0   # parent is the 1st token
				tensor = [vocab.index(x) for x in example[feature]]
			elif parent_encoding == PARENT_RELATIVE:
				assert vocab[0] == PAD_PARENT
				# Use self instead of -1 to denote no parent.
				parents = example[feature]
				tensor = [vocab.index(parents[i]-i) if parents[i] != -1 else vocab.index(0) for i in range(len(parents))]
			elif parent_encoding == PARENT_ATTENTION:
				# Use self instead of -1 to denote no parent.
				# The vocab for parent is hardcoded: [-2, 0, 1, 2, ...]
				assert vocab[0] == PAD_PARENT
				assert vocab[1] == 0
				assert vocab[2] == 1
				parents = example[feature]
				tensor = [parents[i] if parents[i] != -1 else i for i in range(len(parents))]

			else:
				raise ValueError(f"Undefined parent_encoding: {parent_encoding}")
		else:
			tensor = [vocab.index(x) for x in example[feature]]
		feature_tensor.append(tensor)
	return feature_tensor


def create_parent_ids_tensor(dataset, max_len):
	"""This is really an input mask.
	0 when there is an input token
	1 when the input token is padding.
	"""
	feature_tensor = []
	for example in dataset:
		tensor = [0]*max_len
		for i in range(len(example["tokens"]), max_len):
			tensor[i] = 1
		feature_tensor.append(tensor)
	return feature_tensor


def create_dataset_tensors(dataset,vocabs,max_len,batch_size,show_example=False,parent_encoding=None):
	tokens_tensor = create_dataset_feature_tensor(dataset, "tokens", vocabs[0],max_len)
	parent_tensor = create_dataset_feature_tensor(dataset, "parent", vocabs[1],max_len, parent_encoding)
	role_tensor = create_dataset_feature_tensor(dataset, "role", vocabs[2],max_len)
	category_tensor = create_dataset_feature_tensor(dataset, "category",vocabs[3], max_len)
	noun_type_tensor = create_dataset_feature_tensor(dataset, "noun_type",vocabs[4], max_len)
	verb_tensor = create_dataset_feature_tensor(dataset, "verb_name", vocabs[5],max_len)
	buffer_size = len(dataset)
	dataset = SimpleNamespace()

	dataset.tokens = tokens_tensor
	dataset.parent = parent_tensor
	dataset.role = role_tensor
	dataset.category = category_tensor
	dataset.noun_type = noun_type_tensor
	dataset.verb = verb_tensor

	if show_example:
		print("- Sample Example ----------------")
		print(f"tokens: {decode(tokens_tensor[0], vocabs[0])}")
		print(f"parent: {decode(parent_tensor[0], vocabs[1])}")
		print(f"role: {decode(role_tensor[0], vocabs[2])}")
		print(f"category: {decode(category_tensor[0], vocabs[3])}")
		print(f"noun_type: {decode(noun_type_tensor[0], vocabs[4])}")
		print(f"verb_name: {decode(verb_tensor[0], vocabs[5])}")
		print("---------------------------------")

	return dataset


def read_cogs_datasets(dataset_files, parent_encoding, batch_size):
	assert len(dataset_files) == 4, (
			"expected list of dataset paths in this order: train, test, dev, gen; "
			"got %s"
	) % dataset_files
	cogs_train = dataset_files[0]
	cogs_test = dataset_files[1]
	cogs_dev = dataset_files[2]
	cogs_gen = dataset_files[3]

	train_set = read_cogs_datafile(cogs_train)
	test_set = read_cogs_datafile(cogs_test)
	dev_set = read_cogs_datafile(cogs_dev)
	gen_set = read_cogs_datafile(cogs_gen)

	# Create vocabs, and calculate dataset stats:
	tokens_vocab = [PAD_TOKEN]
	# The token with index 0 has to be padding, because loss relies on it.
	# -1 is already used in the tagging datased to denote no parent,
	# so let's use -2 as the padding token.
	parent_vocab_raw = [PAD_PARENT]
	role_vocab = [PAD_TOKEN]
	category_vocab = [PAD_TOKEN]
	noun_type_vocab = [PAD_TOKEN]
	verb_name_vocab = [PAD_TOKEN]

	max_len = 0
	for example in train_set + test_set + dev_set + gen_set:
		for token in example["tokens"]:
			if token not in tokens_vocab:
				tokens_vocab.append(token)
		for token in example["parent"]:
			if token not in parent_vocab_raw:
				parent_vocab_raw.append(token)
		for token in example["role"]:
			if token not in role_vocab:
				role_vocab.append(token)
		for token in example["category"]:
			if token not in category_vocab:
				category_vocab.append(token)
		for token in example["noun_type"]:
			if token not in noun_type_vocab:
				noun_type_vocab.append(token)
		for token in example["verb_name"]:
			if token not in verb_name_vocab:
				verb_name_vocab.append(token)
		l = len(example["tokens"])
		max_len = max(max_len, l)

	if parent_encoding == PARENT_ABSOLUTE:
		parent_vocab = [PAD_PARENT, -1] + list(range(max_len))
	elif parent_encoding == PARENT_RELATIVE:
		parent_vocab = [PAD_PARENT] + list(range(-max_len+1, max_len))
	elif parent_encoding == PARENT_ATTENTION:
		parent_vocab = [PAD_PARENT] + list(range(max_len))
	else:
		raise ValueError(f"Undefined parent_encoding: {parent_encoding}")

	max_len += 1  # guarantee at least one padding token at the end
								# for "no parent"
								# Padding token is also used to stop decoding in decode().

	gen_distribution_split, gen_distributions = split_set_by_distribution(gen_set)

	print(f"n_distributions: {len(gen_distribution_split)}")
	# print(f"max_len: {max_len}")
	# print(f"tokens_vocab: {len(tokens_vocab)}  -->> {tokens_vocab}")
	# print(f"parent_vocab: {len(parent_vocab)}  -->> {parent_vocab}")
	# parent_vocab_missing = sorted(set(parent_vocab) - set(parent_vocab_raw))
	# print(f"parent indices missing from the data: {len(parent_vocab_missing)}  -->> {parent_vocab_missing}")
	# print(f"role_vocab: {len(role_vocab)}  -->> {role_vocab}")
	# print(f"category_vocab: {len(category_vocab)}  -->> {category_vocab}")
	# print(f"noun_type_vocab: {len(noun_type_vocab)}  -->> {noun_type_vocab}")
	# print(f"verb_name_vocab: {len(verb_name_vocab)}  -->> {verb_name_vocab}")


	vocabs = [tokens_vocab, parent_vocab,
						role_vocab, category_vocab,
						noun_type_vocab, verb_name_vocab]
	vocab_dict = dict(zip(["tokens_vocab", "parent_vocab",
						"role_vocab", "category_vocab",
						"noun_type_vocab", "verb_name_vocab"],
		[tokens_vocab, parent_vocab,
						role_vocab, category_vocab,
						noun_type_vocab, verb_name_vocab]))

	train_dataset = create_dataset_tensors(
			train_set, vocabs, max_len, batch_size, show_example=False, parent_encoding=parent_encoding)
	test_dataset = create_dataset_tensors(test_set, vocabs, max_len, batch_size,show_example=False, parent_encoding=parent_encoding)
	dev_dataset = create_dataset_tensors(dev_set, vocabs, max_len, batch_size, show_example=False,parent_encoding=parent_encoding)
	gen_dataset = create_dataset_tensors(gen_set, vocabs, max_len, batch_size,show_example=False, parent_encoding=parent_encoding)

	gen_split_datasets = []
	for distribution in gen_distributions:
		gen_split_datasets.append(
				create_dataset_tensors(gen_distribution_split[distribution],
															 [tokens_vocab, parent_vocab,
																role_vocab, category_vocab,
																noun_type_vocab, verb_name_vocab],
															 max_len,
															 batch_size,
															 parent_encoding=parent_encoding))

	test_sets = [test_dataset, dev_dataset
							] + gen_split_datasets + [gen_dataset]
	test_sets_names = ["test", "dev"] + gen_distributions + ["gen"]

	return train_dataset, test_sets, test_sets_names, vocab_dict, max_len


def set_up_cogs(parent_encoding, batch_size):
	return read_cogs_datasets(DATASET_FILES, parent_encoding, batch_size)


