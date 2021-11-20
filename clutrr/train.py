import os
import logging
import sys
import csv
import argparse
import re
import ast
import random
import numpy as np
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from model import EdgeTransformer
from rat import RAT


parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="edge_transformer")
parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=50)
parser.add_argument('--batch_size',type=int, default=400)
parser.add_argument('--num_message_rounds',type=int, default=8)
parser.add_argument('--dropout',type=float, default=0.2)
parser.add_argument('--dim',type=int, default=200)
parser.add_argument('--num_heads',type=int, default=4)
parser.add_argument('--max_grad_norm',type=float,default=1.0)
parser.add_argument('--share_layers', dest='share_layers', action='store_true')
parser.add_argument('--no_share_layers', dest='share_layers', action='store_false')
parser.set_defaults(share_layers=True)
parser.add_argument('--data_path',type=str,default='data_9b2173cf')
parser.add_argument('--lesion_values', action='store_true')
parser.add_argument('--lesion_scores',  action='store_true')
parser.add_argument('--update_relations', action='store_true') #this is for relation transformer
parser.add_argument('--flat_attention', action='store_true') 
parser.add_argument('--zero_init', dest='zero_init', action='store_true')  #initialization strategy for relation aware transformer
parser.add_argument('--random_init', dest='zero_init', action='store_false')
parser.set_defaults(zero_init=True)
parser.add_argument('--optimizer',type=str,default="Adam")
parser.add_argument('--scheduler',type=str,default="linear_warmup")
parser.add_argument('--num_warmup_steps',type=int,default=100)
parser.add_argument('--ff_factor',type=int,default=4)
parser.add_argument('--log_file',type=str,default='logs/clutrr_log_file.csv')
parser.add_argument('--precision',type=int,default=32,choices=[16,32])
parser.add_argument('--seed',type=int,default=42)


cl_args = parser.parse_args()


def train():

	train_loader, validation_loader, test_loaders, test_filenames = load_files()
	
	optimizer_args = {'lr':cl_args.lr}
	num_training_steps = cl_args.epochs*len(train_loader)
	scheduler_args = {'num_warmup_steps':cl_args.num_warmup_steps,'num_training_steps':num_training_steps}
	cl_args.optimizer_args = optimizer_args
	cl_args.scheduler_args = scheduler_args
	

	trainer = pl.Trainer(
			gpus=1,
			max_epochs=cl_args.epochs,
			gradient_clip_val=cl_args.max_grad_norm,
			progress_bar_refresh_rate=1,
			precision=cl_args.precision
		)

	if cl_args.model_type=='edge_transformer':
		model = EdgeTransformer(cl_args)
	elif cl_args.model_type=='rat':
		model = RAT(cl_args)
	trainer.fit(model,train_loader,validation_loader)

	for i in range(len(test_loaders)):
		test_loader = test_loaders[i]
		test_filename = test_filenames[i]
		print(test_filename)
		trainer.test(model,test_dataloaders=test_loader)


##Load data##


def read_datafile(filename):
	edge_ls = []
	edge_labels_ls = []
	query_edge_ls = []
	query_label_ls = []
	with open(filename, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			edges = row['story_edges']
			edges = ast.literal_eval(edges)
			edge_labels = ast.literal_eval(row['edge_types'])
			query_edge = ast.literal_eval(row['query_edge'])
			query_label = row['target']
			edge_ls.append(edges)
			edge_labels_ls.append(edge_labels)
			query_edge_ls.append(query_edge)
			query_label_ls.append(query_label)

	data = {'edges':edge_ls,'edge_labels':edge_labels_ls,'query_edge':query_edge_ls,'query_label':query_label_ls}

	print(f"loaded {filename}: {len(data)} instances.")
	return data

def edge_labels_to_indices(ls,unique=None):
	if unique is None:
		unique = []
		for labels in ls:
			unique.extend(labels)
		unique = list(set(unique))

	relabeled = [list(map(lambda y: unique.index(y)+1,x)) for x in ls]
	return relabeled, unique

def query_labels_to_indices(ls,unique=None):
	if unique is None:
		unique = list(set(ls))
	
	relabeled = list(map(unique.index,ls))
	return relabeled, unique

def batch_edges(edges,edge_labels):
	batch_size = len(edges)
	lens = torch.tensor(list(map(lambda x: torch.max(x)+1,edges)))
	max_len = max(lens)

	mask = torch.arange(max_len)[None, :] >= lens[:, None]



	batch = []
	for i in range(batch_size):
		s = torch.zeros(max_len,max_len).long()
		edge = edges[i]
		lab = edge_labels[i]
		s[edge[:,0],edge[:,1]] = lab
		batch.append(s)
	batch = torch.stack(batch)

	return batch, mask

def collate(data):
	batched = {}
	batch_size = len(data)
	edges = [d['edges'] for d in data]
	edge_labels = [d['edge_labels'] for d in data]
	query_edge = [d['query_edge'] for d in data]
	query_label = [d['query_label'] for d in data]


	batched_edges, mask = batch_edges(edges,edge_labels)
	batched_query_edges = torch.stack(query_edge)
	batched_query_edges = torch.cat((torch.arange(batch_size).unsqueeze(1),batched_query_edges),dim=1)
	

	batched_query_labels = torch.tensor(query_label)

	batched['batched_graphs']=batched_edges
	batched['query_edges'] = batched_query_edges
	batched['query_labels'] = batched_query_labels
	batched['masks'] = mask

	return batched

class ClutrrDataset(Dataset):

	def __init__(self, dataset,unique_edge_labels=None,unique_query_labels=None):
		self.edges = dataset['edges']
		self.edge_labels, unique_edge_labels = edge_labels_to_indices(dataset['edge_labels'],unique_edge_labels)
		self.query_edge = dataset['query_edge']
		self.query_label, unique_query_labels = query_labels_to_indices(dataset['query_label'],unique_query_labels)
		
		self.unique_edge_labels = unique_edge_labels
		self.unique_query_labels = unique_query_labels
		self.num_edge_labels = len(unique_edge_labels)
		self.num_query_labels = len(unique_query_labels)

	def __len__(self):
		return len(self.edges)

	def __getitem__(self,index):
		item = {
			'edges': torch.LongTensor(self.edges[index]),
			'edge_labels': torch.LongTensor(self.edge_labels[index]),
			'query_edge': torch.LongTensor(self.query_edge[index]),
			'query_label': torch.LongTensor([self.query_label[index]]),
		}
		#print(self.edges[index])
		#print(self.edge_labels[index])
		#assert len(self.edges[index])==len(self.edge_labels[index])

	
		return item

def load_files():
	train_filename = get_filenames('train')[0]
	test_filenames = get_filenames('test')


	data_params = {'batch_size': cl_args.batch_size,
				'shuffle': False,
				'drop_last':False,
				'num_workers':8
				}

	test_params = {'batch_size': cl_args.batch_size,
				'shuffle': False,
				'drop_last':False,
				'num_workers':8
				}
	
	training_data = read_datafile(train_filename)
	training_data = ClutrrDataset(training_data)

	unique_edge_labels = training_data.unique_edge_labels
	unique_query_labels = training_data.unique_query_labels
	cl_args.edge_types = training_data.num_edge_labels+1
	cl_args.target_size = training_data.num_query_labels

	training_len = int(0.8*len(training_data))
	validation_len = len(training_data) - training_len
	training_set, validation_set = torch.utils.data.random_split(training_data, [training_len, validation_len])

	training_loader = DataLoader(training_set, **data_params,collate_fn=collate)
	validation_loader = DataLoader(validation_set, **data_params,collate_fn=collate)

	
	test_loaders = []
	for test_filename in test_filenames:
		test_data = read_datafile(test_filename)
		test_data = ClutrrDataset(test_data,unique_edge_labels,unique_query_labels)
		test_loader = DataLoader(test_data, **test_params,collate_fn=collate)
		test_loaders.append(test_loader)
	return training_loader,validation_loader,test_loaders,test_filenames


def get_filenames(dataset_type):
	data_dir = os.path.join('./data',cl_args.data_path)
	files = []
	for file in os.listdir(data_dir):
		if file.endswith(dataset_type+'.csv'):
			fname = os.path.join(data_dir, file)
			files.append(fname)
	return files

def set_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	seed_everything(seed=seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


if __name__=="__main__":
	set_random_seed(cl_args.seed)
	train()

