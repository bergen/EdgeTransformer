from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset

from model import Parser

import numpy as np
import os
import logging
import random

import torch
import argparse

import load
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="edge_transformer")
parser.add_argument('--optimizer',type=str,default="AdamW",choices=["AdamW","Adam"])
parser.add_argument('--scheduler',type=str,default="linear_warmup",choices=["linear_warmup"])
parser.add_argument('--lr',type=float, default=5e-4)
parser.add_argument('--epochs',type=int, default=200)
parser.add_argument('--batch_size',type=int, default=100)
parser.add_argument('--gen_batch_size',type=int, default=5)
parser.add_argument('--num_warmup_steps',type=int, default=1000)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-08)
parser.add_argument('--accum_grad',type=int, default=1)
parser.add_argument('--num_message_rounds',type=int, default=3)
parser.add_argument('--dropout',type=float, default=0.1)
parser.add_argument('--dim',type=int, default=64)
parser.add_argument('--num_heads',type=int, default=4)
parser.add_argument('--output_dir', default="test_cogs")
parser.add_argument('--input_encoding', default='relational',choices=['positional','relational'])
parser.add_argument('--share_layers',type=bool,default=True)
parser.add_argument('--lesion_scores', action='store_true')
parser.add_argument('--lesion_values', action='store_true')
parser.add_argument('--edge_attention_flat', action='store_true')
parser.add_argument('--ff_factor',type=int,default=4,help="Factor difference between attention dim and feedforward dim")
parser.add_argument('--progress_refresh_rate',type=int,default=1)
parser.add_argument('--num_gpu',type=int,default=1)
parser.add_argument('--precision',type=int,default=16,choices=[16,32])
parser.add_argument('--seed',type=int,default=42)


cl_args = parser.parse_args()



def train():
	PARENT_ENCODING = "parent_attention_encoding"
	train_dataset, test_sets, test_sets_names, vocabs, max_len = load.set_up_cogs(PARENT_ENCODING,1)

	cl_args.token_vocab_size = len(vocabs['tokens_vocab'])
	cl_args.parent_vocab_size = len(vocabs['parent_vocab'])
	cl_args.role_vocab_size = len(vocabs['role_vocab'])
	cl_args.category_vocab_size = len(vocabs['category_vocab'])
	cl_args.noun_type_vocab_size = len(vocabs['noun_type_vocab'])
	cl_args.verb_name_vocab_size = len(vocabs['verb_name_vocab'])


	training_set = CogsDataset(train_dataset)
	val_set = CogsDataset(test_sets[1])
	#test_set = CogsDataset(test_sets[0])
	gen_set = CogsDataset(test_sets[-1])

	max_training_len = max([len(s) for s in training_set.tokens])
	max_val_len = max([len(s) for s in val_set.tokens])
	max_gen_len = max([len(s) for s in gen_set.tokens])


	train_collator = make_collator(vocabs,max_training_len)
	val_collator = make_collator(vocabs,max_val_len)
	gen_collator = make_collator(vocabs,max_gen_len)

	data_params = {'batch_size': cl_args.batch_size,
				'shuffle': True,
				'drop_last':True,
				'num_workers':8
				}

	val_params = {'batch_size': cl_args.batch_size,
				'shuffle': False,
				'num_workers':8
				}

	gen_params = {'batch_size': cl_args.gen_batch_size,
				'shuffle': False,
				'num_workers':8
				}

	training_loader = DataLoader(training_set, **data_params,collate_fn=train_collator)
	val_loader = DataLoader(val_set, **val_params,collate_fn=val_collator)
	#test_loader = DataLoader(test_fset, **data_params,collate_fn,ollator)
	gen_loader = DataLoader(gen_set, **gen_params,collate_fn=gen_collator)


	optimizer_args = {'lr':cl_args.lr,'betas':(cl_args.beta1,cl_args.beta2),'eps':cl_args.eps}
	num_training_steps = len(training_set) / (cl_args.batch_size * cl_args.accum_grad) * cl_args.epochs
	scheduler_args = {'num_warmup_steps':cl_args.num_warmup_steps,'num_training_steps':num_training_steps}
	cl_args.optimizer_args = optimizer_args
	cl_args.scheduler_args = scheduler_args
	model = Parser(cl_args)



	trainer = pl.Trainer(
			gpus=cl_args.num_gpu,
			max_epochs=cl_args.epochs,
			gradient_clip_val=cl_args.max_grad_norm,
			progress_bar_refresh_rate=cl_args.progress_refresh_rate,
			precision=cl_args.precision
		)

	trainer.fit(model,training_loader,val_loader)
	gen_result = trainer.test(model, test_dataloaders=gen_loader)


def make_collator(vocabs,max_len):
	def collate(data):
		batched = {}
		tokens,parent,role,category,noun_type,verb,mask = load.pad_tensors(data)
		batched['tokens'] = tokens
		batched['parent'] = parent
		batched['role'] = role
		batched['category'] = category
		batched['noun_type'] = noun_type
		batched['verb'] =  verb
		batched['mask'] = mask


		return batched
	return collate

class CogsDataset(Dataset):

	def __init__(self, dataset):
		self.tokens = dataset.tokens 
		self.parent = dataset.parent 
		self.role = dataset.role
		self.category = dataset.category 
		self.noun_type = dataset.noun_type
		self.verb = dataset.verb


	def __len__(self):
		return len(self.tokens)

	def __getitem__(self,index):

		item = {
			'tokens': torch.LongTensor(self.tokens[index]),
			'parent': torch.LongTensor(self.parent[index]),
			'role': torch.LongTensor(self.role[index]),
			'category': torch.LongTensor(self.category[index]),
			'noun_type': torch.LongTensor(self.noun_type[index]),
			'verb': torch.LongTensor(self.verb[index])
		}

	
		return item

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
