import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, LayerNorm, Dropout
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

from transformers import T5EncoderModel,T5Config
from transformers import get_linear_schedule_with_warmup


import pytorch_lightning as pl


def _get_clones(module, N):
	return ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class EdgeAttentionFlat(nn.Module):
	def __init__(self,d_model, num_heads, dropout,model_config):
		super(EdgeAttentionFlat, self).__init__()
		# We assume d_v always equals d_k

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		self.linears = _get_clones(nn.Linear(d_model, d_model,bias=False), 4)
		
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

		
	def forward(self, query, key, value, mask=None):
		num_batches = query.size(0)
		num_nodes = query.size(1)
		

		k,v,q = [l(x) for l, x in zip(self.linears, (key, value, query))]
		k = k.view(num_batches,num_nodes,num_nodes,self.num_heads,self.d_k)
		v = v.view_as(k)
		q = q.view_as(k)

		scores_r = torch.einsum("bxyhd,bxzhd->bxyzh",q,k) / math.sqrt(self.d_k)
		scores_r = scores_r.masked_fill(mask.unsqueeze(4), float('-inf'))
		scores_l = torch.einsum("bxyhd,bzyhd->bxyzh",q,k) / math.sqrt(self.d_k)
		scores_l = scores_l.masked_fill(mask.unsqueeze(4), float('-inf'))
		scores = torch.cat((scores_r,scores_l),dim=3)
		
		att = F.softmax(scores,dim=3)
		att = self.dropout(att)
		att_r,att_l = torch.split(att,scores_r.size(3),dim=3)

		x_r = torch.einsum("bxyzh,bxzhd->bxyhd",att_r,v)
		x_l = torch.einsum("bxyzh,bzyhd->bxyhd",att_l,v)

		x = x_r+x_l
		x = torch.reshape(x,(num_batches,num_nodes,num_nodes,self.d_model))

		return self.linears[-1](x)

class EdgeAttention(nn.Module):
	def __init__(self,d_model, num_heads, dropout,model_config):
		super().__init__()
		# We assume d_v always equals d_k

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		self.linears = _get_clones(nn.Linear(d_model, d_model), 6)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

		self.lesion_scores = model_config['lesion_scores']
		self.lesion_values = model_config['lesion_values']

	def forward(self, query, key, value, mask=None):
		num_batches = query.size(0)
		num_nodes = query.size(1)
		
		left_k, right_k, left_v, right_v, query = [l(x) for l, x in zip(self.linears, (key, key, value, value, key))]
		left_k = left_k.view(num_batches,num_nodes,num_nodes,self.num_heads,self.d_k)
		right_k = right_k.view_as(left_k)
		left_v = left_v.view_as(left_k)
		right_v = right_v.view_as(left_k)
		query = query.view_as(left_k)

		if self.lesion_scores:
			query = right_k
			scores = torch.einsum("bxahd,bxyhd->bxayh",left_k,query) / math.sqrt(self.d_k)
		else:
			scores = torch.einsum("bxahd,bayhd->bxayh",left_k,right_k) / math.sqrt(self.d_k)
		if mask is not None:
			scores = scores.masked_fill(mask.unsqueeze(4), -1e3)


		val = torch.einsum("bxahd,bayhd->bxayhd",left_v,right_v)

		att = F.softmax(scores,dim=2)
		att = self.dropout(att)

		if self.lesion_values:
			x = torch.einsum("bxayh,bxahd->bxyhd",att,left_v)
			x=x.contiguous()
			x = x.view(num_batches,num_nodes,num_nodes,self.d_model)
		else:
			x = torch.einsum("bxayh,bxayhd->bxyhd",att,val)
			x = x.view(num_batches,num_nodes,num_nodes,self.d_model)


		
		
		return self.linears[-1](x)

class EdgeTransformerLayer(nn.Module):

	def __init__(self,model_config, activation="relu"):
		super().__init__()

		self.model_config = model_config

		self.num_heads = self.model_config['num_heads']

		dropout = self.model_config['dropout']
		

		d_model = self.model_config['dim']

		self.edge_attention_flat = self.model_config['edge_attention_flat']

		if self.edge_attention_flat:
			self.edge_attention = EdgeAttentionFlat(d_model,self.num_heads, dropout,model_config)
		else:
			self.edge_attention = EdgeAttention(d_model,self.num_heads, dropout,model_config)


		self.linear1 = nn.Linear(d_model, self.model_config['ff_factor']*d_model)
		self.linear2 = nn.Linear(self.model_config['ff_factor']*d_model, d_model)
		
		self.norm1 = LayerNorm(d_model)
		self.norm2 = LayerNorm(d_model)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)

		self.activation = _get_activation_fn(activation)

		self.layernorm_pos = 'pre'



	def forward(self, batched_graphs, mask=None):

		if self.layernorm_pos == 'post':
			batched_graphs2 = self.edge_attention(batched_graphs,batched_graphs,batched_graphs,mask=mask)
			batched_graphs = batched_graphs + self.dropout1(batched_graphs2)
			batched_graphs = self.norm1(batched_graphs)
			batched_graphs2 = self.linear2(self.dropout2(self.activation(self.linear1(batched_graphs))))
			batched_graphs = batched_graphs + self.dropout3(batched_graphs2)
			batched_graphs = self.norm2(batched_graphs)
		elif self.layernorm_pos == 'pre':
			batched_graphs = self.norm1(batched_graphs)
			batched_graphs2 = self.edge_attention(batched_graphs,batched_graphs,batched_graphs,mask=mask)
			batched_graphs = batched_graphs + self.dropout1(batched_graphs2)
			batched_graphs = self.norm2(batched_graphs)
			batched_graphs2 = self.linear2(self.dropout2(self.activation(self.linear1(batched_graphs))))
			batched_graphs = batched_graphs + self.dropout3(batched_graphs2)
			

		return batched_graphs


class EdgeTransformerEncoder(nn.Module):

	def __init__(self,model_config,input_size,activation="relu"):
		super().__init__()

		
		self.model_config = model_config

		self.num_layers = self.model_config['num_message_rounds']

		self.share_layers = self.model_config['share_layers']


		self.left_linear = nn.Linear(self.model_config['dim'],self.model_config['dim'])
		self.right_linear = nn.Linear(self.model_config['dim'],self.model_config['dim'])


		encoder_layer = EdgeTransformerLayer(self.model_config)
		self.layers = _get_clones(encoder_layer, self.num_layers)

		self.pe = PositionalEncoding(input_size,max_len=200)
		self.spatial_relations_builder = SpatialRelationsBuilder(input_size)

		self._reset_parameters()


	def forward(self, edge_inputs, deprel_embeddings, mask=None):

		

		if self.model_config['input_encoding']=='positional':
			edge_inputs = self.pe(edge_inputs)
			left_inputs = self.left_linear(edge_inputs).unsqueeze(2)
			right_inputs = self.right_linear(edge_inputs).unsqueeze(1)
			batch = left_inputs + right_inputs
		elif self.model_config['input_encoding'] == 'relational':

			num_nodes = edge_inputs.size(1)
			loop_matrix = torch.diag(torch.ones(num_nodes).type_as(edge_inputs))
			batch = torch.einsum("bxd,xy->bxyd", edge_inputs, loop_matrix)

			batch += self.spatial_relations_builder(num_nodes)


		if mask is not None:
			new_mask = mask.unsqueeze(2)+mask.unsqueeze(1)
			new_mask = new_mask.unsqueeze(3)+mask.unsqueeze(1).unsqueeze(2)
			

			mask = new_mask



		for mod in self.layers:
			batch = mod(batch, mask=mask)
			
		return batch

	def _reset_parameters(self):

		for p in self.parameters():
			if p.dim() > 1:
				torch.nn.init.xavier_uniform_(p)

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=20):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:,:x.size(1), :]
		return self.dropout(x)


class SpatialRelationsBuilder(nn.Module):

	def __init__(self, dim):
		super().__init__()
		self.max_len = 150
		self.max_rel_len = 16
		self.src_to_tgt_rel = 2 * self.max_rel_len + 1
		self.tgt_to_src_rel = 2 * self.max_rel_len + 2
		self.num_rels_total = 2 * self.max_rel_len + 3
		self.rel_embeddings = nn.Embedding(self.num_rels_total, dim)

		relations = torch.zeros((self.max_len, self.max_len)).long()
		for i in range(self.max_len):
			for j in range(self.max_len):
				relations[i, j] = self.max_rel_len + min(max(j - i, -self.max_rel_len), self.max_rel_len)
		self.register_buffer("relations", relations)
		assert(torch.all(self.relations < 2 * self.max_rel_len + 1))

	def forward(self, src_len):
		return self.rel_embeddings(self.relations[:src_len, :src_len])

class UniversalTransformerEncoder(nn.Module):
	r"""TransformerEncoder is a stack of N encoder layers
	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).
	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		>>> src = torch.rand(10, 32, 512)
		>>> out = transformer_encoder(src)
	"""
	__constants__ = ['norm']

	def __init__(self, encoder_layer, num_layers, norm=None):
		super(UniversalTransformerEncoder, self).__init__()
		self.layer = encoder_layer
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, src, mask=None, src_key_padding_mask=None):
		r"""Pass the input through the encoder layers in turn.
		Args:
			src: the sequence to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""
		output = src

		for i in range(self.num_layers):
			output = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

		if self.norm is not None:
			output = self.norm(output)

		return output

class TransformerBaseline(nn.Module):
	def __init__(self,model_config):
		super().__init__()

		self.model_config = model_config
		self.universal = model_config['share_layers']
		
		encoder_layer = nn.TransformerEncoderLayer(d_model=model_config['dim'], dim_feedforward=model_config['ff_factor']*model_config['dim'], nhead=model_config['num_heads'],dropout=model_config['dropout'])
		
		norm = LayerNorm(model_config['dim'])

		if self.universal:
			self.transformer_encoder = UniversalTransformerEncoder(encoder_layer,model_config['num_message_rounds'],norm)
		else:
			self.transformer_encoder = nn.TransformerEncoder(encoder_layer,model_config['num_message_rounds'],norm)
		
		self.pe = PositionalEncoding(model_config['dim'],max_len=200)


		

		self.decoder_q = nn.Linear(self.model_config['dim'], self.model_config['dim'])
		self.decoder_k = nn.Linear(self.model_config['dim'], self.model_config['dim'])


	def forward(self, encoder_inputs, deprel_embeddings=None, mask=None):
		encoder_inputs = self.pe(encoder_inputs)
		encoder_inputs = torch.transpose(encoder_inputs,0,1) #batch x length -> length x batch
		encoder_out = self.transformer_encoder(encoder_inputs,src_key_padding_mask=mask)
		encoder_out = torch.transpose(encoder_out,0,1)

		q = self.decoder_q(encoder_out)
		k = self.decoder_k(encoder_out)

		pair_logits = torch.bmm(q,k.transpose(1,2))
		
		return pair_logits,encoder_out


class T5(nn.Module):
	def __init__(self,model_config):
		super().__init__()

		self.model_config = model_config
		self.universal = model_config['share_layers']

		tf_config = T5Config(vocab_size=self.model_config['token_vocab_size'],d_model=self.model_config['dim'],
				d_ff=self.model_config['ff_factor']*self.model_config['dim'],num_layers=self.model_config['num_message_rounds'],
				num_heads = self.model_config['num_heads'],dropout_rate=self.model_config['dropout'])
		encoder = T5EncoderModel(tf_config)

		if self.universal: 
			print("WARNING: use the hacky method for sharing parameters across layers")
			def reuse_first_block(model):
				for i in range(1, len(model.encoder.block)):
					model.encoder.block[i] = model.encoder.block[0]        
			reuse_first_block(encoder)

		self.encoder = encoder

		

		self.decoder_q = nn.Linear(self.model_config['dim'], self.model_config['dim'])
		self.decoder_k = nn.Linear(self.model_config['dim'], self.model_config['dim'])

		self._reset_parameters()

	def forward(self, input_ids, mask):
		
		encoder_out = self.encoder(input_ids=input_ids,attention_mask=1-mask.long())
		encoder_out = encoder_out.last_hidden_state


		q = self.decoder_q(encoder_out)
		k = self.decoder_k(encoder_out)

		pair_logits = torch.bmm(q,k.transpose(1,2))
		
		return pair_logits,encoder_out

	def _reset_parameters(self):

		for p in self.parameters():
			if p.dim() > 1:
				torch.nn.init.xavier_uniform_(p)


class Parser(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.save_hyperparameters()
		self._create_model(args)

	def _create_model(self,args):

		self.args = vars(args)

		self.word_emb = nn.Embedding(self.args['token_vocab_size'], self.args['dim'], padding_idx=1)
		
		if self.args['model_type']=='edge_transformer':
			self.encoder = EdgeTransformerEncoder(self.args,self.args['dim'])
		elif self.args['model_type']=='T5':
			self.encoder = T5(self.args)
		elif self.args['model_type']=='transformer':
			self.encoder = TransformerBaseline(self.args)
		

		# classifiers
		self.decode_parent = nn.Linear(self.args['dim'],1)
		self.decode_role = nn.Linear(self.args['dim'],self.args['role_vocab_size'])
		self.decode_category = nn.Linear(self.args['dim'],self.args['category_vocab_size'])
		self.decode_noun_type = nn.Linear(self.args['dim'],self.args['noun_type_vocab_size'])
		self.decode_verb_name = nn.Linear(self.args['dim'],self.args['verb_name_vocab_size'])

		# criterion
		self.pad_index = 0
		self.parent_pad_index = 99999
		self.parent_crit = nn.CrossEntropyLoss(ignore_index=self.parent_pad_index, reduction='mean') # ignore padding
		self.node_crit = nn.CrossEntropyLoss(ignore_index=self.pad_index, reduction='mean') # ignore padding

		self.drop = nn.Dropout(self.args['dropout'])

	def on_post_move_to_device(self):
		if self.args['share_layers']:
			for mod in self.encoder.layers:
				mod.weight = self.encoder.layers[0].weight


	def configure_optimizers(self):
		# We will support Adam or AdamW as optimizers.
		if self.args['optimizer']=="AdamW":
			opt = AdamW
		elif self.args['optimizer']=="Adam":
			opt = Adam
		optimizer = opt(self.parameters(), **self.args['optimizer_args'])
		
		# We will reduce the learning rate by 0.1 after 100 and 150 epochs
		if self.args['scheduler']=='linear_warmup':
			scheduler = get_linear_schedule_with_warmup(optimizer,**self.args['scheduler_args'])
		elif self.args['scheduler']=='tf':
			scheduler = get_tf_schedule(optimizer)
		

		return {'optimizer':optimizer, 'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}


	def _calculate_loss(self, batch):
		tokens = batch['tokens'] 
		parent = batch['parent']
		role = batch['role']
		category = batch['category'] 
		noun_type = batch['noun_type']
		verb = batch['verb'] 
		mask = batch['mask'] 


		if self.args['model_type']=='edge_transformer':
			encoder_inputs = self.word_emb(tokens)
			outputs = self.encoder(encoder_inputs,self.decode_parent,mask=mask)
			

			deprel_scores = self.decode_parent(outputs).squeeze(-1) #batch_size x sent_len x sent_len

			node_output_vectors = []
			for i in range(outputs.shape[1]):
				node_output_vectors.append(outputs[:, i, i, :])
			node_outputs = torch.stack(node_output_vectors, dim=1)

		elif self.args['model_type']=='T5':
			deprel_scores, node_outputs = self.encoder(tokens,mask)
		elif self.args['model_type']=='transformer':
			encoder_inputs = self.word_emb(tokens)
			deprel_scores, node_outputs = self.encoder(encoder_inputs,mask)


		deprel_scores = deprel_scores.masked_fill(mask.unsqueeze(2)+mask.unsqueeze(1),-1e3)
		deprel_loss = self.parent_crit(deprel_scores.view(-1,deprel_scores.size(2)),parent.view(-1))



		


		role_logits = self.decode_role(node_outputs)
		category_logits = self.decode_category(node_outputs)
		noun_type_logits = self.decode_noun_type(node_outputs)
		verb_logits = self.decode_verb_name(node_outputs)

		role_loss = self.node_crit(role_logits.view(-1,role_logits.size(-1)),role.view(-1))
		category_loss = self.node_crit(category_logits.view(-1,category_logits.size(-1)),category.view(-1))
		noun_type_loss = self.node_crit(noun_type_logits.view(-1,noun_type_logits.size(-1)),noun_type.view(-1))
		verb_loss = self.node_crit(verb_logits.view(-1,verb_logits.size(-1)),verb.view(-1))

		loss = deprel_loss + role_loss + category_loss + noun_type_loss + verb_loss


		return loss, (deprel_scores,role_logits,category_logits,noun_type_logits,verb_logits)

	def training_step(self,batch,batch_idx):
		loss, _ = self._calculate_loss(batch)

		scheduler = self.lr_schedulers()

		return loss

	def compute_acc(self,batch,scores):
		(deprel_scores,role_logits,category_logits,noun_type_logits,verb_logits) = scores

		edge_preds = deprel_scores.max(2)[1]
		role_preds = role_logits.max(-1)[1]
		category_preds = category_logits.max(-1)[1]
		noun_type_preds = noun_type_logits.max(-1)[1]
		verb_preds = verb_logits.max(-1)[1]

		parent = batch['parent']
		role = batch['role']
		category = batch['category'] 
		noun_type = batch['noun_type']
		verb = batch['verb'] 
		mask = batch['mask'] 


		correct = ((torch.eq(edge_preds,parent) & torch.eq(role_preds,role) & torch.eq(category_preds,category) \
			& torch.eq(noun_type_preds,noun_type) & torch.eq(verb_preds,verb)) | mask).detach()


		acc = correct.all(1).sum(0)/correct.size(0)

		return acc

	def validation_step(self,batch,batch_idx):
		loss, (deprel_scores,role_logits,category_logits,noun_type_logits,verb_logits) = self._calculate_loss(batch)


		acc = self.compute_acc(batch,(deprel_scores,role_logits,category_logits,noun_type_logits,verb_logits))

		self.log("val_loss",loss,prog_bar=True)
		self.log("val_acc",acc,prog_bar=True)

	def test_step(self,batch,batch_idx):
		loss, (deprel_scores,role_logits,category_logits,noun_type_logits,verb_logits) = self._calculate_loss(batch)


		acc = self.compute_acc(batch,(deprel_scores,role_logits,category_logits,noun_type_logits,verb_logits))

		self.log("test_loss",loss,prog_bar=True)
		self.log("test_acc",acc,prog_bar=True)
