import time
import copy
from typing import Optional
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch._C import DeviceObjType
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, LayerNorm, Dropout, CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.generation_utils import GenerationMixin
from transformers.configuration_utils import PretrainedConfig

# MODELLING

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class EdgeAttentionEncoder(nn.Module):
    def __init__(self,d_model, num_heads, dropout):
        super().__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model), 5)
        self.reduce_dim_key = nn.Linear(2*d_model,d_model)
        self.reduce_dim_value = nn.Linear(2*d_model,d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, state, mask=None):
        query, key, value = state, state, state
        # mask signature: bxay
        num_batches = query.size(0)
        num_nodes = query.size(1)
        device = query.device
       
        left_k, right_k, left_v, right_v = [l(x) for l, x in zip(self.linears, (key, key, value, value))]
        left_k = left_k.view(num_batches, num_nodes, num_nodes, self.num_heads, self.d_k)
        right_k = right_k.view_as(left_k)
        left_v = left_v.view_as(left_k)
        right_v = right_v.view_as(left_k)

        scores = torch.einsum("bxahd,bayhd->bxayh",left_k,right_k) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(4), -1e3)

        val = torch.einsum("bxahd,bayhd->bxayhd",left_v,right_v)

        att = F.softmax(scores,dim=2)
        att = self.dropout(att)

        x = torch.einsum("bxayh,bxayhd->bxyhd",att,val)
        x = x.view(num_batches,num_nodes,num_nodes,self.d_model)

        
        return self.linears[-1](x)


class EdgeTransformerLayer(nn.Module):

    def __init__(self,model_config, activation="relu"):
        super().__init__()

        self.model_config = model_config

        self.num_heads = self.model_config.num_heads

        dropout = self.model_config.dropout
        
        d_model = self.model_config.dim
        d_ff = self.model_config.expand_ff * self.model_config.dim

        self.edge_attention = EdgeAttentionEncoder(d_model,self.num_heads, dropout)

        print(d_ff)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, batched_graphs, src_len=None, mask=None):
        batched_graphs = self.norm1(batched_graphs)
        batched_graphs2 = self.edge_attention(batched_graphs,mask=mask)
        batched_graphs = batched_graphs + self.dropout1(batched_graphs2)
        batched_graphs = self.norm2(batched_graphs)
        batched_graphs2 = self.linear2(self.dropout2(self.activation(self.linear1(batched_graphs))))
        batched_graphs = batched_graphs + self.dropout3(batched_graphs2)
        return batched_graphs


class SpatialRelationsBuilder(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.max_len = 150
        self.max_rel_len = 32
        self.src_to_tgt_rel = 2 * self.max_rel_len + 1
        self.tgt_to_src_rel = 2 * self.max_rel_len + 2
        self.num_rels_total = 2 * self.max_rel_len + 3
        self.rel_embeddings = nn.Embedding(self.num_rels_total, dim)

        self.relations = torch.zeros((self.max_len, self.max_len), dtype=torch.int64)
        for i in range(self.max_len):
            for j in range(self.max_len):
                self.relations[i, j] = self.max_rel_len + min(max(j - i, -self.max_rel_len), self.max_rel_len)
        assert(torch.all(self.relations < 2 * self.max_rel_len + 1))

    def forward(self, src_len, device, tgt_len=None):
        if tgt_len is None:
            return self.rel_embeddings(self.relations[:src_len, :src_len].to(device))
        
        relations = self.relations[:src_len + tgt_len, :src_len + tgt_len].clone().to(device)
        relations[:src_len, src_len:] = self.src_to_tgt_rel
        relations[src_len:, :src_len] = self.tgt_to_src_rel
        return self.rel_embeddings(relations)


class EdgeTransformerEncoder(nn.Module):

    def __init__(self,model_config,activation="relu"):
        super().__init__()

        
        self.model_config = model_config

        self.num_layers = self.model_config.num_message_rounds

        self.share_layers = self.model_config.share_layers


        self.left_linear = nn.Linear(self.model_config.dim, self.model_config.dim)
        self.right_linear = nn.Linear(self.model_config.dim, self.model_config.dim)

        self.bos_vector = nn.Parameter(torch.zeros(self.model_config.dim))
        self.spatial_relations_builder = SpatialRelationsBuilder(self.model_config.dim)



        encoder_layer = EdgeTransformerLayer(self.model_config)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()


    def forward(self, edge_inputs, mask=None):
        if self.model_config.input_encoding == 'pairwise-sum':
            left_inputs = self.left_linear(edge_inputs).unsqueeze(2)
            right_inputs = self.right_linear(edge_inputs).unsqueeze(1)
            batch = left_inputs + right_inputs
        elif self.model_config.input_encoding == 'relational':
            edge_inputs[:, 0] += self.bos_vector

            num_nodes = edge_inputs.size(1)
            loop_matrix = torch.diag(torch.ones(num_nodes).to(edge_inputs.device))
            batch = torch.einsum("bxd,xy->bxyd", edge_inputs, loop_matrix)

            batch += self.spatial_relations_builder(num_nodes, edge_inputs.device)
        else: 
            raise ValueError()

        if mask is not None:
            # mask: bx
            new_mask = mask.unsqueeze(2)+mask.unsqueeze(1)
            # new_mask: bxa
            new_mask = new_mask.unsqueeze(3)+mask.unsqueeze(1).unsqueeze(2)
            # new_mask: bxay
            mask = new_mask

        if not self.share_layers:
            for mod in self.layers:
                batch = mod(batch, mask=mask)
        else:
            for i in range(self.num_layers):
                batch = self.layers[0](batch, mask=mask)
            
        return batch

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

class EdgeTransformerDecoder(nn.Module):

    def __init__(self,model_config,activation="relu"):
        super().__init__()
       
        self.model_config = model_config

        self.num_layers = self.model_config.num_message_rounds

        self.share_layers = self.model_config.share_layers

        self.left_linear = nn.Linear(self.model_config.dim, self.model_config.dim)
        self.right_linear = nn.Linear(self.model_config.dim, self.model_config.dim)

        self.bos_vector = nn.Parameter(torch.zeros(self.model_config.dim))
        self.spatial_relations_builder = SpatialRelationsBuilder(self.model_config.dim)


        encoder_layer = EdgeTransformerLayer(self.model_config)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()


    def forward(self, src, src_mask, src_encoded, tgt):
        num_examples = src.shape[0]
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]
        edge_inputs = torch.cat([src, tgt], 1)
        num_nodes = src_len + tgt_len

        if self.model_config.input_encoding == 'pairwise-sum':
            left_inputs = self.left_linear(edge_inputs).unsqueeze(2)
            right_inputs = self.right_linear(edge_inputs).unsqueeze(1)
            batch = left_inputs + right_inputs
        elif self.model_config.input_encoding == 'relational':
            edge_inputs[:, src_len] += self.bos_vector

            loop_matrix = torch.diag(torch.ones(num_nodes).to(edge_inputs.device))
            batch = torch.einsum("bxd,xy->bxyd", edge_inputs, loop_matrix)
            
            batch += self.spatial_relations_builder(src_len, edge_inputs.device, tgt_len)
        else:
            raise ValueError()


        # TODO: will this back-prop?
        batch[:, :src_len, :src_len, :] = src_encoded

        new_causal_mask = torch.zeros((num_nodes, num_nodes, num_nodes), dtype=torch.bool)
        for a in range(src_len, num_nodes):
            new_causal_mask[:a, a, :a] = True
        causal_mask = new_causal_mask.to(batch.device).unsqueeze(0)
        
        # mask src padding
        joint_src_mask = torch.zeros((num_examples, num_nodes), dtype=torch.bool).to(batch.device)
        joint_src_mask[:, :src_len] = src_mask
        tmp_mask = joint_src_mask.unsqueeze(2) + joint_src_mask.unsqueeze(1)
        joint_src_mask = tmp_mask.unsqueeze(3) + joint_src_mask.unsqueeze(1).unsqueeze(2)
        mask = causal_mask | joint_src_mask

        if not self.share_layers:
            for mod in self.layers:
                batch = mod(batch, mask=mask)
        else:
            for i in range(self.num_layers):
                batch = self.layers[0](batch, mask=mask)
            
        return batch

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

# INTEGRATION WITH TRANSFORMERS

class EdgeConfig(PretrainedConfig):
    #TODO: inherit from PretrainedModelConfig? But it is nice to have this as 
    # a dataclass..
 
    def __init__(
        self,
        vocab_size : int,
        dim : int,
        expand_ff : int,
        num_heads : int,
        num_message_rounds : int,
        share_layers : bool,
        dropout : float,
        decode_from : str,
        input_encoding : str,
        **kwargs
    ):
        kwargs['is_encoder_decoder'] = True
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.expand_ff = expand_ff
        self.num_heads = num_heads
        self.num_message_rounds = num_message_rounds
        self.share_layers = share_layers
        self.dropout = dropout
        self.decode_from = decode_from
        self.input_encoding = input_encoding


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=300):
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


@dataclass
class EdgeTransformerEncoderOutput(BaseModelOutput):
    input_ids : torch.LongTensor = None


class EdgeTransfomerForConditionalGeneration(nn.Module, GenerationMixin):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.emb = nn.Embedding(
            config.vocab_size, 
            config.dim, 
            padding_idx=config.pad_token_id)
        if config.input_encoding == 'relational':
            self.pe = lambda x: x
        elif config.input_encoding == 'pairwise-sum':
            self.pe = PositionalEncoding(config.dim, config.dropout)
        else:
            raise ValueError()
            
        self.encoder = EdgeTransformerEncoder(config)
        self.decoder = EdgeTransformerDecoder(config)
        # TODO: why bias=False
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def _shift_right(self, input_ids):
        # largely copy-pasted from T5ForConditionalGeneration
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), ("self.model.config.decoder_start_token_id has to be defined."
            "In T5 it is usually set to the pad_token_id. See T5 docs for more information")

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def get_encoder(self):
        # for compatibility with GenerationMixin
        def _encode(input_ids, attention_mask, return_dict=True, output_attentions=False, output_hidden_states=False):
            assert return_dict == True and output_attentions == False and output_hidden_states == False
            input_embs = self.pe(self.emb(input_ids))
            return EdgeTransformerEncoderOutput(
                last_hidden_state=self.encoder(input_embs, mask=~attention_mask.bool()), 
                input_ids=input_ids
            )
        return _encode

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs
    ):
        return {
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }        

    def forward(
        self,
        attention_mask,
        input_ids=None,
        labels=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False
    ):
        """
        Note: attention mask here identifies tokens that *should* be attended.
        Whereas in the edge transformer code the mask identifies triangles that should be masked out.

        Note: almost all arguments are optional because this method is used for both
        the loss computation and generation. The only argument that is always provided
        is `attention_mask`.

        """
        if input_ids is not None:
            assert encoder_outputs is None
            assert labels is not None
            assert decoder_input_ids is None
            loss_computation_mode = True
        if encoder_outputs is not None:
            assert labels is None
            assert decoder_input_ids is not None
            loss_computation_mode = False
        assert return_dict and not output_attentions and not output_hidden_states
        src_len = attention_mask.shape[1]

        # encode (if needed)
        if loss_computation_mode:
            input_embs = self.pe(self.emb(input_ids))
            input_encoding = self.encoder(input_embs, mask=~attention_mask.bool())
            decoder_input_ids = self._shift_right(labels)
        else:
            input_embs = self.pe(self.emb(encoder_outputs['input_ids']))
            input_encoding = encoder_outputs['last_hidden_state']        
        tgt_len = decoder_input_ids.shape[1]
        decoder_input_embs = self.pe(self.emb(decoder_input_ids))

        # run the main decoder model
        decoder_encoding = self.decoder(input_embs, ~attention_mask.bool(), input_encoding, decoder_input_embs)
        # TODO: just try the edge connecting 0 and this one?
        output_vectors = []
        for i in range(tgt_len):
            if self.config.decode_from == 'start_to_i':
                output_vectors.append(decoder_encoding[:, src_len, src_len + i, :])
            elif self.config.decode_from == 'i_to_i':
                output_vectors.append(decoder_encoding[:, src_len + i, src_len + i, :])
            else:
                raise ValueError()
            
        output = torch.stack(output_vectors, dim=1)
        lm_logits = self.lm_head(output)

        loss = None
        if loss_computation_mode:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            decoder_hidden_states=decoder_encoding
        )