import copy
import math
import torch
from torch import nn
from torch.nn import Parameter, ModuleList, LayerNorm, Dropout
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

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

#from the original Sinha et al. CLUTRR code
def get_mlp(input_dim, output_dim, num_layers=2, dropout=0.0):
    network_list = []
    assert num_layers > 0
    if num_layers > 1:
        for _ in range(num_layers-1):
            network_list.append(nn.Linear(input_dim, input_dim))
            network_list.append(nn.ReLU())
            network_list.append(nn.Dropout(dropout))
    network_list.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        *network_list
    )

class EdgeAttentionFlat(nn.Module):
    def __init__(self,d_model, num_heads, dropout,model_config):
        super(EdgeAttentionFlat, self).__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model,bias=False), 4)
        
        self.dropout = nn.Dropout(p=dropout)

        
    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes = query.size(1)
        

        k,v,q = [l(x) for l, x in zip(self.linears, (key, value, query))]
        k = k.view(num_batches,num_nodes,num_nodes,self.num_heads,self.d_k)
        v = v.view_as(k)
        q = q.view_as(k)

        scores_r = torch.einsum("bxyhd,bxzhd->bxyzh",q,k) / math.sqrt(self.d_k)
        scores_r = scores_r.masked_fill(mask.unsqueeze(4), -1e9)
        scores_l = torch.einsum("bxyhd,bzyhd->bxyzh",q,k) / math.sqrt(self.d_k)
        scores_l = scores_l.masked_fill(mask.unsqueeze(4), -1e9)
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
        super(EdgeAttention, self).__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model,bias=False), 6)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.lesion_scores = model_config.lesion_scores
        self.lesion_values = model_config.lesion_values
        
    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes = query.size(1)
        

        left_k, right_k, left_v, right_v, query = [l(x) for l, x in zip(self.linears, (key, key, value, value,key))]
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
        scores = scores.masked_fill(mask.unsqueeze(4), -1e9)

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

    def __init__(self,model_config ,activation="relu"):
        super().__init__()

        self.num_heads = model_config.num_heads

        dropout = model_config.dropout
        

        d_model = model_config.dim
        d_ff = model_config.ff_factor * d_model


        self.flat_attention = model_config.flat_attention

        if self.flat_attention:
            self.edge_attention = EdgeAttentionFlat(d_model,self.num_heads, dropout,model_config)
        else:
            self.edge_attention = EdgeAttention(d_model,self.num_heads, dropout,model_config)


        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)




    def forward(self, batched_graphs, mask=None):

        batched_graphs = self.norm1(batched_graphs)
        batched_graphs2 = self.edge_attention(batched_graphs,batched_graphs,batched_graphs,mask=mask)
        batched_graphs = batched_graphs + self.dropout1(batched_graphs2)
        batched_graphs = self.norm2(batched_graphs)
        batched_graphs2 = self.linear2(self.dropout2(self.activation(self.linear1(batched_graphs))))
        batched_graphs = batched_graphs + self.dropout3(batched_graphs2)
            

        return batched_graphs


class EdgeTransformerEncoder(nn.Module):

    def __init__(self,model_config, shared_embeddings=None,activation="relu"):
        super().__init__()
        self.model_config = model_config

        self.num_heads = self.model_config.num_heads
        
        self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.edge_types+1,
                                                embedding_dim=self.model_config.dim)
        
        self.deep_residual = False

        self.share_layers = self.model_config.share_layers



        self.num_layers = self.model_config.num_message_rounds

        encoder_layer = EdgeTransformerLayer(self.model_config)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()


    def forward(self, batch):

        batched_graphs = batch['batched_graphs']
        batched_graphs = self.embedding(batched_graphs) #B x N x N x node_dim

        mask = batch['masks']

        

        
        if mask is not None:
            new_mask = mask.unsqueeze(2)+mask.unsqueeze(1)
            new_mask = new_mask.unsqueeze(3)+mask.unsqueeze(1).unsqueeze(2)
            

            mask = new_mask


        all_activations = [batched_graphs]

        if not self.share_layers:
            for mod in self.layers:
                batched_graphs = mod(batched_graphs, mask=mask)
        else:
            for i in range(self.model_config.num_message_rounds):
                batched_graphs = self.layers[0](batched_graphs,mask=mask)


        return batched_graphs

    def _reset_parameters(self):

        # for n,p in self.named_parameters():
        #     if ("linear" in n and "weight" in n) or ("embedding" in n):
        #         torch.nn.init.orthogonal_(p)
        #     else:
        #         if p.dim()>1:
        #             nn.init.xavier_uniform_(p)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)


class EdgeTransformer(pl.LightningModule):
    def __init__(self,model_config):
        super().__init__()

        self.save_hyperparameters()
        self._create_model(model_config)

    def _create_model(self,model_config):
        self.model_config = model_config

        self.encoder = EdgeTransformerEncoder(model_config)
        input_dim = model_config.dim
        self.decoder2vocab = get_mlp(
            input_dim,
            model_config.target_size
        )

        self.crit = nn.CrossEntropyLoss(reduction='mean')

    def configure_optimizers(self):
        # We will support Adam or AdamW as optimizers.
        if self.model_config.optimizer=="AdamW":
            opt = AdamW
        elif self.model_config.optimizer=="Adam":
            opt = Adam
        optimizer = opt(self.parameters(), **self.model_config.optimizer_args)
        

        if self.model_config.scheduler=='linear_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,**self.model_config.scheduler_args)
        
        

        return {'optimizer':optimizer, 'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}

        
        #return {'optimizer':optimizer}


    def _calculate_loss(self, batch):
        batched_graphs = self.encoder(batch)

        query_edges = batch['query_edges']

        logits = self.decoder2vocab(batched_graphs[query_edges[:,0],query_edges[:,1],query_edges[:,2]])


        loss = self.crit(logits,batch['query_labels'])

        


        return loss, logits

    def training_step(self,batch,batch_idx):
        loss, _ = self._calculate_loss(batch)

        scheduler = self.lr_schedulers()

        return loss

    def compute_acc(self,batch,scores):
        preds = scores.max(-1)[1]

        labels = batch['query_labels']



        acc = ((torch.eq(preds,labels).sum(0))/preds.size(0)).detach()

        return acc

    def validation_step(self,batch,batch_idx):
        loss, logits = self._calculate_loss(batch)


        acc = self.compute_acc(batch,logits)

        self.log("val_loss",loss,prog_bar=True)
        self.log("val_acc",acc,prog_bar=True)

    def test_step(self,batch,batch_idx):
        loss, logits = self._calculate_loss(batch)


        acc = self.compute_acc(batch,logits)

        self.log("test_loss",loss,prog_bar=True)
        self.log("test_acc",acc,prog_bar=True)



