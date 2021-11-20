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

class RelationAttention(nn.Module):
    def __init__(self,d_model, num_heads, dropout, model_config):
        super().__init__()
        # We assume d_v always equals d_k

        self.update_relations = model_config.update_relations

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)

        self.left_linear = nn.Linear(d_model,d_model)
        self.right_linear = nn.Linear(d_model,d_model)

        self.relation_key_map = nn.Linear(self.d_model, self.d_model)
        self.relation_value_map = nn.Linear(self.d_model, self.d_model)

        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


        
    def forward(self, x, graphs_k=None, graphs_v=None, batched_graphs=None, mask=None):
        
        if self.update_relations:
            graphs_k = self.relation_key_map(batched_graphs)
            graphs_v = self.relation_value_map(batched_graphs)


        num_batches = x.size(0)
        num_nodes = x.size(1)
        device = x.device
        
        query, key, value = \
            [l(x) for l, x in zip(self.linears, (x, x, x))]

        query = query.view(num_batches,num_nodes,self.num_heads,self.d_k)
        key = key.view_as(query)
        value = value.view_as(query)

        graphs_k = graphs_k.view(num_batches,num_nodes,num_nodes,self.num_heads,self.d_k)
        graphs_v = graphs_v.view_as(graphs_k)

        scores_k = torch.einsum("bxhd,byhd->bxyh",query,key)
        scores_graph = torch.einsum("bxhd,bxyhd->bxyh",query,graphs_k)
        scores = scores_k+scores_graph
        scores = scores / math.sqrt(self.d_k)

        scores = scores.masked_fill(mask.unsqueeze(3), -1e9)
        p_attn = F.softmax(scores, dim=2)
        p_attn = self.dropout(p_attn)

        x_v = torch.einsum("bxyh,byhd->bxhd",p_attn,value)
        x_graph = torch.einsum("bxyh,bxyhd->bxhd",p_attn,graphs_v)

        x = x_v+x_graph
        x = x.contiguous().view(num_batches,num_nodes,self.d_model)

        if self.update_relations:

            x_left = self.left_linear(x).unsqueeze(2)
            x_right = self.right_linear(x).unsqueeze(1)
            batched_graphs = x_left+x_right


        

        
        return self.linears[-1](x), batched_graphs

class RelationTransformerLayer(nn.Module):

    def __init__(self,model_config ,activation="relu"):
        super().__init__()
        self.model_config = model_config
        self.update_relations = model_config.update_relations
        self.num_heads = self.model_config.num_heads

        dropout = self.model_config.dropout
        

        d_model = self.model_config.dim
        d_ff = self.model_config.ff_factor*d_model
        
        self.rel_attention = RelationAttention(d_model,self.num_heads, dropout, model_config)


        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.linear_graph1 = nn.Linear(d_model, d_ff)
        self.linear_graph2 = nn.Linear(d_ff, d_model)
        
        self.norm_graph1 = LayerNorm(d_model)
        self.norm_graph2 = LayerNorm(d_model)
        self.dropout_graph1 = Dropout(dropout)
        self.dropout_graph2 = Dropout(dropout)
        self.dropout_graph3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.layernorm_pos = 'post'



    def forward(self, x, graphs_k=None, graphs_v=None, batched_graphs=None,mask=None):
        

        x_2, batched_graphs_2 = self.rel_attention(x,graphs_k=graphs_k, graphs_v=graphs_v, batched_graphs=batched_graphs,mask=mask)
        x = x + self.dropout1(x_2)
        x = self.norm1(x)
        x_2 = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + self.dropout3(x_2)
        x = self.norm2(x)

        if self.update_relations:
            batched_graphs = batched_graphs + self.dropout_graph1(batched_graphs_2)
            batched_graphs = self.norm_graph1(batched_graphs)
            batched_graphs2 = self.linear_graph2(self.dropout_graph2(self.activation(self.linear_graph1(batched_graphs))))
            batched_graphs = batched_graphs + self.dropout_graph3(batched_graphs2)
            batched_graphs = self.norm_graph2(batched_graphs)
            
                

        return x, batched_graphs


class RelationTransformerEncoder(nn.Module):

    def __init__(self,model_config, shared_embeddings=None,activation="relu"):
        super().__init__()
        self.model_config = model_config
        self.update_relations = model_config.update_relations
        self.zero_init = model_config.zero_init

        self.num_heads = self.model_config.num_heads
        
        self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.edge_types+1,
                                                embedding_dim=self.model_config.dim)



        self.share_layers = self.model_config.share_layers

        self.num_layers = self.model_config.num_message_rounds

        self.d_model = self.model_config.dim

        self.relation_key_map = nn.Linear(self.d_model, self.d_model)
        self.relation_value_map = nn.Linear(self.d_model, self.d_model)


        encoder_layer = RelationTransformerLayer(self.model_config)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()


    def forward(self, batch):


        batched_graphs = batch['batched_graphs']
        batched_graphs = self.embedding(batched_graphs) #B x N x N x node_dim

        if not self.update_relations:
            #this is the ordinary relation transformer
            #we do not update relation keys/values at each layer
            graphs_k = self.relation_key_map(batched_graphs)
            graphs_v = self.relation_value_map(batched_graphs)
        else:
            graphs_k = None
            graphs_v = None


        batch_size = batched_graphs.size(0)
        num_nodes = batched_graphs.size(1)
        if not self.zero_init:
            x = torch.rand((batch_size,num_nodes,self.d_model)).to(batched_graphs.device)
        else:
            x = torch.zeros((batch_size,num_nodes,self.d_model)).to(batched_graphs.device)


        mask = batch['masks']
        mask = mask.unsqueeze(2)+mask.unsqueeze(1)

        for i in range(self.num_layers):
            if self.share_layers:
                x, batched_graphs = self.layers[0](x, graphs_k=graphs_k, graphs_v=graphs_v, batched_graphs=batched_graphs, mask=mask)
            else:
                x, batched_graphs = self.layers[i](x, graphs_k=graphs_k, graphs_v=graphs_v, batched_graphs=batched_graphs, mask=mask)
        
        if self.update_relations:
            return batched_graphs
        else:
            return x

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)



class RAT(pl.LightningModule):
    def __init__(self,model_config):
        super().__init__()

        

        self.save_hyperparameters()
        self._create_model(model_config)

    def _create_model(self,model_config):
        self.model_config = model_config
        self.update_relations = model_config.update_relations

        self.encoder = RelationTransformerEncoder(model_config)
        input_dim = model_config.dim

        if self.update_relations:
            self.decoder2vocab = get_mlp(
                input_dim,
                model_config.target_size
            )
        else:
            self.decoder2vocab = get_mlp(
                2*input_dim,
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

        if not self.update_relations:
            result_l = batched_graphs[query_edges[:,0],query_edges[:,1]]
            result_r = batched_graphs[query_edges[:,0],query_edges[:,2]]


            scores = torch.cat((result_l,result_r),dim=-1)
        else:

            scores = batched_graphs[query_edges[:,0],query_edges[:,1],query_edges[:,2]]

        logits = self.decoder2vocab(scores)

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