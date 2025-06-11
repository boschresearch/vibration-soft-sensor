# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:38:17 2023

@author: WET2RNG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.models.forecasting.tft_submodels import _GatedResidualNetwork

from softsensor.recurrent_models import _parallel_RNN, _pred_lstm, _filter_parameters

class TFT(nn.Module):
    def __init__(self, input_channels, pred_size, blocks, hidden_window=1,
                 num_layers=1, blocktype='LSTM', n_heads=4, bias=True, dropout=None,
                 forecast=1, GRN=True, Pred_Type='Point'):
        super().__init__()
        self.params = _filter_parameters(locals().copy())
        self.input_channels = input_channels
        self.GRN = GRN
        self.forecast = forecast
        self.pred_size = pred_size
        self.Type = 'RNN'
        self.Pred_Type = Pred_Type
        self.Ensemble = False
        self.window_size = 1
        self.precomp = True
        self.rnn_window = None
        
        if self.Pred_Type == 'Point': 
            preds = 1
        elif self.Pred_Type == 'Mean_Var':
            preds = 2
        else:
            print('No valid Pred_Type given')
        
        self.RecBlock = _parallel_RNN(input_channels, pred_size, blocks, hidden_window,
                                      num_layers, blocktype, bias, dropout, forecast)
        
        if self.GRN:
            self.GRN1 = _GatedResidualNetwork(blocks, blocks, blocks, dropout=dropout)
        self.multiAtt = InterpretableMultiHeadAttention(n_head=n_heads,
                                                        d_model=blocks)
        if self.GRN:
            self.GRN2 = _GatedResidualNetwork(blocks, blocks, blocks, dropout=dropout)

        self.linear = nn.Linear(blocks, pred_size*forecast*preds,
                                bias=bias)


    def forward(self, inp, device='cpu'):
        inp = self.RecBlock(inp, device)
        if self.GRN:
            inp = self.GRN1(inp)
        inp, atts = self.multiAtt(inp, inp, inp)
        if self.GRN:
            inp = self.GRN2(inp)
        inp = self.linear(inp)[:, -self.forecast:, :]
        
        if self.Pred_Type == 'Point':
            return inp.reshape(-1, self.pred_size, self.forecast)
        elif self.Pred_Type == 'Mean_Var':
            pred = inp.reshape(-1, self.pred_size, self.forecast, 2)
            mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
            var = F.softplus(hidden_std)
            return mean, var

    def estimate_uncertainty_mean_std(self, inp, x_rec):
        return self(inp, x_rec)

    def prediction(self, dataloader, device='cpu', loss_fkt=None):
        self.RecBlock.RecBlock.init_hidden()
        self.RecBlock.precomp_hidden_states()
        return _pred_lstm(self, dataloader, device, loss_fkt)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q)
                                       for _ in range(self.n_head)])

        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k)
                                       for _ in range(self.n_head)])

        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        
        
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        

        return outputs, attn

