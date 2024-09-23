import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

from model.DPM import DMP
from model.loss_functions import ContrastLoss, GatheringLoss


class Attention(nn.Module):
    def __init__(self,memory,fea_dim, device,contras_temperature):
        super(Attention, self).__init__()
        self.n_memory = memory
        self.fea_dim = fea_dim
        self.device = device
        self.dmp =DMP(self.n_memory, self.fea_dim,device=self.device)
        self.gathering_loss = GatheringLoss(reduce=False)
        self.contrastloss =ContrastLoss(temperature=contras_temperature,device=device)


    def get_attn_score(self, query, key):
        '''
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C      M*C    B * L *C
        '''
        attn = torch.einsum('btc,bmc->btm', query, key)   # B * T x M
        attn = F.softmax(attn, dim=-1)
        return attn
    def forward(self, queries, keys):  # # 256 * 100 * 55 * 128
        '''
        queries : N x L x Head x d
        keys : N x L(s) x Head x d
        values : N x L x Head x d
        '''

        attn_scores = torch.einsum('ncd,nsd->ncs', queries, keys)

        attn_weights = F.normalize(attn_scores, dim=-1)   #season = b * t * d
        #attn_weights =torch.softmax(attn_scores, dim=-1)     #B * C * C

        mem_items = self.dmp(attn_weights) #B * n * C

        attn_contrast , attn_gather = self.contrastloss(attn_weights,mem_items)

        attn_loss = self.gathering_loss(attn_weights,mem_items)

        return attn_loss,attn_contrast,attn_gather
    

class AttentionLayer(nn.Module):
    def __init__(self, patch_len, d_ff,n_memory, c_dim , device,contras_temperature):
        super(AttentionLayer, self).__init__()
        self.patch_len = patch_len  # d_model = C
        self.d_ff = d_ff
        self.n_memory = n_memory
        self.c_dim = c_dim
        self.device = device
        # Linear projections to Q, K, V
        self.W_Q = nn.Linear(self.patch_len, self.d_ff)
        self.W_K = nn.Linear(self.patch_len, self.d_ff)
        self.stride = 1
        self.attn = Attention(self.n_memory, self.c_dim,device=self.device,contras_temperature =contras_temperature)

        self.padding_patch_layer = nn.ReplicationPad1d((self.patch_len//2 , self.patch_len//2))

    def forward(self, input):   # 256 * 55 * 100
        '''
        input : N x L x C(=d_model)
        '''
        N, C, L = input.shape
        z = self.padding_patch_layer(input)
        representations = z.unfold(dimension=-1, size=self.patch_len,step=self.stride)  # representations B * C * P * N  128 * 55 *100 * 3
        input_data = representations.permute(0,2,1,3).contiguous().view(-1,C,self.patch_len)   ## 25600 * 55 * 3

        Q = self.W_Q(input_data)
        K = self.W_K(input_data)
        attn_loss,attn_contrast,attn_gather= self.attn(Q, K) # 25600 * 55 * 32

        attn_loss = attn_loss.view(N, L , C)
        return attn_loss,attn_contrast,attn_gather

