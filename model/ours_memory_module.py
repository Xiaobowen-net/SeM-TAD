from __future__ import absolute_import, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.DPM import DMP

class MemoryModule(nn.Module):
    def __init__(self,n_memory, fea_dim,contras_temperature,zero_probability,read_K,read_tau,topk,shrink_thres=0.0025, device=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.contras_temperature = contras_temperature
        self.zero_probability =zero_probability
        self.read_K= read_K
        self.read_tau = read_tau
        self.topk = topk
        self.count = 0
        self.dmp =DMP(self.n_memory, self.fea_dim,self.topk,device=self.device)
    def get_attn_score(self, query, key):
        '''
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C      M*C    B * L *C
        '''
        attn = torch.einsum('btc,bmc->btm', query, key)   # B * T x M
        attn = F.softmax(attn, dim=-1)
        return attn
    def attn_score(self,query, key,K=20, tau = 0.3):
        attn = torch.einsum('btc,bmc->btm', query, key)  # B * T x M * 1
        gumbels = (-torch.empty(attn.shape+ (K,), memory_format=torch.legacy_contiguous_format).exponential_().log()).to(self.device) #B * T *M *K
        attn = gumbels + attn.unsqueeze(-1)
        attn = F.softmax(attn/tau, dim=2)
        return attn

    def generate_binary_matrix(self,batch_size,time_steps,zero_probability):
        mask_ratio = zero_probability
        mask = torch.rand(batch_size, time_steps) < mask_ratio
        mask = mask.unsqueeze(-1)
        return mask

    def read(self, query, m_items,mode): # m_items B * n * d_model
        s = query.data.shape
        query = query.contiguous() # B * T x C
        if mode == 'train':
            binary_matrix = torch.ones((s[0],s[1], 1))
            ones_matrix = torch.ones((s[0],s[1],1)) # 0为确定，1为随机
            binary_matrix = binary_matrix.to(self.device)
            ones_matrix = ones_matrix.to(self.device)

            zero_probability = self.zero_probability  ## 0.1
            # 生成一个二进制矩阵
            s = query.data.shape
            mask = self.generate_binary_matrix(s[0],s[1], zero_probability).to(self.device)  #mask B * T * 1
            attn= torch.einsum('btc,bmc->btm', query, m_items)  # B * T x M
            binary_matrix = binary_matrix.masked_fill(mask, 0)  #为true的全为0
            attn = F.softmax(attn, dim=-1)* binary_matrix
            add_que_memory= torch.einsum('btm,bmc->btc', attn, m_items)  # B * T x C

            ones_matrix = torch.where(mask, ones_matrix, torch.tensor(0., device=self.device))
            attn = self.attn_score(query, m_items,self.read_K, self.read_tau) # T * M *K
            add_mem = torch.einsum("btmk,bmc->kbtc", attn, m_items)
            add_sui_memory = torch.mean(add_mem, dim=0)* ones_matrix
            read_query = add_sui_memory + add_que_memory
        else:
            attn = self.get_attn_score(query, m_items) # B * T x M
            read_query = torch.einsum('btm,bmc->btc', attn, m_items)  # B * T x C
        # read_query = F.normalize(read_query, dim=-1)
        return read_query

    def forward(self, query,mode):
        '''
        query (encoder output features) : batch * L * d_models
        '''
        mem_items = self.dmp(query)     ## mem_items B * n * d_model
        read_query = self.read(query,mem_items, mode)
        return {'output': read_query, 'm_items':mem_items}