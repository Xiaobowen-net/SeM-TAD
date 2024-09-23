from __future__ import absolute_import, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
class Gate_netwook(nn.Module):
    def __init__(self, n_memory, fea_dim,topk,device=None):
        super(Gate_netwook, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.device = device
        self.W = nn.Linear(fea_dim, 1)
        self.topk =topk
        self.softmax=nn.Softmax(dim=1)
    # def forward(self, m_items_matrix,query): ##m_items_matrix= t * d_models
    #     # elu = self.U(query.permute(1,0))
    #     weight = self.softmax(self.W(m_items_matrix))  # weight =B * t * 1
    #     elu = torch.einsum('bti,btd->bid', weight, m_items_matrix)  # N x L x d
    #     return elu
    def forward(self, m_items_matrix,query): ##m_items_matrix= t * d_models
        #weight = self.softmax(torch.norm(m_items_matrix, p=2, dim=-1).unsqueeze(-1))
        weight = self.softmax(self.W(m_items_matrix))
        top_k_weights, top_k_indices = torch.topk(weight ,self.topk, dim=1)  # b * 11 * 1
        indices_items = m_items_matrix.gather(dim=1, index=top_k_indices.expand(-1, -1, m_items_matrix.size(-1)))
        elu = torch.einsum('bkc,bki->bic', indices_items, top_k_weights)
        return elu

class DMP(nn.Module):
    def __init__(self, n_memory, fea_dim,topk,device=None):
        super(DMP, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.device = device
        self.topk = topk
        self.memory_function = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=self.fea_dim, out_features=self.fea_dim),  # 输入层到隐藏层
                nn.LeakyReLU(negative_slope=0.01+(0.2 / self.n_memory)*l ),
                nn.Linear(in_features=self.fea_dim, out_features=self.fea_dim),
            nn.Tanh()) for l in range(self.n_memory)
        ])
        # self.memory_function =nn.Sequential(
        #         nn.Linear(in_features=self.fea_dim, out_features=self.fea_dim),  # 输入层到隐藏层
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.Linear(in_features=self.fea_dim, out_features=self.fea_dim),nn.Tanh())
        self.gate_network =nn.ModuleList([
             Gate_netwook(self.n_memory, self.fea_dim,self.topk) for l in range(self.n_memory)])
    def forward(self, query):
        '''
        query (encoder output features) : b * L * d_models
        '''
        s = query.data.shape
        query = query.contiguous()
        m_item_list = [] # n x C
        for i, (memory_layer, gate_layer) in enumerate(zip(self.memory_function, self.gate_network)):
            x = memory_layer(query)
            m_item_list.append(gate_layer(x,query))
        m_item = torch.cat(m_item_list, dim=1)
        m_items = F.normalize(m_item, dim=-1)
        return m_items
    # def forward(self, query):
    #     '''
    #     query (encoder output features) : b * L * d_models
    #     '''
    #     s = query.data.shape
    #     query = query.contiguous()
    #     m_item_list = [] # n x C
    #     x = self.memory_function(query)
    #
    #     for i, gate_layer in enumerate(self.gate_network):
    #         m_item_list.append(gate_layer(x,query))
    #     m_item = torch.cat(m_item_list, dim=1)
    #     m_items = F.normalize(m_item, dim=-1)
    #     return m_items