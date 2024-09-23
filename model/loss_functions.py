from __future__ import absolute_import, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from utils.utils import kl_loss, Gaussian


class GatheringLoss(nn.Module):
    def __init__(self,reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        score = torch.einsum('btc,bmc->btm', query, key)  # B * T x C
        score = F.softmax(score, dim=-1)  #B * T * M
        return score

    def forward(self, queries, items):
        '''
        queries : N x L x C
        items : M x C
        '''
        loss_mse = torch.nn.MSELoss(reduce=self.reduce)
        queries = queries.contiguous()  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # #B * T * M
        _, indices = torch.topk(score, 1, dim=-1)   ## B * T

        indices_items = items.gather(dim=1, index=indices.expand(-1, -1, items.size(-1)))    # B * T * C

        gathering_loss = loss_mse(queries, indices_items)  # 25600 * 512
        if self.reduce:
            return gathering_loss
        gathering_loss = torch.sum(gathering_loss, dim=-1)
        gathering_loss = gathering_loss.contiguous()  # N x L   256 * 100
        # print('diyi ',gathering_loss)
        # print('dier ',gathering_loss/ 0.1)
        # print('disan ',torch.softmax(gathering_loss/ 0.1, dim=-1))
        return gathering_loss


class KLLoss(nn.Module):
    def __init__(self,device,reduce=True):
        super(KLLoss, self).__init__()
        self.reduce = reduce
        self.device = device

        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)
    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        score = torch.einsum('btc,bmc->btm', query, key)  # B * T x C
        score = F.softmax(score, dim=-1)  #B * T * M
        return score

    def forward(self, queries, items,sigma):
        '''
        queries : N x L x C
        items : M x C
        '''
        queries = queries.contiguous()  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # #B * T * M

        prior = Gaussian(score, sigma,self.device)
        kld_loss = kl_loss(prior , score) + kl_loss(score, prior)

        kld_loss = kld_loss.contiguous()  # N x L   256 * 100
        return kld_loss
class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature,d_model, device):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)
        self.lsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.lsoftmax = nn.LogSoftmax()
        self.d_model = d_model
    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        score = torch.einsum("btc,bmc->btm", query, key)   # B * T x M
        soft_score = F.softmax(score, dim=-1)  #B * T x M
        return soft_score, score
    def euclidean_distance(self,A,B):
        # 计算差值
        diff = A - B
        # 对差值的平方进行求和，沿着最后一个维度
        squared_diff = torch.sum(diff ** 2, dim=-1)
        # 取平方根，得到欧式距离
        euclidean_distance = torch.sqrt(squared_diff)
        return euclidean_distance
    def forward(self, queries, items,sigma): # query B *T *d_model   items B * M *d_model
        batch_size = queries.size(0)
        timestep = queries.size(1)
        queries = queries.contiguous()
        soft_score, score = self.get_score(queries, items)
        distribution = torch.distributions.Categorical(logits= score)
        indices = distribution.sample()  ##  B * T
        ############################# KL散度    b * t * m
        prior = Gaussian(soft_score, sigma,self.device)
        kld_loss = torch.mean(kl_loss(prior , soft_score)) + torch.mean(kl_loss(soft_score ,prior))  ## B * T

        ####################聚集损失
        loss_mse = nn.MSELoss(reduce=False)
        _, indices_soft = torch.topk(soft_score, 1, dim=-1)
        indices_items = items.gather(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, items.size(-1)))    # B * T * C
        indices_soft_items = items.gather(dim=1, index=indices_soft.expand(-1, -1, items.size(-1)))         # B * T * C
        weight = torch.sum(loss_mse(indices_items, indices_soft_items),dim=-1)  # B * T
        #gather_loss = torch.sum(torch.sum(loss_mse(queries, indices_items),dim=-1))  # T
        gather_loss = torch.sum(torch.sum(loss_mse(queries, indices_soft_items),dim=-1))  # T
        gather_loss /= batch_size * timestep

        ####################对比损失
        logits = torch.einsum("btc,bmc->btm",queries, items)
        logits /= self.temperature
        nec = self.lsoftmax(logits)

        contrast_loss = nec.gather(dim=2, index=indices.unsqueeze(-1))         # B * T * C
        contrast_loss = torch.sum(contrast_loss)  # T
        contrast_loss /= -1. * batch_size * timestep
        # positives = F.pairwise_distance(queries, items[indices])
        # # 一个维度，使得两个矩阵的形状为 (T, 1, C) 和 (1, M , C )
        # num_rows = items.size(0)
        # mask = torch.ones((indices.size(0), num_rows), dtype=torch.bool)
        # row_indices = torch.arange(len(indices))
        # col_indices = indices
        # mask[row_indices, col_indices] = False  # 5*3
        # neg = items.unsqueeze(0).repeat(len(indices), 1, 1)[mask.unsqueeze(2).repeat(1, 1, items.size(1))].view(
        #     len(indices), -1, items.size(1))  # 25600* 4 * 128
        # negatives =torch.cdist(queries.unsqueeze(1), neg, p=2) # T * M
        # logits = torch.cat((positives.unsqueeze(1), negatives.squeeze(1)), dim=-1)
        # logits /= self.temperature
        # contrast_loss = torch.sum(self.lsoftmax(logits)[:,0])
        # contrast_loss /= -1. * batch_size * timestep
        return contrast_loss, gather_loss,kld_loss

        # batch_size = queries.size(0)
        # timestep = queries.size(1)
        # d_model = queries.size(-1)
        # queries = queries.contiguous().view(-1, d_model)  # (N x L) x C >> T x C
        # soft_score, score = self.get_score(queries, items)  # T x M
        # distribution = torch.distributions.Categorical(logits=soft_score)
        # indices = distribution.sample()
        # # sample = F.gumbel_softmax(score, tau=tau, hard=False, dim=-1)  # T * M
        # # _, indices = torch.topk(sample, 1, dim=-1)
        # ####################聚集损失
        # loss_mse = nn.MSELoss(reduce=False)
        # _, indices_soft = torch.topk(soft_score, 1, dim=1)
        # weight = torch.sum(self.similarity_function(items[indices], items[indices_soft[:, 0]]),dim=-1)  # T
        # weight = torch.softmax(weight.contiguous().view(batch_size, -1), dim=-1).view(-1)  # T
        # gather_loss = torch.sum(weight) * F.pairwise_distance(queries, items[indices]))  # T
        # gather_loss /= batch_size * timestep
        #
        # positives = F.pairwise_distance(queries, items[indices])
        # # 添加一个维度，使得两个矩阵的形状为 (T, 1, C) 和 (1, M , C )
        #
        # num_rows = items.size(0)
        # mask = torch.ones((indices.size(0), num_rows), dtype=torch.bool)
        # row_indices = torch.arange(len(indices))
        # col_indices = indices
        # mask[row_indices, col_indices] = False  # 5*3
        # neg = items.unsqueeze(0).repeat(len(indices), 1, 1)[mask.unsqueeze(2).repeat(1, 1, items.size(1))].view(len(indices), -1,items.size(1)) # 25600* 4 * 128
        #
        # negatives =torch.sum(loss_mse(queries.unsqueeze(1),neg),dim=-1)  # T * M
        # logits = torch.cat((-positives.unsqueeze(1), -negatives), dim=-1)
        # logits /= self.temperature
        # labels = torch.zeros(timestep * batch_size).to(self.device).long()
        # contrast_loss = self.criterion(logits, labels)
        # contrast_loss /= 1. * batch_size * timestep
        #
        # return contrast_loss, gather_loss
