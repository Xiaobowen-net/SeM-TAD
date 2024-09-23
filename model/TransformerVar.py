import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import heat_map, att_mermory_heat_map
from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding
from .encoder import STEncoder
from .loss_functions import ContrastLoss

# ours
from .ours_memory_module import MemoryModule
# memae
# from .memae_memory_module import MemoryModule
# mnad
# from .mnad_memory_module import MemoryModule
class Decoder(nn.Module):
    def __init__(self, d_model, c_out):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)
    def forward(self, x):
        out = self.out_linear(x)
        return out      # N x L x c_out
class Patch_time(torch.nn.Module):
    def __init__(self, win_size,patch_len,d_model,d_ff):
        super(Patch_time, self).__init__()
        self.patch_len = patch_len
        self.stride = patch_len
        self.win_size =win_size
        self.d_ff =d_ff
        self.W = nn.Linear(patch_len*d_model, d_ff)
        self.patch_num = int((self.win_size - self.patch_len)/self.stride + 1)+1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))  #右边填充 stride 个值, 左边填充 0 个
    def forward(self, z):  #B * L * C
        batch_size = z.shape[0]  #128
        z = z.permute(0, 2, 1)     #z B * C * L
        z = self.padding_patch_layer(z)
        representations = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  #representations:B * C * P * N  128 *128 *51 * 2
        representations = representations.permute(0, 2, 3 , 1).reshape(batch_size,self.patch_num,-1)   # B * P N*C   128 * 51 * 256
        return representations
class TransformerVar(nn.Module):
    def __init__(self, win_size,enc_in, c_out, n_memory,
                 batch_size, contras_temperature,
                 zero_probability, read_K,read_tau,topk,\
                 shrink_thres=0, \
                 d_model=512, \
                 device=None):
        super(TransformerVar, self).__init__()
        # Encoder
        self.d_model = d_model
        self.win_size = win_size
        self.contras_temperature = contras_temperature
        self.encoder =STEncoder(input_dims=enc_in,device=device)
        self.point_mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model,
                                       contras_temperature =contras_temperature,zero_probability=zero_probability, read_K=read_K,
                                       read_tau=read_tau,topk=topk, shrink_thres=shrink_thres, device=device)
        # ours
        self.weak_decoder = Decoder(3 * d_model, c_out)
        self.trend_linear = nn.Linear(d_model, d_model)
        self.sigma_projection = nn.Linear(self.d_model , 1)
        self.contrastloss =ContrastLoss(temperature=self.contras_temperature,d_model=d_model,device=device)

    def forward(self, x,mode='test'):
        '''
        x (input time window) : N x L x enc_in
        '''
        s = x.data.shape
        trend, season,query = self.encoder(x)
        trend = self.trend_linear(trend)
        season = F.normalize(season, dim=-1)   #season = b * t * d
        outputs = self.point_mem_module(season,mode)
        read_query, mitems = outputs['output'], outputs['m_items']

        sigma = self.sigma_projection(season).view(s[0],s[1])
        contrastloss , gather_loss ,kld_loss= self.contrastloss(season,mitems,sigma)

        read_query = read_query.reshape(s[0],-1,self.d_model)
        out = torch.cat((query,trend,read_query), dim=-1)
        out = self.weak_decoder(out)
        return {"out":out, "m_items":mitems, "queries":season,"contrastloss":contrastloss,"gather_loss":gather_loss,"kld_loss":kld_loss,"sigma":sigma}