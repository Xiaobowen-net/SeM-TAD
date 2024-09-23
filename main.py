import argparse
import torch

from torch.backends import cudnn
from utils.utils import *
##################
# ours
from solver import Solver
def str2bool(v):
    return v.lower() in ('true')
def set_seed(seed_value):
    if seed_value == -1:
        return
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
def main(config):
    #set_seed(2022)
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        # m_items = F.normalize(torch.rand((config.n_memory, config.patch_len * config.d_model), dtype=torch.float), dim=1).to(config.device)
        # pairwise_distances = torch.cdist(point_m_items, point_m_items, p=2)
        # mermory_heat_map(pairwise_distances.detach().cpu().numpy(), 1)
        solver.train()
        # 计算所有行之间的余弦相似度
        solver.test(test=1)
        # pairwise_distances = torch.cdist(m_items, m_items, p=2)
        # mermory_heat_map(pairwise_distances.detach().cpu().numpy(),2)
    elif config.mode == 'test':
        # load_path = f'./memory_item/{config.dataset}_memory_item.pth'
        # m_items = torch.load(load_path).to(config.device)
        # print('loading memory item vectors')
        solver.test(test=1)
    return solver
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--data_path', type=str, default='./data/MSL/MSL/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--n_memory', type=int, default=5, help='number of memory items')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.1, help='潜在空间偏差温度参数')
    parser.add_argument('--lamda_1', type=float, default=1)
    parser.add_argument('--beta_1', type=float, default=1)
    parser.add_argument('--beta_2', type=float, default=1)
    parser.add_argument('--contras_temperature',type=float, default= 0.5, help='contrast learn temperature')
    parser.add_argument('--zero_probability',type=float, default=1.0, help='越大，表示采用随机挑选越大')
    parser.add_argument('--read_K',type=int, default=5, help='contrast learn temperature')
    parser.add_argument('--read_tau',type=float, default=0.5, help='contrast learn temperature')
    parser.add_argument('--topk',type=int, default=30, help='top k memory')

    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
