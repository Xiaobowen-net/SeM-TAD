import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
def min_max_normalize_np(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def normalized(data_np):
    mean = np.mean(data_np)
    std_dev = np.std(data_np)
    normalized_data = (data_np - mean) / std_dev
    return normalized_data

def min_max_normalization(tensor):
    # 找到张量中每列的最小值和最大值
    min_vals, _ = torch.min(tensor, dim=1)
    max_vals, _ = torch.max(tensor, dim=1)
    # 最大-最小归一化
    normalized_tensor = (tensor - min_vals.unsqueeze(1)) / (max_vals.unsqueeze(1) - min_vals.unsqueeze(1))

    return normalized_tensor

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
def categorie(data):
    # 使用imshow函数绘制热力图
    mkdir('zhuzhuangtu')
    categories =[i for i in range(10)]
    plt.bar(categories, data, color='blue')
    plt.title('Example Bar Chart')
    plt.xlabel('Categories')
    plt.ylabel('Values')    # 显示图形

    plt.savefig('./zhuzhuangtu/'+str(data[0])+'.png')
    plt.close()
def mermory_heat_map(data,id):
    # 使用imshow函数绘制热力图
    mkdir('figure')
    plt.imshow(data, cmap='hot', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    # 显示图形
    plt.savefig('./figure/'+str(id)+'.png')
    plt.close()
def heat_map(data,phase_type,id):
    # 使用imshow函数绘制热力图
    mkdir('quers')
    plt.imshow(data, cmap='hot', interpolation='nearest')
    # 添加颜色条
    plt.colorbar()
    # 显示图形
    plt.savefig('./quers/'+phase_type+str(id)+'.png')
    plt.close()


def att_mermory_heat_map(data,phase_type,id):
    # 使用imshow函数绘制热力图
    mkdir('scoremermory')
    plt.figure(figsize=(12, 6))
    heatmap = plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 100, 0, 10])
    # 添加颜色条
    plt.colorbar(heatmap)
    # 添加边缘线
    # 调整刻度
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 11, 1))
    plt.savefig('./scoremermory/'+phase_type+str(id)+'.png')
    plt.close()
def att_quer_heat_map(data,id):
    # 调整图像大小
    mkdir('scorequery')

    plt.figure(figsize=(12, 6))

    # 画热力图
    heatmap = plt.imshow(data.T, cmap='viridis', interpolation='nearest', aspect='auto', extent=[0, 100, 0, 10])

    # 添加颜色条
    plt.colorbar(heatmap)

    # 添加边缘线
    # 调整刻度
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 11, 1))
    plt.savefig('./scorequery/'+str(id)+'.png')
    plt.close()

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


##1代表异常，0 表示正常
def point_score(outputs, trues):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse(outputs, trues)       #128 * 100* 55
    normal = (1 - torch.exp(-error))   #  128 * 100*55
    score = (torch.sum(normal* loss_func_mse(outputs,trues),dim =-1) / torch.sum(normal,dim=-1))
    return score

def cos_score(outputs, trues):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse(outputs, trues)       #128 * 100* 55
    error = F.softmax(error,dim=-1)
    normal = 1/(1-error)   # 128 * 100
    score = (torch.sum(normal* loss_func_mse(outputs,trues),dim =-1) / torch.sum(normal,dim=-1))
    return score


def visualization(best_thresh,point_label,anomaly_score,true_labels,true_list):
    win_size = 230
    path_true = os.path.join(os.getcwd(), str('picture'), "true")
    path_score = os.path.join(os.getcwd(), str('picture'), "Score")
    if (not os.path.exists(path_true)):
        mkdir(path_true)
    if (not os.path.exists(path_score)):
        mkdir(path_score)
    index_len =(point_label.shape[0] - win_size) // win_size+ 1
    for i in range(0,index_len):
        index = i * win_size
        print(index)
        pre_label = point_label[index:index+win_size]
        score =anomaly_score[index:index+win_size]
        true_label =true_labels[index:index+win_size]
        true =true_list[index:index+win_size]
        plt.figure()
        plt.tick_params()
        plt.plot(true, label='Ground truth')
        Anomaly =True
        for i, value in enumerate(true_label):
            if value == 1:
                if Anomaly:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3, label='True anomaly')
                    Anomaly = False
                else:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3)
        plt.legend()
        plt.xlabel('Time points')
        x_major_locator = MultipleLocator(50)
        # 把x轴的刻度间隔设置为1，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        plt.savefig(path_true+str(index)+'.png')
        plt.close()
        plt.figure()
        plt.plot(score, label='Anomaly score')
        plt.tick_params()
        plt.axhline(y=best_thresh, color='red', linestyle='--', label='Threshold')
        Anomaly =True
        for i, value in enumerate(pre_label):
            if value == 1:
                if Anomaly:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3, label='Pred anomaly')
                    Anomaly = False
                else:
                    plt.axvspan(i, i + 1, color='pink', alpha=0.3)
        plt.legend()
        plt.xlabel('Time points')
        x_major_locator = MultipleLocator(50)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(path_score+str(index)+'.png')
        plt.close()


def xiangxiantu(test_latent_energy,test_labels):
    abnormal = []
    normal= []
    for i in range(0,len(test_labels)):
        if test_labels[i] == 1:
            abnormal.append(test_latent_energy[i])
        elif test_labels[i] == 0:
            normal.append(test_latent_energy[i])
    abnormal_mean = np.mean(abnormal)
    abnormal_var = np.var(abnormal)
    print('异常数据均值',abnormal_mean)
    print('异常数据方差',abnormal_var)
    abnormal_min = np.min(abnormal)
    abnormal_xia = np.percentile(abnormal, 25)
    abnormal_media = np.median(abnormal)
    abnormal_sha = np.percentile(abnormal, 75)
    abnormal_max = np.max(abnormal)

    print("异常数据最小值:", abnormal_min)
    print("异常数据第一四分位数:", abnormal_xia)
    print("异常数据中位数:", abnormal_media)
    print("异常数据第三四分位数:", abnormal_sha)
    print("异常数据最大值:", abnormal_max)

    normal_mean = np.mean(normal)
    normal_var = np.var(normal)
    print('正常数据均值', normal_mean)
    print('正常数据方差', normal_var)
    normal_min = np.min(normal)
    normal_xia = np.percentile(normal, 25)
    normal_media = np.median(normal)
    normal_sha = np.percentile(normal, 75)
    normal_max = np.max(normal)

    print("正常数据最小值:", normal_min)
    print("正常数据第一四分位数:", normal_xia)
    print("正常数据中位数:", normal_media)
    print("正常数据第三四分位数:", normal_sha)
    print("正常数据最大值:", normal_max)

    f = open("dp.txt", 'a')
    f.write('异常数据均值' + str(abnormal_mean) + "  \n")
    f.write('异常数据方差' + str(abnormal_var) + "  \n")
    f.write('正常数据均值' + str(normal_mean) + "  \n")
    f.write('正常数据方差' + str(normal_var) + "  \n" )
    f.write('\n')
    f.write('\n')
    f.close()


def kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

def Gaussian(score,sigma,device):  # sigma b * T
    batch_size = score.size(0)
    timestep = score.size(1)
    memory = score.size(2)
    sigma = torch.sigmoid(sigma * 5) + 1e-5
    sigma = torch.pow(3, sigma) - 1  # B L
    sigma = sigma.unsqueeze(-1).repeat(1,1, memory)

    _, indices = torch.topk(score, 1, dim=-1)
    positions = torch.arange(memory).unsqueeze(0).unsqueeze(0).repeat(batch_size,timestep, 1).to(device)

    prior = torch.abs(positions - indices.repeat(1,1,memory))
    prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(- prior ** 2 / 2 /(sigma ** 2))
    prior = prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1,memory)
    return prior