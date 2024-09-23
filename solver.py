import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from metrics.metrics import *
from model.TransformerVar import TransformerVar
from utils.RevIN import RevIN
from utils.utils import *
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
def softmax_10(logits, dim=-1):
    # 计算以 10 为底的指数化的张量
    logits_10 =torch.pow(10.0, logits)
    return logits_10 / torch.sum(logits_10, dim=dim, keepdim=True)
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
    def __call__(self, val_loss, model, path,ids):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,ids)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path,ids)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path,ids):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset)+ str(ids) + f'_checkpoint.pth'))
        self.val_loss_min = val_loss

def save_best_checkpoint(model, path, dataset,ids):
    torch.save(model.state_dict(), os.path.join(path, str(dataset)+str(ids) + f'_best_checkpoint.pth'))


class OneEarlyStopping:
    def __init__(self, patience=5, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.F1_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, F1, model, path):
        score = F1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(F1, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(F1, model, path)
            self.counter = 0

    def save_checkpoint(self, F1, model, path):
        if self.verbose:
            print(f'F1 increased ({self.F1_min:.6f} --> {F1:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint.pth'))
        self.F1_min = F1
        return True

torch.autograd.set_detect_anomaly(True)


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index,self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.criterion = nn.MSELoss()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
    def build_model(self):
        self.model = TransformerVar(win_size =self.win_size,enc_in=self.input_c, c_out=self.output_c, \
                                    d_model=self.d_model, n_memory=self.n_memory,
                                    batch_size=self.batch_size, contras_temperature=self.contras_temperature,
                                    zero_probability=self.zero_probability, read_K=self.read_K,
                                    read_tau=self.read_tau,topk = self.topk,
                                    device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,betas=(0.9, 0.999))
        if torch.cuda.is_available():
            self.model.to(self.device)
    def vali(self, vali_loader):
        with torch.no_grad():
            self.model.eval()
            valid_loss_list = []
            vail_rec_loss_list = []
            valid_contrast_loss_list = []
            valid_gather_list = []
            valid_KL_list = []

            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output_dict = self.model(input,mode='val')
                output, queries = output_dict['out'], output_dict['queries']
                contrastloss = output_dict['contrastloss']
                gather_loss = output_dict['gather_loss']
                kld_loss = output_dict['kld_loss']

                rec_loss = self.criterion(output, input)
                loss = rec_loss + self.lamda_1 * kld_loss

                vail_rec_loss_list.append(rec_loss.item())
                valid_contrast_loss_list.append(contrastloss.item())
                valid_gather_list.append(gather_loss.item())
                valid_KL_list.append(kld_loss.item())

                valid_loss_list.append(loss.item())
            return np.average(valid_loss_list),np.average(vail_rec_loss_list),np.average(valid_contrast_loss_list),np.average(valid_gather_list),np.average(valid_KL_list)

    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)
        early_stopping = TwoEarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        # early_stopping = OneEarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        f1 = 0
        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = [];contrast_loss_list=[]
            gather_loss_list = []
            kl_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output_dict = self.model(input,mode='train')
                output, m_items, queries = output_dict['out'], output_dict['m_items'], output_dict['queries']
                kld_loss = output_dict['kld_loss']

                rec_loss = self.criterion(output, input)

                loss = rec_loss + self.lamda_1 * kld_loss

                loss_list.append(loss.item())
                kl_loss_list.append(kld_loss.item())
                rec_loss_list.append(rec_loss.item())
                if (i + 1) % 200 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.mean().backward()
                self.optimizer.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)
            train_kl_loss = np.average(kl_loss_list)

            train_rec_loss = np.average(rec_loss_list)
            valid_loss,valid_rec,valid_contrast,valid_gather,valid_kl = self.vali(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {2:.7f} KL loss Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_rec_loss,train_kl_loss))
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {2:.7f} KL loss Loss: {3:.7f}".format(
                    epoch + 1, train_steps, valid_rec,valid_kl))
            acc,prec,recall,F1,_ = self.test()
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(acc, prec, recall, F1))
            ####################
            early_stopping(valid_loss, self.model, path,self.read_K)
            # early_stopping(F1, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
    def test(self,test=0):
        if test == 1:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.dataset)+ str(self.read_K) + '_checkpoint.pth')))
            # self.model.load_state_dict(
            #     torch.load(
            #         os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
            print("======================TEST MODE======================")
        with torch.no_grad():
            self.model.eval()
            criterion = nn.MSELoss(reduce=False)
            gathering_loss = GatheringLoss(reduce=False)
            kl_loss = KLLoss(self.device,reduce=False)
            temperature = self.temperature
            test_labels = []
            true_list= []

            test_rec_energy = []
            test_latent_energy = []
            test_kl_energy = []

            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output_dict= self.model(input)
                output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['m_items']
                sigma = output_dict['sigma']
                rec_loss = torch.mean(criterion(input, output), dim=-1)
                rec_loss = torch.softmax(rec_loss/temperature, dim=-1)

                latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
                kl_score = torch.softmax(kl_loss(queries, mem_items,sigma)/temperature, dim=-1)

                rec_loss = rec_loss.detach().cpu().numpy()
                latent_score = latent_score.detach().cpu().numpy()
                kl_score = kl_score.detach().cpu().numpy()

                test_rec_energy.append(rec_loss)
                test_latent_energy.append(latent_score)
                test_kl_energy.append(kl_score)

                test_labels.append(labels)
                input = input.detach().cpu().numpy()
                true_zhi = input[:, :, 0:1]
                true_list.append(true_zhi)

            test_rec_energy = np.concatenate(test_rec_energy, axis=0).reshape(-1, 1)
            # test_rec_energy = StandardScaler().fit_transform(test_rec_energy)
            # test_rec_energy = MinMaxScaler().fit_transform(test_rec_energy)


            test_latent_energy = np.concatenate(test_latent_energy, axis=0).reshape(-1, 1)
            # test_latent_energy = StandardScaler().fit_transform(test_latent_energy)

            test_kl_energy = np.concatenate(test_kl_energy, axis=0).reshape(-1, 1)

            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)  # 51200
            true_list = np.concatenate(true_list, axis=0).reshape(-1, 1)  # 50688 * 1
            true_list = np.array(true_list).reshape(-1)

            x = self.beta_1 *  test_latent_energy + (1 - self.beta_1) * test_kl_energy + self.beta_2 * test_rec_energy
            test_energy = np.array(x).reshape(-1)
            Accuracy = 0
            F1 = 0
            Precision = 0
            Recall = 0
            thr = 0
            thre = [(90 + (i / 10)) for i in range(100)]
            thres = np.percentile(test_energy, thre)
            for i in thres:
                thresh = i
                pred_4 = (test_energy > thresh).astype(int)
                gt_4 = test_labels.astype(int)

                pred = (test_energy > thresh).astype(int)
                gt = test_labels.astype(int)
                anomaly_state = False
                for i in range(len(gt)):
                    if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                        anomaly_state = True
                        for j in range(i, 0, -1):
                            if gt[j] == 0:
                                break
                            else:
                                if pred[j] == 0:
                                    pred[j] = 1
                        for j in range(i, len(gt)):
                            if gt[j] == 0:
                                break
                            else:
                                if pred[j] == 0:
                                    pred[j] = 1
                    elif gt[i] == 0:
                        anomaly_state = False
                    if anomaly_state:
                        pred[i] = 1
                pred = np.array(pred)
                gt = np.array(gt)
                accuracy = accuracy_score(gt, pred)
                precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                                      average='binary')
                if f_score > F1:
                    Accuracy = accuracy
                    F1 = f_score
                    Precision = precision
                    Recall = recall
                    thr = thresh
                    Gt = gt
                    Pred = pred
                    Gt_4 = gt_4
                    Pred_4 = pred_4

            if test == 1:
                # pairwise_distances = torch.cdist(mem_items, mem_items, p=2)
                # mermory_heat_map(pairwise_distances[1].detach().cpu().numpy(),2)
                # visualization(thr,Pred,test_energy,test_labels,true_list)
                print("Threshold :", thr)
                print(
                    "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        Accuracy,
                        Precision,
                        Recall,
                        F1))
                print('=' * 50)
                f = open("result.txt", 'a')
                f.write("数据集  " + str(self.dataset) + "  \n")
                f.write("read_K" + str(self.read_K) + "  \n")
                f.write(
                    "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(Precision, Recall,
                                                                                       F1) + "  \n")
                f.write('\n')
                f.close()

                ############################# 其他评价指标
                matrix = [self.index]
                scores_simple = combine_all_evaluation_scores(Pred, Gt, test_energy)
                for key, value in scores_simple.items():
                    matrix.append(value)
                    print('{0:21} : {1:0.4f}'.format(key, value))
                # import csv
                # with open('result/' + self.dataset + '.csv', 'a+') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(matrix)
            ###############################
            return Accuracy, Precision, Recall, F1, thr
