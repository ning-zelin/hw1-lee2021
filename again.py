import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt
import csv
from torch.utils.data import DataLoader, Dataset

"""prepare data"""

fp_train = r'D:\DeepLearning\Data\(新冠预测)ml2021spring-hw1/covid.train.csv'
fp_test = r'D:\DeepLearning\Data\(新冠预测)ml2021spring-hw1/covid.test.csv'

with open(fp_train, 'r') as fp:
    data = list(csv.reader(fp))[1:]
    data = np.array(data, dtype = np.float32)[:, 1:]
    data = torch.Tensor(data).cuda()
    mean = data[:, 40: -1].mean(dim = 0, keepdim = True)
    std = data[:, 40: -1].std(dim = 0, keepdim = True)
class codata(Dataset):
    def __init__(self, fp = fp_train, mode = 'train'):
        self.fp = fp
        self.mode = mode
        with open(fp, 'r') as fp:
            data = list(csv.reader(fp))[1:]
            data = np.array(data, dtype = np.float32)[:, 1:]
            data = torch.Tensor(data).cuda()
            # data[:, 40: -1] = (data[:, 40: -1] - mean) / std

        if mode in ['train', 'dev']:
            tr_idx = [i for i in range(len(data)) if i % 5 != 0]
            dev_idx = [i for i in range(len(data)) if i % 5 == 0]

            target = data[:, -1]
            data = data[:, :-1]
            if mode == 'train':
                self.data = data[tr_idx]
                self.target = target[tr_idx]

            if mode == 'dev':
                self.data = data[dev_idx]
                self.target = target[dev_idx]

        if mode == 'test':
            self.data = data

        self.data[:, 40:] = self.data[:, 40:] - mean
        self.data[:, 40:] = self.data[:, 40:] / std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.data[idx]
        else:
            return self.data[idx], self.target[idx]

def getloader(mode):
    if mode == 'test':
        dataset = codata(fp_test, 'test')
        dataloader = DataLoader(dataset, batch_size = 540, shuffle = False)
    else:
        dataset = codata(fp_train, mode)
        dataloader = DataLoader(dataset, batch_size = 160, shuffle = True)
    return dataloader

"""design model"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(93, 45),
            nn.ReLU(),
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.net2 = nn.Sequential(
            nn.Linear(93, 45),
            nn.ReLU(),
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.net3 = nn.Sequential(
            nn.Linear(93, 45),
            nn.ReLU(),
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.net1(x) + self.net2(x) + self.net3(x)
        return x
model = Net().cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.SGD(model.parameters() ,lr = 0.001, momentum = 0.9)

"""training cycle"""
train_loader = getloader('train')
dev_loader = getloader('dev')
test_loader = getloader('test')
train_record = []
def train():
    for x, y in train_loader:
        model.train()
        pred = model(x)
        loss = criterion(pred, y.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_item = loss.detach().cpu().item()
        train_record.append(loss_item)
dev_record = []
def dev():
    model.eval()
    with torch.no_grad():
        for x, y in dev_loader:
            pred = model(x)
            loss = criterion(pred, y.view(-1, 1))
            loss_item = loss.detach().cpu().item()
            dev_record.append(loss_item)
            return loss_item


def test():
    model.eval()
    test_set = test_loader
    preds = []
    with torch.no_grad():
        for x in test_set:
            x = x.cuda()
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim = 0).view(-1).numpy()
    return preds
def training_cycle():
    epoch = 0
    min_loss = 1000
    early_stop_cnt = 0
    while epoch < 5000:
        epoch += 1
        train()
        dev_loss = dev()
        if dev_loss < min_loss:
            min_loss = dev_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), r'./model.pth')
            print('save model', epoch, min_loss)
        else:
            early_stop_cnt += 1
            if early_stop_cnt == 500:
                break
    print('finish ', epoch)
def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])