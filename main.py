import torch
from torch.utils.data import DataLoader, Dataset
import csv
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

"""prepare data"""

fp_train = r'D:\DeepLearning\Data\(新冠预测)ml2021spring-hw1/covid.train.csv'
fp_test = r'D:\DeepLearning\Data\(新冠预测)ml2021spring-hw1/covid.test.csv'


class COVID19DATA(Dataset):
    def __init__(self, fp, mode = 'train'):
        # fp ： filepath
        self.mode = mode
        with open(fp, 'r') as f_p:
            data = list(csv.reader(f_p))
            data = np.array(data[1:], dtype = np.float32)
            data = data[:, 1:]
            data = torch.FloatTensor(data)
            data[:, 40: -1] = (data[:, 40:-1] - data[:, 40:-1].mean(dim = 0, keepdim = True)) / \
                           (data[:, 40:-1].max(0, keepdim = True)[0] - data[:, 40: -1].min(0, keepdim = True)[0])
        if mode == 'test':
            self.data = data

        else:
            target = data[:, -1]
            data = data[:, :-1]

            if mode == 'train':
                index = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                index = [i for i in range(len(data)) if i % 10 == 0]

            self.data = data[index]
            self.target = target[index]

    def __getitem__(self, idx):
        if self.mode in ['train', 'dev']:
            return self.data[idx], self.target[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_loader(fp, mode):
    dataset = COVID19DATA(fp, mode)
    dataloader = DataLoader(dataset, batch_size = 100)  # mode = 'train'时 shuffle为true
    return dataloader


""" design model"""


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(93, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.net1(x)
        return x


model = NN().cuda()
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

"""training cycle"""
dev_loss_record = []
dev_loader = get_loader(fp_train, 'dev')


def dev():
    loss_item = 0
    model.eval()
    with torch.no_grad():
        for x, y in dev_loader:
            x, y = x.cuda(), y.cuda().view(-1, 1)
            pred = model(x)
            loss = criterion(pred, y)
            loss_item += loss.detach().cpu().item() * len(x)
        loss_item = loss_item / len(dev_loader.dataset)
        dev_loss_record.append(loss_item)
        return loss_item


train_loss_record = []
train_loader = get_loader(fp_train, 'train')


def train():
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda().view(-1, 1)
        pred = model(x)
        loss = criterion(pred, y)
        loss_item = loss.detach().cpu().item()
        train_loss_record.append(loss_item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def training_cycle():
    epoch = 0
    min_loss = 1000
    early_stop_cnt = 0
    while epoch < 500:
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
            if early_stop_cnt == 200:
                break
    print('finish ', epoch)


def plot():
    x1 = range(len(train_loss_record))
    x2 = x1[::(len(train_loss_record) // len(dev_loss_record))]
    plt.plot(x1, train_loss_record, label = 'train')
    plt.plot(x2, dev_loss_record, label = 'dev')
    plt.legend()
    plt.show()


def test():
    model.eval()
    test_set = get_loader(fp_test, 'test')
    preds = []
    with torch.no_grad():
        for x in test_set:
            x = x.cuda()
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim = 0).view(-1).numpy()
    return preds


def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


training_cycle()
plot()