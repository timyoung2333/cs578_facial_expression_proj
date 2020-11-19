#!/usr/bin/env python3
# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
from FER2013 import FER2013
import numpy as np
from tqdm import tqdm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def train(self, X, y, epoch_num=500, debug=False):
        inputs = torch.Tensor(X.reshape((len(X), 1, 48, 48)))
        labels = torch.Tensor(y)

        # inputs = inputs.to(self.device)
        # labels = labels.to(self.device).long()

        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=8)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epoch_num):  # loop over the dataset multiple times

            loss = None
            for local_batch, local_labels in dataloader:

                local_batch = local_batch.to(self.device)
                local_labels = local_labels.to(self.device).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(local_batch)
                loss = criterion(outputs, local_labels)
                loss.backward()
                optimizer.step()

            # print loss
            if debug:
                score1 = self.score(X_train, y_train)
                score2 = self.score(X_test, y_test)
                scores_train.append(score1)
                scores_test.append(score2)
                print("epoch: {}, score (train): {:.3f}, score (test): {:.3f}".format(epoch+1, score1, score2))
            else:
                print("epoch: {}, loss: {:.3f}".format(epoch+1, loss.item()))

        print('Finished Training')

        if debug:
            pickle.dump(scores_train, open("../result/iter_vs_acc/ResNet_scores_train.pkl", "wb"))
            pickle.dump(scores_test, open("../result/iter_vs_acc/ResNet_scores_test.pkl", "wb"))
            print('Debugging pickle files have been saved.')

        torch.cuda.ipc_collect()

    def save(self, path="./model/resnet.pth"):
        """
        Save the trained model
        """
        torch.save(self.state_dict(), path)
        print("Model has been saved in {}".format(path))

    def predict(self, X):

        inputs = torch.Tensor(X.reshape((len(X), 1, 48, 48)))
        inputs = inputs.to(self.device)
        outputs = self(inputs)
        _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def score(self, X, y):
        """
        Input: matrix X of features, with n rows (samples), d columns (features)
                   X(i,j) is the j-th feature of the i-th sample
               vector y of labels, with n rows (samples), 1 column
                   y(i) is the label (+1 or -1) of the i-th sample
        Output: scalar, mean accurary on the test set [X, y]
        """
        y_hat = self.predict(X)
        return sum(y == y_hat) / len(y)

    def set_params(self, params):
        pass

if __name__ == "__main__":

    # Sample code
    fer = FER2013(filename='../data/subset3500.csv')
    img_ids = ["{:05d}".format(i) for i in range(1000)]

    import random
    random.shuffle(img_ids)
    X_train, y_train = fer.getSubset(img_ids[:800], encoding="raw_pixels")
    X_test, y_test = fer.getSubset(img_ids[800:], encoding="raw_pixels")

    # ResNet 34
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    scores_train = []
    scores_test = []
    model.train(X_train, y_train, epoch_num=2000, debug=True)
    # print("mean accuracy (train):", model.score(X_train, y_train))
    # print("mean accuracy (test):", model.score(X_test, y_test))
