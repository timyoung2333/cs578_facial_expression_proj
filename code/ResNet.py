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
import torch.nn.init as init
import torchvision.models.resnet

class ResNet(torchvision.models.resnet.ResNet):
    """ResNet-152 Model

    Reference: https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet152
    """
    def __init__(self):
        super(ResNet, self).__init__('resnet152', torchvision.models.resnet.Bottleneck, [3, 8, 36, 3])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def train(self, X, y, epoch_num=500):

        inputs = torch.Tensor(X.reshape((len(X), 1, 48, 48)))
        labels = torch.Tensor(y)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device).long()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epoch_num):  # loop over the dataset multiple times

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print loss
            print("epoch: {}, loss: {:.3f}".format(epoch+1, loss.item()))

        print('Finished Training')

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

if __name__ == "__main__":

    # # Sample code
    fer = FER2013("../data/sample.csv")

    train_list = ["{:05d}".format(i) for i in range(80)]
    X_train, y_train = fer.getSubset(train_list, encoding="raw_pixels")

    test_list = ["{:05d}".format(i) for i in range(80, 100)]
    X_test, y_test = fer.getSubset(test_list, encoding="raw_pixels")

    model = ResNet()
    model.train(X_train, y_train, epoch_num=5000)
    print("mean accuracy (train):", model.score(X_train, y_train))
    print("mean accuracy (test):", model.score(X_test, y_test))
