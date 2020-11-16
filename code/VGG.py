#!/usr/bin/env python3
from FER2013 import FER2013
import numpy as np
from tqdm import tqdm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

class VGG(nn.Module):
    """VGG-16 Model

    Reference: https://github.com/pytorch/vision.git
    """
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self.make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

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

    def save(self, path="./model/vgg.pth"):
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
    fer = FER2013()

    train_list = ["{:05d}".format(i) for i in range(1000)]
    X_train, y_train = fer.getSubset(train_list, encoding="raw_pixels")

    test_list = ["{:05d}".format(i) for i in range(8000, 1000)]
    X_test, y_test = fer.getSubset(test_list, encoding="raw_pixels")

    model = VGG()
    model.train(X_train, y_train)
    print("mean accuracy (train):", model.score(X_train, y_train))
    print("mean accuracy (test):", model.score(X_test, y_test))
