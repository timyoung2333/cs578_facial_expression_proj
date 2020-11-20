#!/usr/bin/env python3
from FER2013 import FER2013
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle

class CNN(nn.Module):
    """Convolutional neural network

    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, X, y, epoch_num=2000, debug=False):

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
            pickle.dump(scores_train, open("../result/iter_vs_acc/CNN_scores_train.pkl", "wb"))
            pickle.dump(scores_test, open("../result/iter_vs_acc/CNN_scores_test.pkl", "wb"))
            print('Debugging pickle files have been saved.')

    def save(self, path="./model/cnn.pth"):
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
    img_ids = ["{:05d}".format(i) for i in range(3500)]

    import random
    random.shuffle(img_ids)
    X_train, y_train = fer.getSubset(img_ids[:3000], encoding="raw_pixels")
    X_test, y_test = fer.getSubset(img_ids[3000:], encoding="raw_pixels")

    model = CNN()
    scores_train = []
    scores_test = []
    model.train(X_train, y_train, epoch_num=10000, debug=True)
    # print("mean accuracy (train):", model.score(X_train, y_train))
    # print("mean accuracy (test):", model.score(X_test, y_test))
