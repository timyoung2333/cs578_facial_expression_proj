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

class CNN(nn.Module):
    """Convolutional neural network

    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, X, y, epoch_num=100):

        inputs = torch.Tensor(X.reshape((len(X), 1, 48, 48)))
        labels = torch.Tensor(y)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

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
            print("epoch: {}, loss: {:.3f}".format(epoch, loss.items()))

        print('Finished Training')

    def save(self, path="./model/cnn.pth"):
        """
        Save the trained model
        """
        torch.save(self.state_dict(), path)
        print("Model has been saved in {}".format(path))


    # def predict(self, testset):

    #     testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                              shuffle=False, num_workers=2)
    #     images, labels = data
    #     outputs = self(testset)
    #     _, predicted = torch.max(outputs.data, 1)
    #     return predicted

def main():

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./tmp', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root='./tmp', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    fer = FER2013("../data/sample.csv")

    train_list = ["{:05d}".format(i) for i in range(100)]
    X_train, y_train = fer.getSubset(train_list, encoding="raw_pixels")

    cnn = CNN()
    cnn.train(X_train, y_train)
    cnn.save()

if __name__ == "__main__":
    main()
