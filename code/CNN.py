#!/usr/bin/env python3
from FER2013 import FER2013
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class CNN(nn.Module):
    """Convolutional neural network

    Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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

    def train(self, X, y, batch_size=256, epoch_num=100):

        dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epoch_num):  # loop over the dataset multiple times

            epoch_loss = 0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # print loss
            print("epoch: {}, loss: {:.3f}".format(epoch, epoch_loss/len(dataloader)))

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

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    fer = FER2013("../data/sample.csv")

    train_list = ["{:05d}".format(i) for i in range(100)]
    X_train, y_train = fer.getSubset(train_list, encoding="raw_pixels+landmarks")

    cnn = CNN()
    cnn.train(X_train, y_train)
    cnn.save()
    # cnn.predict(trainset)

if __name__ == "__main__":
    main()
