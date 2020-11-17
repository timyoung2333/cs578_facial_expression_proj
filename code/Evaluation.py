#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
import numpy as np
import cv2
import os
import random
from FER2013 import FER2013
from Perceptron import Perceptron
from AdaBoost import AdaBoost
from MLP import MLP
from SVM import SVM
from Visualize import Visualize
import matplotlib.pyplot as plt
import csv

class Evaluation:

    def __init__(self, raw_data, subset_size, encoding='raw_pixels'): # You may change the directory to make the code work
        """
        Load data
        """
        self.fer = raw_data
        self.subDataset = self.fer.getSubDataset(subset_size, encoding)
        self.num = subset_size
        self.label2expression = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }
        self.params_accu_dict = {}
    
    def getFeatureSize(self, mode):
        if mode == "raw":
            return 2304
    
    
    def kfoldSplit(self, k, idx):
        """
        idx represents the index of the fold of test set. 
        """
        feature_size = self.getFeatureSize("raw")
        test_size = int(self.num / k)
        train_size = self.num - test_size
        test_id = np.arange(int(idx*self.num/k), int((idx+1)*self.num/k))
        train_id = np.concatenate((np.arange(0, int(idx*self.num/k)), np.arange(int((idx+1)*self.num/k), self.num)))
        temp_train_data, temp_test_data = np.empty((0,feature_size+1)), np.empty((0,feature_size+1))
        for i in range(7):
            temp_train_data = np.append(temp_train_data, np.concatenate((self.subDataset[i][train_id], np.array([[i] * train_size]).T), axis=1), axis=0)
            temp_test_data = np.append(temp_test_data, np.concatenate((self.subDataset[i][test_id], np.array([[i] * test_size]).T), axis=1), axis=0)

        # Previously, the data types of both training data and test data are [0,0,0,...,1,1,1,...,2,2,2,...]
        # Now we shuffle them to ensure a random order.
        np.random.shuffle(temp_train_data)
        np.random.shuffle(temp_test_data)

        X_train, y_train = temp_train_data[:, 0:feature_size], temp_train_data[:, feature_size]
        X_test, y_test = temp_test_data[:, 0:feature_size], temp_test_data[:, feature_size]
        return X_train, y_train, X_test, y_test
    
    def testModel(self, k, model):
        X_train, y_train, X_test, y_test = self.kfoldSplit(k, 0)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        print(model.score(X_test, y_test))
    
    def kfoldCrossValidation(self, k, model):
        train_acc, test_acc = 0, 0
        cf_matrix = np.zeros((7, 7)) # y-axis is true label, x-axis is predicted label
        for t in range(0, k):
            X_train, y_train, X_test, y_test = self.kfoldSplit(k, t)
            model.train(X_train, y_train)
            y_pred_train = model.predict(X_train)
            train_acc += model.score(X_train, y_train)
            y_pred_test = model.predict(X_test)
            test_acc += model.score(X_test, y_test)
            cf_matrix += confusion_matrix(y_test, y_pred_test)
        train_acc, test_acc = train_acc / k, test_acc / k
        for em in range(0, 7):
            cf_matrix[em] = np.divide(cf_matrix[em], np.sum(cf_matrix[em]))
        return train_acc, test_acc, cf_matrix

    def kfoldCV(self, k, model, proba=False):
        y_train_pred = []
        y_train_true = []
        y_test_pred = []
        y_test_true = []
        scores = []
        for f in range(k):
            X_train, y_train, X_test, y_test = self.kfoldSplit(k, f)
            y_train_true.append(y_train)
            y_test_true.append(y_test)
            model.train(X_train, y_train)
            if proba:
                y_train_pred.append(model.predict_proba(X_train))
                y_test_pred.append(model.predict_proba(X_test))
            else:
                y_train_pred.append(model.predict(X_train))
                y_test_pred.append(model.predict(X_test))
            scores.append(model.score(X_test, y_test))
        return y_train_pred, y_train_true, y_test_pred, y_test_true, scores

    def gridSearchCV(self, k, model, param_grid=None, save_path=''):
        if param_grid is None:
            print('Should provide at least one param!')
            return
        # train each model with different params and get cross-validated scores
        param_grid = ParameterGrid(param_grid)
        for params in param_grid:
            model.set_params(**params)
            _, _, _, _, sc = self.kfoldCV(k, model)
            self.params_accu_dict[tuple(params.values())] = sc
        # save the params and k-fold scores as a single line into the .csv file
        if save_path != '':
            with open(save_path, 'a') as f:
                writer = csv.writer(f, dialect='excel')
                for params in self.params_accu_dict:
                    writer.writerow(list(params) + self.params_accu_dict[params])


    # def bootstrappingSplit(self):
        

    # def bootstrapping(self, B):


# fer = FER2013(filename="/Users/timyang/Downloads/CS578-Project-master/data/icml_face_data.csv")
#
# ev = Evaluation(fer, 500)
# model = Perceptron(tol=1e-3, random_state=0, verbose=1, n_jobs=8)
# ev.testModel(10, model)

if __name__ == "__main__":
    algorithms = {"Perceptron": Perceptron(Perceptron(tol=1e-3, random_state=0, verbose=1, n_jobs=8)),
                  "SVM":        SVM(C=1.0, decision_function_shape='ovo', kernel='rbf', tol=0.001, verbose=True),
                  "AdaBoost":   AdaBoost(n_estimators=100, random_state=0),
                  "MLP":        MLP(random_state=1, max_iter=300, verbose=True)}

    fer = FER2013()
    # generate confusion matrix for every algorithm
    # save results also to file
    f = open('result.csv', 'w')  # change to append later
    csv_writer = csv.writer(f, dialect='excel')
    csv_writer.writerow(list(algorithms.keys()))
    K = 10
    algo_score_mean = []
    # may need variance of accuracy
    for key in algorithms:
        model = algorithms[key]
        eva = Evaluation(fer, 500)
        y_train_pred, y_train_true, y_test_pred, y_test_true, scores = eva.kfoldCV(K, model)
        algo_score_mean.append(np.mean(scores))
        vis = Visualize(y_test_pred, y_test_true, str(key))
        vis.plotConfusionMatrix()
    csv_writer.writerow(algo_score_mean)
    f.close()

    # generate Accuracy v.s. Sample Size graph for each algorithm
    T = 10  # Total number of subset size trials
    subset_sizes = np.linspace(0, 500, T+1).astype('int')
    subset_sizes = subset_sizes[1:]
    print(subset_sizes)

    for key in algorithms:
        model = algorithms[key]
        train_acc_array = []
        test_acc_array = []
        for subset_size in subset_sizes:
            eva = Evaluation(fer, subset_size)
            train_acc, test_acc, cf_matrix = eva.kfoldCrossValidation(K, model)
            train_acc_array.append(train_acc)
            test_acc_array.append(test_acc)
        plt.figure()
        plt.plot(subset_sizes, test_acc_array)
        plt.plot(subset_sizes, train_acc_array)
        plt.scatter(subset_sizes, test_acc_array)
        plt.scatter(subset_sizes, train_acc_array)
        for i in range(len(subset_sizes)):
            plt.text(subset_sizes[i], test_acc_array[i], '%.02f' % test_acc_array[i])
            plt.text(subset_sizes[i], train_acc_array[i], '%.02f' % train_acc_array[i])
        plt.xlabel('Sample Size')
        plt.ylabel('Accuracy')
        plt.title(''.join([key, ': Accuracy of Training and Testing Set v.s. Total Dataset Size']))
        plt.legend(subset_sizes)
        plt.savefig(''.join([key, 'Accuracy.pdf']))  # one fig per model

    # different feature encoding methods and their accuracy
    # could be bar plot of each model in one fig
    # todo








