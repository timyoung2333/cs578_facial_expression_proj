#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import os
import random
from FER2013 import FER2013
from Perceptron import Perceptron
from Adaboost import AdaBoost
from MLP import MLP
from SVM import SVM

class Evaluation:

    def __init__(self, raw_data, subset_size): # You may change the directory to make the code work
        """
        Load data
        """
        self.fer = raw_data
        self.subDataset = self.fer.getSubDataset(subset_size)
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
            y_pred_train = model.predict(X_train)
            train_acc += model.score(X_train, y_train)
            y_pred_test = model.predict(X_test)
            test_acc += model.score(X_test, y_test)
            cf_matrix += confusion_matrix(y_test, y_pred_test)
        train_acc, test_acc = train_acc / k, test_acc / k
        for em in range(0, 7):
            cf_matrix[em] = np.divide(cf_matrix[em], np.sum(cf_matrix[em]))
        return train_acc, test_acc, cf_matrix
        
    # def bootstrappingSplit(self):
        

    # def bootstrapping(self, B):


fer = FER2013(filename="/Users/timyang/Downloads/CS578-Project-master/data/icml_face_data.csv")

ev = Evaluation(fer, 500)
model = Perceptron(tol=1e-3, random_state=0, verbose=1, n_jobs=8)
ev.testModel(10, model)




