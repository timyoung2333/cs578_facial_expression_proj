#!/usr/bin/env python3
# A class to handle data reader, image encoding, etc.
from tqdm import tqdm
import numpy as np

class FER2013:

    def __init__(self, filename="../data/icml_face_data.csv"):
        """
        Load data
        """
        self.X_dic = {}
        self.Y_dic = {}

        with open(filename) as f:

            print("Loading FER2013 dataset from {}".format(filename))
            i = 0
            lines = f.readlines()
            for line in tqdm(lines[1:]):

                emotion, usage, pixels = line.split(",")
                pixels = np.asarray([int(j) for j in pixels.split(" ")])

                self.X_dic[i] = self.encoding(pixels)
                self.Y_dic[i] = emotion

                i += 1

    def encoding(self, vec):
        """
        TODO modify this method in the future to explore different encoding methods.
        Here we use one hot encoding first.
        """
        return vec

    def getVector(self, img_id):
        """
        Returen feature vector
        """
        return self.X_dic[img_id]

    def getLabel(self, img_id):
        """
        Return image label
        """
        return self.Y_dic[img_id]

    def getTrainset(self, id_list):
        """
        Input: the IDs of images in the train set, id_list
        Output: matrix X of features, with n rows (samples), d columns (features)
                    X(i,j) is the j-th feature of the i-th sample
                vector y of labels, with n rows (samples), 1 column
                    y(i) is the label (+1 or -1) of the i-th sample
        """
        X = []
        y = []

        for i in id_list:
            X.append(self.getVector(i))
            y.append(self.getLabel(i))

        return X, y

    def getTestset(self, id_list):
        """
        Input: the IDs of images in the test set, id_list
        Output: matrix X of features, with n rows (samples), d columns (features)
                    X(i,j) is the j-th feature of the i-th sample
                vector y of labels, with n rows (samples), 1 column
                    y(i) is the label (+1 or -1) of the i-th sample
        """
        X = []
        y = []

        for i in id_list:
            X.append(self.getVector(i))
            y.append(self.getLabel(i))

        return X, y
