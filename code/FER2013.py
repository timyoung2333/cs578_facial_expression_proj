#!/usr/bin/env python3
# A class to handle data reader, image encoding, etc.
from tqdm import tqdm
import numpy as np

class FER2013:

    def __init__(self, filename="../data/icml_face_data.csv"):
        """
        Load data
        """
        self.X = {}
        self.Y = {}

        with open(filename) as f:

            print("Loading FER2013 dataset from {}".format(filename))
            i = 0
            lines = f.readlines()
            for line in tqdm(lines[1:]):

                emotion, usage, pixels = line.split(",")
                pixels = np.asarray([int(j) for j in pixels.split(" ")])

                self.X[i] = self.encoding(pixels)
                self.Y[i] = emotion

                i += 1

    def encoding(self, vec):
        """
        TODO modify this method in the future to explore different encoding methods.
        Here we use one hot encoding first.
        """
        return vec

    def getX(self, img_id):
        """
        Returen feature vector
        """
        return self.X[img_id]

    def getY(self, img_id):
        """
        Return image label
        """
        return self.Y[img_id]
