#!/usr/bin/env python3
# A class to handle data reader, image encoding, etc.
from tqdm import tqdm
import numpy as np

class FER2013:

    def __init__(filename="../data/icml_face_data.csv"):
        """
        Load data
        """
        self.X = {}
        self.Y = {}

        with open(filename) as f:

            i = 0
            lines = f.readlines()
            for line in tqdm(lines[1:]):

                emotion, usage, pixels = line.split(",")
                pixels = np.asarray([int(i) for i in pixels.split(" ")])

                self.X[i] = pixels
                self.Y[i] = emotion

                i += 1

    def getX(img_id):
        """
        Returen feature vector (one hot encoding by default)
        """
        return self.X[img_id]

    def getY(img_id):
        """
        Return image label
        """
        return self.Y[img_id]
