#!/usr/bin/env python3
# A class to handle data reader, image encoding, etc.
from tqdm import tqdm
import numpy as np
import cv2

class FER2013:

    def __init__(self, filename="../data/icml_face_data.csv"):
        """
        Load data
        """
        self.X_dic = {}
        self.Y_dic = {}

        self.label2expression = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        with open(filename) as f:

            print("Loading FER2013 dataset from {}".format(filename))
            i = 0
            lines = f.readlines()
            for line in tqdm(lines[1:]):

                emotion, usage, pixels = line.split(",")
                emotion = int(emotion)
                pixels = np.asarray([int(j) for j in pixels.split(" ")])

                img_id = "{:05d}".format(i)
                self.X_dic[img_id] = pixels
                self.Y_dic[img_id] = emotion

                i += 1

    def getVector(self, img_id):
        """
        Input: image id
        Output: vector vec of features, with 1 row, d columns (features)
        TODO: Modify this method in the future to explore different encoding methods.
              Here we use one hot encoding first.
        """
        return np.array(self.X_dic[img_id]) / 255

    def getLabel(self, img_id):
        """
        Input: img_id
        Output: label
        """
        return self.Y_dic[img_id]

    def getExpression(self, img_id):
        """
        Input: img_id
        Output: expression
        """
        label = self.Y_dic[img_id]
        return self.label2expression[label]

    def getSubset(self, id_list):
        """
        Input: id_list, the list of image ids
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

        return np.array(X), np.array(y)

    def showImage(self, img_id):
        """
        Input: image id
        Output: none
        Show image on the screen.
        """
        img = np.asarray(self.X_dic[img_id]).reshape((48, 48)) / 255
        text = "image: {}, label: {} ({})".format(img_id, self.getLabel(img_id), self.getExpression(img_id))
        cv2.imshow(text, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=="__main__":

    # Example code
    fer = FER2013("../data/sample.csv")
    fer.showImage(img_id="00010")

