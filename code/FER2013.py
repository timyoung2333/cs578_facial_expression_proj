#!/usr/bin/env python3
# A class to handle data reader, image encoding, etc.
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib

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

        # dlib facial landmark detector: http://dlib.net/face_landmark_detection.py.html
        # self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    def getVector(self, img_id, encoding="raw_pixels"):
        """
        Input: image id
        Output: vector vec of features, with 1 row, d columns (features)
        TODO: Modify this method in the future to explore different encoding methods.
              Here we use one hot encoding first.
        """
        vec = []

        if encoding == "raw_pixels":
            vec = np.array(self.X_dic[img_id]) / 255

        if encoding == "landmarks":
            img = np.uint8(np.asarray(self.X_dic[img_id]).reshape((48, 48)) / 255)
            face_rect = dlib.rectangle(left=0, top=0, right=47, bottom=47)
            landmarks = self.dlib_predictor(img, face_rect)

            # use one hot encoding for landmarks
            vec = np.zeros((48, 48))
            for i in range(68):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                # TODO sometimes the predicted landmark may out of bound
                if landmarks and 0 <= x < 48 and 0 <= y < 48:
                    vec[y, x] = 1

        return vec

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

    def getSubset(self, id_list, encoding="raw_pixels"):
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
            X.append(self.getVector(i, encoding))
            y.append(self.getLabel(i))

        return np.array(X), np.array(y)

    def getSubsetByLabel(self, label):
        """
        Input: label, an integer that represents the label
        Output: matrix X of features, with n rows (samples), d columns (features)
                    X(i,j) is the j-th feature of the i-th sample
                vector y of labels, with n rows (samples), 1 column
                    y(i) is the label (+1 or -1) of the i-th sample
        """
        X = []
        y = []

        for i in self.X_dic:
            if self.getLabel(i) == label:
                X.append(self.getVector(i))
                y.append(self.getLabel(i))

        return np.array(X), np.array(y)

    def getImageIdByLabel(self, label):
        """
        Input: label, an integer that represents the label
        Output: id_list, the list of image ids
        """
        res = []

        for i in self.X_dic:
            if self.getLabel(i) == label:
                res.append(i)

        return res

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

    def showDistribution(self):
        """
        Show label distribution in the entire dataset
        """
        res = [len(self.getSubsetByLabel(label)[0]) for label in self.label2expression]
        plt.bar(list(self.label2expression.keys()), res)
        for x, y in zip(self.label2expression, res):
            plt.text(x, y, '%d' % y, ha='center', va='bottom')
            plt.text(x, y*0.95, '%.02f%%' % (y / sum(res) * 100), ha='center', va='top')
        plt.xticks(list(self.label2expression.keys()), self.label2expression.values())
        plt.title('Number/Percentage of Each Label in the FER-2013 Dataset')
        plt.show()

if __name__=="__main__":

    fer = FER2013("../data/sample.csv")
    fer.getVector(img_id="00010", encoding="landmarks")

    # # Example code
    # # fer = FER2013("../data/sample.csv")
    # fer = FER2013("../data/icml_face_data.csv")
    # # fer.showImage(img_id="00010")
    # fer.showDistribution()

    # # get image id for fig:fer-examples
    # print(fer.getImageIdByLabel(label=0)[:4])
    # print(fer.getImageIdByLabel(label=1)[:4])
    # print(fer.getImageIdByLabel(label=2)[:4])
    # print(fer.getImageIdByLabel(label=3)[:4])
    # print(fer.getImageIdByLabel(label=4)[:4])
    # print(fer.getImageIdByLabel(label=5)[:4])
    # print(fer.getImageIdByLabel(label=6)[:4])

