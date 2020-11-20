#!/usr/bin/env python3
# A class to handle data reader, image encoding, etc.
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib
from collections import defaultdict
import csv


class FER2013:

    def __init__(self, filename="../data/icml_face_data.csv"):
        """
        Load data
        """
        self.X_dic = {}
        self.Y_dic = {}
        self.label_dic = defaultdict(list)

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
                self.label_dic[emotion].append(img_id)

                i += 1

        # dlib facial landmark detector: http://dlib.net/face_landmark_detection.py.html
        self.dlib_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

        # Haar Cascade Face Detector: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
        self.faceCascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

    def getVector(self, img_id, encoding="raw_pixels"):
        """
        Input: image id
        Output: vector vec of features, with 1 row, d columns (features)
        """

        # ============================================
        # raw pixel
        vec_raw_pixels = np.array(self.X_dic[img_id]) / 255

        # ============================================
        # facial landmark
        img = np.uint8(np.asarray(self.X_dic[img_id]).reshape((48, 48)))
        face_rect = dlib.rectangle(left=0, top=0, right=47, bottom=47)
        landmarks = self.dlib_predictor(img, face_rect)

        # extract features using the method from this paper: https://arxiv.org/pdf/1812.04510.pdf
        vec_landmarks = []
        key_points = [[37, 40], [38, 42], [43, 46], [44, 48], [18, 22], [23, 27], [49, 55], [18, 41], [48, 27], [34, 67], [42, 49], [47, 55]]

        for i, j in key_points:
            p1 = np.array([landmarks.part(i).x, landmarks.part(i).y])
            p2 = np.array([landmarks.part(j).x, landmarks.part(j).y])
            dis = np.linalg.norm(p1 - p2)
            vec_landmarks.append(dis)

        vec_landmarks = np.array(vec_landmarks)

        # ============================================
        # Haar face detection
        img = np.uint8(np.asarray(self.X_dic[img_id]).reshape((48, 48)) / 255)
        faces = self.faceCascade.detectMultiScale(img)

        vec_haar = np.array([0, 0, 47, 47])
        if len(faces) > 0:
            # TODO always fail because the input image is too small
            face = faces[0]
            vec_haar[0] = face.left()
            vec_haar[1] = face.top()
            vec_haar[2] = face.right() - face.left()
            vec_haar[3] = face.bottom() - face.top()

        if encoding == "raw_pixels":
            return vec_raw_pixels

        if encoding == "landmarks":
            return vec_landmarks

        if encoding == "Haar":
            return vec_haar

        if encoding == "raw_pixels+landmarks":
            return np.concatenate((vec_raw_pixels, vec_landmarks), axis=None)

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

    def getSubDataset(self, num, encoding, method=None):
        """
        Input: the number of images from each class (equal num)
        Output: a sub dataset dictionary
        """
        subDataset = {}
        for k, v in self.label_dic.items():
            v = np.array(v)
            if method == "random":
                random.shuffle(v)
            subDataset[k] = self.getSubset(v[0:num], encoding=encoding)[0]
        return subDataset

    def showImage(self, img_id, showLandmark=False):
        """
        Input: image id
        Output: none
        Show image on the screen.
        """
        img = np.uint8(np.asarray(self.X_dic[img_id]).reshape((48, 48)))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        text = "image: {}, label: {} ({})".format(img_id, self.getLabel(img_id), self.getExpression(img_id))

        if showLandmark:
            face_rect = dlib.rectangle(left=0, top=0, right=47, bottom=47)
            landmarks = self.dlib_predictor(img, face_rect)

            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            cv2.imwrite("{}_landmarks.png".format(img_id), img)

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
            plt.text(x, y * 0.95, '%.02f%%' % (y / sum(res) * 100), ha='center', va='top')
        plt.xticks(list(self.label2expression.keys()), self.label2expression.values())
        plt.title('Number/Percentage of Each Label in the FER-2013 Dataset')
        plt.show()

    def saveSubset(self, num_per_label, path):
        with open(path, 'w') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(['emotion', 'Usage', 'pixels'])
            for k, v in self.label_dic.items():
                if len(v) < num_per_label:
                    print('Feature {} has samples {} less than requested number {}!'.format(k, len(v), num_per_label))
                    return
                # Currently save the first portion of all samples
                for img_id in v[0:num_per_label]:
                    writer.writerow(list((k, 'Training')) + [' '.join(str(i) for i in self.X_dic[img_id])])
        print('Subset saved successfully!')


if __name__ == "__main__":
    # fer = FER2013("../data/sample.csv")
    # fer.getVector(img_id="00000", encoding="landmarks")
    # fer.getVector(img_id="00000", encoding="Haar")
    # fer.getVector(img_id="00000", encoding="raw_pixels+landmarks")

    # Example code
    # fer = FER2013("../data/sample.csv")
    fer = FER2013("../data/icml_face_data.csv")
    # fer.saveSubset(500, '../data/subset3500.csv')
    fer.showImage(img_id="{:05d}".format(0), showLandmark=True)
    # fer.showDistribution()

    # # get image id for fig:fer-examples
    # print(fer.getImageIdByLabel(label=0)[:4])
    # print(fer.getImageIdByLabel(label=1)[:4])
    # print(fer.getImageIdByLabel(label=2)[:4])
    # print(fer.getImageIdByLabel(label=3)[:4])
    # print(fer.getImageIdByLabel(label=4)[:4])
    # print(fer.getImageIdByLabel(label=5)[:4])
    # print(fer.getImageIdByLabel(label=6)[:4])
