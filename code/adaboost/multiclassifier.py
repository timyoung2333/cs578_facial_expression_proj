#!/usr/bin/env python3
import numpy as np
from itertools import combinations
from adaboost import AdaBoost
from FER2013 import FER2013


class MultiClassifier:

    def __init__(self, X, y, method='one-versus-all'):
        self.X = X
        self.y = y
        self.label_num = 0
        self.clf_num = 0
        self.training_data = []  # training samples per classifier
        self.training_label = []  # training labels per classifier
        self.classifiers = []  # models of all classifiers

        y_unique = np.unique(y)
        self.label_num = len(y_unique)
        assert self.label_num >= 2, 'The labels size should be at least 2!'

        # generate output code matrix
        # retrieve training data for each classifier
        if method == 'one-versus-all':
            self.clf_num = self.label_num
            self.output_code = np.eye(self.label_num, self.label_num) * 2 - np.ones([self.label_num, self.label_num])
            # only need to separate labels
            for m in range(self.clf_num):
                self.training_data.append(np.asarray(X))
                self.training_label.append(np.array([1 if label == m else -1 for label in self.y]))
        elif method == 'all-pairs':
            combination = list(combinations(range(self.label_num), 2))  # list of tuple with rows ids
            self.clf_num = len(combination)  # column number of output code matrix
            self.output_code = np.zeros([self.label_num, self.clf_num])
            for m in range(self.clf_num):
                # assign 1 and -1 by column
                self.output_code[combination[m][0], m] = 1
                self.output_code[combination[m][1], m] = -1
                # construct samples from X and y by combinations
                pos_y_index = np.array(self.y == combination[m][0])
                neg_y_index = np.array(self.y == combination[m][1])
                new_y = np.zeros(np.shape(self.y))
                new_y[pos_y_index] = 1
                new_y[neg_y_index] = -1
                new_y = new_y[new_y != 0]
                self.training_label.append(new_y)
                self.training_data.append(np.asarray(X[pos_y_index + neg_y_index, :]))
        else:
            print('Method {} for building multi-class classifiers is unrecognized!'.format(method))
            exit()
        print('The generalized hamming distance is ', self.getHammingDist())

    def getHammingDist(self):
        min_dist = np.inf
        combination = list(combinations(range(self.label_num), 2))
        for row_tuple in combination:
            dist = np.sum((1 - self.output_code[row_tuple[0]] * self.output_code[row_tuple[1]]) / 2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def getLabel(self, multiway_label):
        min_dist = np.inf
        best_label = 0
        for row in range(self.label_num):
            dist = np.sum((1 - self.output_code[row] * multiway_label) / 2)
            if dist < min_dist:
                min_dist = dist
                best_label = row
        return best_label

    def fit(self):
        assert len(self.training_data) > 0, 'Training data is empty!'
        assert len(self.training_data) == len(self.training_label), 'Size not match!'
        for data, label in zip(self.training_data, self.training_label):
            adaboost = AdaBoost(data, label, 50)
            classifier = adaboost.fit()
            self.classifiers.append(classifier)

    def predict(self, X_test):
        n, d = np.shape(X_test)
        adaboost = AdaBoost(self.X, self.y, 50)
        # n-by-m output code matrix with n being num of test samples and m being num of classifiers
        code = np.empty([n,0])
        for clf in self.classifiers:
            np.column_stack((code, adaboost.predict(X_test, clf)))
        predicted_multi_label = []
        for i in range(n):
            predicted_multi_label.append(self.getLabel(code[i]))
        return predicted_multi_label

    def score(self, y_test, predicted_label):
        return


def main():
    fer = FER2013(filename='data/icml_face_data.csv')

    train_list = ["{:05d}".format(i) for i in range(3500)]
    X_train, y_train = fer.getSubset(train_list)

    test_list = ["{:05d}".format(i) for i in range(3500, 4000)]
    X_test, y_test = fer.getSubset(test_list)

    my_classifier = MultiClassifier(X_train, y_train)
    my_classifier.fit()
    y_predicted = my_classifier.predict(X_test)
    my_classifier.score(y_test, y_predicted)

if __name__ == '__main__':
    main()


