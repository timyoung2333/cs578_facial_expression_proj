#!/usr/bin/env python3
import numpy as np
from configparser import ConfigParser
from itertools import combinations
from tqdm import tqdm
from FER2013 import FER2013


class AdaBoostBinary:

    def __init__(self, X, y):
        """
        :param X: training data matrix with rows being sample feature vector
        :param y: training data vector with labels of all samples
        """
        self.X = X
        self.y = y
        self.weak_stumps = []
        self.cf = ConfigParser()
        self.cf.read("parameters.conf")

    @staticmethod
    def decisionStump(feature_vec, mu, do_flip):
        """
        :param feature_vec: feature vector of one sample
        :param mu: threshold value of the stump
        :param do_flip: if true, feature less than mu is 1, greater or equal to mu is -1;
                        otherwise reversed
        :return: the sign vector of the sample
        """
        result = np.ones(feature_vec.shape, dtype='float')
        if do_flip:
            result[feature_vec >= mu] = -1.0
        else:
            result[feature_vec <= mu] = -1.0
        return result

    def __getBestStump(self, weights, feature_pieces):
        n, d = np.shape(self.X)
        best_stump = {}  # structure of the theta vector(map) in lecture
        best_dJ = np.inf  # minimum training loss
        best_decision = np.ones(n)  # prediction vector of decision stump

        # traverse all the features
        for j in range(d):
            min_feature = np.min(self.X[:, j])
            max_feature = np.max(self.X[:, j])
            # traverse all the intermediate feature values of certain feature
            for mu in np.linspace(min_feature, max_feature, feature_pieces):
                # traverse double ways
                for do_flip in [0, 1]:
                    # print('j: {}, mu: {}, do_flip: {}'.format(j, mu, do_flip))
                    decision_vec = self.decisionStump(self.X[:, j], mu, do_flip)
                    agreement_vec = decision_vec * (-self.y)  # agreement with the opposite label
                    # print('agreement vector is ', agreement_vec)
                    dJ = np.dot(weights, agreement_vec)  # inner product of sample weights and decision agreement
                    # update the best stump info
                    if dJ < best_dJ:
                        best_dJ = dJ
                        best_decision = decision_vec
                        best_stump['feature_id'] = j
                        best_stump['mu'] = mu
                        best_stump['do_flip'] = do_flip
        # print('Best feature id {}, best feature value {}, best dJ {}'.
        #       format(best_stump['feature_id'], best_stump['mu'], best_dJ))
        return best_stump, best_dJ, best_decision

    def train(self, loss_function='exponential'):
        n, d = np.shape(self.X)
        weights = np.ones(n) / n
        accumulated_decision = np.zeros(n)  # keep track of the ensemble error for debug use

        iterations = int(self.cf.get("AdaBoost", "iterations"))
        if iterations > len(self.X[0]):
            print('Iteration num {} is over feature dimension {}'.format(iterations, len(self.X[0])))
        for iter in tqdm(range(iterations)):
            stump, dJ, decision = self.__getBestStump(weights, int(self.cf.get("AdaBoost", "fixed_feature_pieces")))
            if dJ >= 0:
                print('AdaBoost gets positive deltaJ on {:d}-th iteration'.format(iter))
                break

            # calculate alpha that minimizes J
            dJ = max(dJ, -0.99)  # restrict deltaJ for log calculation below
            alpha = 0.5 * np.log((1 - dJ) / (1 + dJ))
            stump['alpha'] = alpha
            self.weak_stumps.append(stump)

            # update weights of samples
            # can support more loss functions when needed
            exp_weighted_error = np.exp(-alpha * self.y * decision)
            if loss_function == 'exponential':
                weights = weights * exp_weighted_error
            elif loss_function == 'logistic':
                weights = 1 / (1 + (1 / weights - 1) * exp_weighted_error)
            else:
                print('Unrecognized loss function {}'.format(loss_function))
                break
            weights = weights / sum(weights)  # normalize weights
        print('Generated {} weak classifiers.'.format(iterations))

    @staticmethod
    def predict(X_test, weak_stumps):
        predicted_labels = []
        for x in X_test:
            # weights of all stumps, size = iterations
            all_alpha = [stump['alpha'] for stump in weak_stumps]
            # feature ids of all stumps, size = iteration
            all_feature_ids = [stump['feature_id'] for stump in weak_stumps]
            prediction = [
                AdaBoostBinary.decisionStump(x[feature_id], weak_stumps[stump_id]['mu'], weak_stumps[stump_id]['do_flip']) for
                feature_id, stump_id in zip(all_feature_ids, range(len(x)))]
            predicted_labels.append(np.sign(np.dot(all_alpha, prediction)))
        print('Predicted labels: {}'.format(predicted_labels))
        return predicted_labels

    def score(self, X_test, y_test):
        y_predicted = AdaBoostBinary.predict(X_test, self.weak_stumps)
        return sum(y_test == y_predicted) / len(y_test)


class AdaBoostMulti:

    def __init__(self):
        self.label_num = 0
        self.clf_num = 0
        self.training_data = []  # training samples per classifier
        self.training_label = []  # training labels per classifier
        self.classifiers = []  # models of all classifiers
        self.cf = ConfigParser()
        self.cf.read('parameters.conf')

    def __initEssentials(self, X, y):
        self.X = X
        self.y = y
        self.label_num = len(np.unique(y))
        assert self.label_num >= 2, 'The labels size should be at least 2!'
        print('Total training data -> samples: {}, features: {}, labels {}.'.
              format(len(self.X), len(self.X[0]), self.label_num))

        # generate output code matrix
        # retrieve training data for each classifier
        method = self.cf.get("AdaBoost", "multi_classification_method")
        if method == "one-versus-all":
            self.clf_num = self.label_num
            self.output_code = np.eye(self.label_num, self.label_num) * 2 - np.ones([self.label_num, self.label_num])
            # only need to separate labels
            for m in range(self.clf_num):
                self.training_data.append(np.asarray(X))
                self.training_label.append(np.array([1 if label == m else -1 for label in self.y]))
        elif method == "all-pairs":
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
        print('The generalized hamming distance for {} is {}.'.format(method, self.getHammingDist()))

    def getHammingDist(self):
        min_dist = np.inf
        combination = list(combinations(range(self.label_num), 2))
        for row_tuple in combination:
            dist = np.sum((1 - self.output_code[row_tuple[0]] * self.output_code[row_tuple[1]]) / 2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def __getLabel(self, multiway_label):
        min_dist = np.inf
        best_label = 0
        for row in range(self.label_num):
            dist = np.sum((1 - self.output_code[row] * multiway_label) / 2)
            if dist < min_dist:
                min_dist = dist
                best_label = row
        return best_label

    def train(self, X, y):
        self.__initEssentials(X, y)
        assert len(self.training_data) > 0, 'Training data is empty!'
        assert len(self.training_data) == len(self.training_label), 'Size not match!'
        for data, label in tqdm(zip(self.training_data, self.training_label)):
            print('Binary training data -> samples: {}, features: {}'.
                  format(len(data), len(data[0]), len(label)))
            adaboost = AdaBoostBinary(data, label)
            adaboost.train(self.cf.get("AdaBoost", "loss_function"))
            self.classifiers.append(adaboost.weak_stumps)

    def predict(self, X_test):
        assert len(self.classifiers) > 0, 'No classifiers yet, run fit first!'
        n, d = np.shape(X_test)
        # n-by-m output code matrix with n being num of test samples and m being num of classifiers
        code = np.empty([n, 0])
        for clf in self.classifiers:
            code = np.column_stack((code, AdaBoostBinary.predict(X_test, clf)))
        # n-by-1 predicted labels of all n test samples
        predicted_multi_label = []
        for i in range(n):
            predicted_multi_label.append(self.__getLabel(code[i]))
        return predicted_multi_label

    def score(self, X_test, y_test):
        y_predicted = self.predict(X_test)
        assert len(y_test) == len(y_predicted), 'y size not match!'
        print('y test is {}'.format(y_test))
        print('y predict is {}'.format(y_predicted))
        return sum(y_test == y_predicted) / len(y_test)


def main():
    fer = FER2013(filename='../data/icml_face_data.csv')

    train_list = ["{:05d}".format(i) for i in range(20000)]
    X_train, y_train = fer.getSubset(train_list)

    test_list = ["{:05d}".format(i) for i in range(20000, 25000)]
    X_test, y_test = fer.getSubset(test_list)

    model = AdaBoostMulti()
    model.train(X_train, y_train)
    print('Mean accuracy: {}'.format(model.score(X_test, y_test)))


if __name__ == '__main__':
    main()
