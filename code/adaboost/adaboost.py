#!/usr/bin/env python3
import numpy as np


class AdaBoost:

    def __init__(self, X, y, iterations):
        """
        :param X: training data matrix with rows being sample feature vector
        :param y: training data vector with labels of all samples
        """
        self.X = X
        self.y = y
        self.iterations = iterations
        if iterations > len(X[0]):
            print('Iteration num {} is over feature dimension {}'.format(iterations, len(X[0])))

    def decisionStump(self, feature_vec, mu, do_flip):
        """
        :param feature_vec: feature vector of one sample
        :param mu: threshold value of the stump
        :param do_flip: if true, feature less than mu is 1, greater or equal to mu is -1;
                        otherwise reversed
        :return: the sign vector of the sample
        """
        result = np.ones(feature_vec.shape, dtype='float')
        if do_flip:
            result[result >= mu] = -1.0
        else:
            result[result <= mu] = -1.0
        return result

    def getBestStump(self, weights, feature_pieces):
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
                    decision_vec = self.decisionStump(self.X[:, j], mu, do_flip)
                    agreement_vec = decision_vec * (-self.y)  # agreement with the opposite label
                    dJ = np.dot(weights, agreement_vec)  # inner product of sample weights and decision agreement
                    # update the best stump info
                    if dJ < best_dJ:
                        best_dJ = dJ
                        best_decision = decision_vec
                        best_stump['feature_id'] = j
                        best_stump['mu'] = mu
                        best_stump['do_flip'] = do_flip

        return best_stump, best_dJ, best_decision

    def fit(self, loss_function='exponential'):
        feature_pieces = 10
        n, d = np.shape(self.X)
        weights = np.ones(n) / n
        weak_stumps = []  # keep all weak stumps here
        accumulated_decision = np.zeros(n)  # keep track of the ensemble error for debug use

        for iter in range(self.iterations):
            stump, dJ, decision = self.getBestStump(weights, feature_pieces)
            if dJ >= 0:
                print('AdaBoost gets positive deltaJ on {:d}-th iteration'.format(iter))
                break

            # calculate alpha that minimizes J
            dJ = max(dJ, -0.99)  # restrict deltaJ for log calculation below
            alpha = 0.5 * np.log((1 - dJ) / (1 + dJ))
            stump['alpha'] = alpha
            weak_stumps.append(stump)

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
        print('Generated {} weak classifiers.'.format(self.iterations))
        return weak_stumps

    def predict(self, X_test, weak_stumps):
        predicted_labels = []
        for x in X_test:
            # weights of all stumps, size = iterations
            all_alpha = [stump['alpha'] for stump in weak_stumps]
            # feature ids of all stumps, size = iteration
            all_feature_ids = [stump['feature_id'] for stump in weak_stumps]
            prediction = [self.decisionStump(x[feature_id], weak_stumps[stump_id]['mu'], weak_stumps[stump_id]['do_flip']) for
                          feature_id, stump_id in zip(all_feature_ids, range(len(x)))]
            predicted_labels.append(np.sign(np.dot(all_alpha, prediction))[0])
        return predicted_labels
