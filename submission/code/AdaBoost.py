#!/usr/bin/env python3
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.tree
from FER2013 import FER2013
import pickle
import csv
import numpy as np


class AdaBoost(sklearn.ensemble.AdaBoostClassifier):
    """An AdaBoost classifier.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    """

    def train(self, X, y):
        """
        Input: matrix X of features, with n rows (samples), d columns (features)
                   X(i,j) is the j-th feature of the i-th sample
               vector y of labels, with n rows (samples), 1 column
                   y(i) is the label (+1 or -1) of the i-th sample
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Input: matrix X of features, with n rows (samples), d columns (features)
                   X(i,j) is the j-th feature of the i-th sample
        Output: vector y of labels, with n rows (samples), 1 column
                    y(i) is the label (+1 or -1) of the i-th sample
        """
        return super().predict(X)

    def score(self, X, y):
        """
        Input: matrix X of features, with n rows (samples), d columns (features)
                   X(i,j) is the j-th feature of the i-th sample
               vector y of labels, with n rows (samples), 1 column
                   y(i) is the label (+1 or -1) of the i-th sample
        Output: scalar, mean accuracy on the test set [X, y]
        """
        y_hat = self.predict(X)
        return sum(y == y_hat) / len(y)

    def staged_score(self, X, y):
        return super().staged_score(X, y)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def decision_func(self, X):
        return super().decision_function(X)

    def set_params(self, params):
        return super().set_params(**params)


if __name__ == "__main__":

    fer = FER2013(filename='../data/subset3500.csv')
    from Evaluation import Evaluation
    from Visualize import Visualize

    # example code to fast test ROC curve and confusion matrix for a single model
    # test_conf_mat = True
    # test_roc_curve = True
    # eva = Evaluation(fer, 100)
    # model = AdaBoost()
    # if test_roc_curve:
    #     y_train_pred, y_train_true, y_test_pred, y_test_true, scores = eva.kfoldCV(10, model, proba=True)
    #     vis = Visualize(algo_name='AdaBoost')
    #     vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True,
    #                        save_fig_path='../result/AdaBoost/roc_curve.pdf',
    #                        save_coords_path='../result/AdaBoost/roc_coords.csv')
    # if test_conf_mat:
    #     y_train_pred, y_train_true, y_test_pred, y_test_true, scores = eva.kfoldCV(10, model, proba=False)
    #     vis = Visualize(algo_name='AdaBoost')
    #     vis.set_y(y_test_pred, y_test_true)
    #     vis.plotConfusionMatrix(save_path='../result/AdaBoost/conf_mat.pdf')
    #     vis.saveConfMat('../result/AdaBoost/AdaBoostConfMat.csv', 'DecisionTreeClassifier_Depth1', 50)
    # exit(0)

    # May just use DecisionTreeClassifier later
    base_estimators = {"DecisionTreeClassifier_MaxDepth1": sklearn.tree.DecisionTreeClassifier(max_depth=1),
                       "BernoulliNB": sklearn.naive_bayes.BernoulliNB(),
                       "MultinomialNB": sklearn.naive_bayes.MultinomialNB(),
                       "ExtraTreeClassifier": sklearn.tree.ExtraTreeClassifier()
                       }

    # Parameter tuning: 60 models in total for each dataset
    estimator_sizes = np.arange(50, 500 + 1, 50)  # 50, 100, 150, ..., 500
    learning_rates = np.logspace(-3, 1, 5, base=10)  # 0.001, 0.01, ..., 10
    params = {'n_estimators': estimator_sizes, 'learning_rate': learning_rates}

    # # Subset size: 10 subset data sizes
    # samples_per_expression = np.arange(50, 500 + 1, 50)  # balanced sampling from all labels

    # run with mutiple terminals
    import sys
    if len(sys.argv) != 2:
        print("Usage ./script.py <subset_size>")
        exit(1)
    samples_per_expression = [int(sys.argv[1])]

    # Feature encoding
    feature_encoding_methods = ['raw_pixels', 'raw_pixels+landmarks']

    # Train models by cross validation and save results
    k = 10  # k-fold Cross Validation
    for encoding in feature_encoding_methods:
        for subset_size in samples_per_expression:
            eva = Evaluation(fer, subset_size, encoding)
            model = AdaBoost()
            eva.gridSearchCV(k, model, param_grid=params,
                             save_path='../result/AdaBoost/{0}-subset{1}.csv'.format(encoding, subset_size))
            print('Finished training for subset size {} of all parameters!'.format(subset_size))
        print('Finished training for feature encoding {} of all subset sizes and parameters!'.format(encoding))
    print('Finished all the training, congratulations!')

    # Visualization
    # todo load and display
