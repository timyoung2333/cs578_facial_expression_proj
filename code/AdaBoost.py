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
        Output: scalar, mean accurary on the test set [X, y]
        """
        return super().score(X, y)

    def staged_score(self, X, y):
        return super().staged_score(X, y)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def decision_func(self, X):
        return super().decision_function(X)

if __name__=="__main__":

    fer = FER2013()
    from Evaluation import Evaluation
    from Visualize import Visualize

    # example code to fast test ROC curve for a single model
    # eva = Evaluation(fer, 500)
    # model = AdaBoost()
    # y_train_pred, y_train_true, y_test_pred, y_test_true, scores = eva.kfoldCV(10, model, proba=True)
    # vis = Visualize(algo_name='AdaBoost')
    # vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True)
    # exit(0)

    estimator_sizes = np.arange(50, 500+1, 25)  # 50, 75, 100, ..., 500
    base_estimators = {"DecisionTreeClassifier_MaxDepth1": sklearn.tree.DecisionTreeClassifier(max_depth=1),
                       "DecisionTreeClassifier_MaxDepth3": sklearn.tree.DecisionTreeClassifier(max_depth=3),
                       "BernoulliNB": sklearn.naive_bayes.BernoulliNB(),
                       "MultinomialNB": sklearn.naive_bayes.MultinomialNB(),
                       "ExtraTreeClassifier": sklearn.tree.ExtraTreeClassifier()
                       }

    samples_per_expression = 500  # balanced sampling from all labels
    k = 10  # k-fold Cross Validation
    # traverse all types of weak learners and all max number of weak learners for each type
    for key in base_estimators:
        for estimator_size in estimator_sizes:
            eva = Evaluation(fer, samples_per_expression)
            model = AdaBoost(base_estimator=base_estimators[key], n_estimators=estimator_size, random_state=0)
            y_train_pred, y_train_true, y_test_pred, y_test_true, scores = eva.kfoldCV(k, model)
            # save all accuracy scores
            f = open('../result/AdaBoost/adaboost.csv', 'a')
            csv_writer = csv.writer(f, dialect='excel')
            csv_writer.writerow([key, estimator_size] + scores)
            f.close()

            vis = Visualize(y_test_pred, y_test_true, 'AdaBoost')
            path = '../result/AdaBoost/AdaBoostConfMat' + str(key) + '_Iteration' + str(estimator_size)
            # save all confusion matrix as pdf
            vis.plotConfusionMatrix(save_path=path + '.pdf')
            # save all confusion matrices to plot ROC, used after plot conf mat method
            vis.saveConfMat('../result/AdaBoost/AdaBoostConfMat.csv', str(key), str(estimator_size))

