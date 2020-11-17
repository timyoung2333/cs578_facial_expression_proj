#!/usr/bin/env python3
import sklearn.tree
from FER2013 import FER2013
import numpy as np

class DecisionTree(sklearn.tree.DecisionTreeClassifier):
    """DecisionTree

    Reference: https://scikit-learn.org/stable/modules/tree.html
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

    def set_params(self, params):
        return super().set_params(**params)

if __name__=="__main__":

    # Example code
    fer = FER2013(filename='../data/subset3500.csv')
    param_dict = {'max_depth': np.arange(1, 11, 1)}
    from Evaluation import Evaluation
    from Visualize import Visualize
    eva = Evaluation(fer, 500)
    model = DecisionTree()
    eva.gridSearchCV(10, model, param_dict, save_path='../result/AdaBoost/DecisionTree.csv')
    best_model, best_score = eva.getBestModelAndScore()
    print('Best model param is max_depth={}, best score is {}'.format(best_model, best_score))
    test_score_dict = eva.getMeanScores()
    depth_score_dict = {params[0]: score for params, score in test_score_dict.items()}
    vis = Visualize('DecisionTree')
    vis.plotAccuracy(depth_score_dict, xlabel='Max depth of Decision Tree',
                     title='Accuracy v.s. max_depths of DecisionTreeClassifier')


