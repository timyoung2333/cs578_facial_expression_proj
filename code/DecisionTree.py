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
        y_hat = self.predict(X)
        return sum(y == y_hat) / len(y)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def set_params(self, params):
        return super().set_params(**params)

if __name__=="__main__":

    # Example code
    fer = FER2013(filename='../data/subset3500.csv')

    from Evaluation import Evaluation
    from Visualize import Visualize

    # eva = Evaluation(fer, 500)
    # model = DecisionTree()
    # eva.gridSearchCV(10, model, param_dict, save_path='../result/AdaBoost/DecisionTree.csv')
    # best_model, best_score = eva.getBestModelAndScore()
    # print('Best model param is max_depth={}, best score is {}'.format(best_model, best_score))
    # test_score_dict = eva.getMeanScores()
    # depth_score_dict = {params[0]: score for params, score in test_score_dict.items()}
    # vis = Visualize('DecisionTree')
    # vis.plotAccuracy(depth_score_dict, xlabel='Max depth of Decision Tree',
    #                  title='Accuracy v.s. max_depths of DecisionTreeClassifier')

    # Hyperparameter tuning
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 8, 16, 32, None],
        'min_samples_split': [2, 4, 8, 16, 32]
    }

    # Subset size: 10 subset data sizes
    samples_per_expression = np.arange(50, 500 + 1, 50)  # balanced sampling from all labels

    # Feature encoding
    feature_encoding_methods = ['raw_pixels', 'raw_pixels+landmarks']

    # Train models by cross validation and save results
    k = 10  # k-fold Cross Validation
    for encoding in feature_encoding_methods:
        for subset_size in samples_per_expression:
            eva = Evaluation(fer, subset_size, encoding)
            model = DecisionTree()
            eva.gridSearchCV(k, model, param_grid=params,
                             save_path='../result/DecisionTree/{0}-subset{1}.csv'.format(encoding, subset_size))
            print('Finished training for subset size {} of all parameters!'.format(subset_size))
        print('Finished training for feature encoding {} of all subset sizes and parameters!'.format(encoding))
    print('Finished all the training, congratulations!')

