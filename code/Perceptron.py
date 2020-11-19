#!/usr/bin/env python3
import sklearn.linear_model
from FER2013 import FER2013
import numpy as np

class Perceptron(sklearn.linear_model.Perceptron):
    """Perceptron

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
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

    def set_params(self, params):
        return super().set_params(**params)

if __name__=="__main__":

    fer = FER2013(filename='../data/subset3500.csv')
    from Evaluation import Evaluation
    from Visualize import Visualize

    # Hyperparameter tuning
    params = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 500, 1000]
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
            model = Perceptron()
            eva.gridSearchCV(k, model, param_grid=params,
                             save_path='../result/Perceptron/{0}-subset{1}.csv'.format(encoding, subset_size))
            print('Finished training for subset size {} of all parameters!'.format(subset_size))
        print('Finished training for feature encoding {} of all subset sizes and parameters!'.format(encoding))
    print('Finished all the training, congratulations!')

    # train_list = ["{:05d}".format(i) for i in range(800)]
    # X_train, y_train = fer.getSubset(train_list, encoding="raw_pixels+landmarks")

    # test_list = ["{:05d}".format(i) for i in range(800, 1000)]
    # X_test, y_test = fer.getSubset(test_list, encoding="raw_pixels+landmarks")

    # model = Perceptron(tol=1e-3, random_state=0, verbose=1, n_jobs=8)
    # model.train(X_train, y_train)
    # print("mean accuracy:", model.score(X_test, y_test))

