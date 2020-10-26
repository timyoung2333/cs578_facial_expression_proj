#!/usr/bin/env python3
import sklearn.linear_model
from FER2013 import FER2013

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
        return super().score(X, y)

if __name__=="__main__":

    fer = FER2013()

    train_list = ["{:05d}".format(i) for i in range(20000)]
    X_train, y_train = fer.getSubset(train_list)

    test_list = ["{:05d}".format(i) for i in range(20000, 25000)]
    X_test, y_test = fer.getSubset(test_list)

    model = Perceptron(tol=1e-3, random_state=0, verbose=1, n_jobs=8)
    model.train(X_train, y_train)
    print("mean accuracy:", model.score(X_test, y_test))

