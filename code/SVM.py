#!/usr/bin/env python3
import sklearn.svm
from FER2013 import FER2013
import numpy as np

class SVM(sklearn.svm.SVC):
    """C-Support Vector Classification

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
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

    # # Example code
    # fer = FER2013()

    # train_list = ["{:05d}".format(i) for i in range(20000)]
    # X_train, y_train = fer.getSubset(train_list)

    # test_list = ["{:05d}".format(i) for i in range(20000, 25000)]
    # X_test, y_test = fer.getSubset(test_list)

    # model = SVM(C=1.0, decision_function_shape='ovo', kernel='rbf', tol=0.001, verbose=True)
    # model.train(X_train, y_train)
    # print("mean accuracy:", model.score(X_test, y_test))

 if __name__=="__main__":

    # # Example code
    # fer = FER2013()

    # train_list = ["{:05d}".format(i) for i in range(20000)]
    # X_train, y_train = fer.getSubset(train_list)

    # test_list = ["{:05d}".format(i) for i in range(20000, 25000)]
    # X_test, y_test = fer.getSubset(test_list)

    # model = SVM(C=1.0, decision_function_shape='ovo', kernel='rbf', tol=0.001, verbose=True)
    # model.train(X_train, y_train)
    # print("mean accuracy:", model.score(X_test, y_test))

    fer = FER2013(filename='../data/subset3500.csv')
    from Evaluation import Evaluation
    from Visualize import Visualize

    # Hyperparameter tuning
    params = {}
    params["raw_pixels"] = {'C': [100], 'kernel': ['rbf'], 'gamma': [0.01]}    
    params["raw_pixels+landmarks"] = {'C': [10], 'kernel': ['rbf'], 'gamma': ['auto']}

    # Subset size: 10 subset data sizes
    samples_per_expression = np.arange(50, 450 + 1, 50)  # balanced sampling from all labels

    # Feature encoding
    feature_encoding_methods = ['raw_pixels+landmarks']

    # Train models by cross validation and save results
    k = 10  # k-fold Cross Validation
    for encoding in feature_encoding_methods:
        for subset_size in samples_per_expression:
            eva = Evaluation(fer, subset_size, encoding)
            model = SVM()
            path = '../result/SVM/{0}-subset{1}-new.csv'.format(encoding, subset_size)
            print(f"Saving file to {path}")
            eva.gridSearchCV(k, model, param_grid=params[encoding], save_path=path)
            print('Finished training for subset size {} of all parameters!'.format(subset_size))
        print('Finished training for feature encoding {} of all subset sizes and parameters!'.format(encoding))
    print('Finished all the training, congratulations!')

