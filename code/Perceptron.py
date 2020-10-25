#!/usr/bin/env python3
from sklearn.linear_model import Perceptron
from FER2013 import FER2013

if __name__=="__main__":

    fer = FER2013()

    train_list = ["{:05d}".format(i) for i in range(20000)]
    X_train, y_train = fer.getSubset(train_list)

    test_list = ["{:05d}".format(i) for i in range(20000, 25000)]
    X_test, y_test = fer.getSubset(test_list)

    clf = Perceptron(tol=1e-3, random_state=0, verbose=1, n_jobs=8)
    clf.fit(X_train, y_train)
    print("mean accuracy:", clf.score(X_test, y_test))

