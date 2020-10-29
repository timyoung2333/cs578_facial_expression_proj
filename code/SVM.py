import sklearn.svm as svm
from FER2013 import FER2013

if __name__=="__main__":

    # Example code
    fer = FER2013()

    train_list = ["{:05d}".format(i) for i in range(20000)]
    X_train, y_train = fer.getSubset(train_list)

    test_list = ["{:05d}".format(i) for i in range(20000, 25000)]
    X_test, y_test = fer.getSubset(test_list)

    model = svm.SVC(C=1.0, decision_function_shape='ovo', kernel='rbf', tol=0.001)
    model.fit(X_train, y_train)
    print("mean accuracy:", model.score(X_test, y_test))