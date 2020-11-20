from FER2013 import FER2013
from Evaluation import Evaluation
import numpy as np
from Perceptron import Perceptron

if __name__ == "__main__":
    fer = FER2013(filename="../data/subset3500.csv")
    # Feature encoding
    feature_encoding_methods = ['raw_pixels', 'raw_pixels+landmarks']
    # Best tuned hyper param for penalty and alpha
    param = {
            'penalty': ['l1'],
            'alpha': [0.0001],
            'max_iter': np.arange(100, 1000+1, 10)
        }

    # Train models by cross validation and save results
    k = 10  # k-fold Cross Validation
    for encoding in feature_encoding_methods:
        eva = Evaluation(fer, 500, encoding)
        model = Perceptron()
        path = '../result/Perceptron/iter.csv'
        print(f"Saving file to {path}")
        eva.gridSearchCV(k, model, param_grid=param, save_path=path, run_once=True)
        print('Finished training for feature encoding {} of all parameters!'.format(encoding))
    print('Finished all the training, congratulations!')