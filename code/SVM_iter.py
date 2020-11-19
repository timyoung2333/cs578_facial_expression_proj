from FER2013 import FER2013
from Evaluation import Evaluation
import numpy as np
from SVM import SVM

fer = FER2013(filename="../data/subset3500.csv")

params = {}
params["raw_pixels"] = {'C': [100], 'kernel': ['rbf'], 'gamma': [0.01], 'max_iter': np.arange(50, 500+1, 50)} 
params["raw_pixels+landmarks"] = {'C': [10], 'kernel': ['rbf'], 'gamma': ['auto'], 'max_iter': np.arange(50, 500+1, 50)}

# Subset size: 10 subset data sizes
max_iters = np.arange(50, 500 + 1, 50)  # balanced sampling from all labels

# Feature encoding
feature_encoding_methods = ['raw_pixels', 'raw_pixels+landmarks']

# Train models by cross validation and save results
k = 10  # k-fold Cross Validation
for encoding in feature_encoding_methods:
    eva = Evaluation(fer, 500, encoding)
    model = SVM()
    path = '../result/SVM/iter.csv'
    print(f"Saving file to {path}")
    eva.gridSearchCV(k, model, param_grid=params[encoding], save_path=path)
    print('Finished training for feature encoding {} of all subset sizes and parameters!'.format(encoding))
print('Finished all the training, congratulations!')