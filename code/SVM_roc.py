import itertools
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import os
from FER2013 import FER2013
from SVM import SVM

from Evaluation import Evaluation

from Visualize import Visualize

fer = FER2013(filename="../data/subset3500.csv")
eva = Evaluation(fer, 500)

model1 = SVM(C=100,kernel='rbf',tol=0.01, probability=True)
y_train_pred, y_train_true, y_test_pred, y_test_true, train_scores, test_scores = eva.kfoldValid(10, model1, proba=True)  # choose your own fold number
vis = Visualize(algo_name='SVM')
vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True, save_fig_path='../result/SVM/SVM_roc_rp.pdf', save_coords_path='../result/SVM/SVM_roc_rp.csv')  # can only show mean


model2 = SVM(C=10,kernel='rbf',gamma='auto',tol=0.01, probability=True)
y_train_pred, y_train_true, y_test_pred, y_test_true, train_scores, test_scores = eva.kfoldValid(10, model2, proba=True)  # choose your own fold number
vis = Visualize(algo_name='SVM')
vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True, save_fig_path='../result/SVM/SVM_roc_rp+lm.pdf', save_coords_path='../result/SVM/SVM_roc_rp+lm.csv')  # can only show mean