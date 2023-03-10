#!/usr/bin/env python3
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import os
import re
import pickle

label2expression = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


class Visualize:

    def __init__(self, algo_name=""):
        """
        Class that can generate confusion matrix, ROC curve plot and other model-accuracy-related plots
        :param algo_name: Algorithm to visualize, if not specified, would be Anonymous
        """
        if algo_name == '':
            self.algo_name = 'Anonymous'
        else:
            self.algo_name = algo_name
        self.labels = list(label2expression.keys())
        self.expressions = list(label2expression.values())
        self.label_num = len(self.labels)
        self.conf_mat = np.zeros([self.label_num, self.label_num])
        self.__conf_mats = []
        self.y_predicts = []
        self.y_tests = []
        self.num = 0

    def set_y(self, y_predicts, y_tests):
        self.y_predicts = y_predicts
        self.y_tests = y_tests
        assert len(y_predicts) == len(y_tests), "y test and y true size not match!"
        self.num = len(y_predicts)
        if self.num <= 0:
            print('y_predicts has size 0!')
        else:
            self.__genConfusionMatrix()
        print("Get {} Estimation(s).".format(self.num))

    def __genConfusionMatrix(self):
        for i in range(self.num):
            mat = confusion_matrix(self.y_tests[i], self.y_predicts[i])
            self.conf_mat = self.conf_mat + mat
        # use the mean value for each entry
        self.conf_mat = self.conf_mat / self.num

    def __normalize(self, by_row=True):
        ax = 1 if by_row else 0
        self.conf_mat = self.conf_mat.astype('float') / self.conf_mat.sum(axis=ax)[:, np.newaxis]

    def plotConfusionMatrix(self, conf_matrix=None, norm=True, save_path=''):
        if conf_matrix is not None:
            self.conf_mat = conf_matrix
        if norm:
            self.__normalize()
        print(self.conf_mat)

        plt.figure()
        plt.imshow(self.conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(self.labels, self.expressions, rotation=45)
        plt.yticks(self.labels, self.expressions)
        thresh = self.conf_mat.max() / 2.
        for i, j in itertools.product(range(self.label_num), range(self.label_num)):
            plt.text(j, i, '%.02f' % self.conf_mat[i, j],
                     horizontalalignment="center",
                     color="white" if self.conf_mat[i, j] > thresh else "black")
        plt.tight_layout()
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path != '':
            plt.savefig(save_path)
        else:
            plt.show()

    def saveConfMat(self, path, param1, param2):
        with open(path, 'a') as f:
            csv_writer = csv.writer(f, dialect='excel')
            csv_writer.writerow([param1, param2])
            csv_writer.writerows(np.round(self.conf_mat, 2))

    def loadConfMats(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            while True:
                try:
                    # param1 and param2 are 2 types of params used in the algorithm
                    param1, param2 = next(reader)
                    mat = []
                    for i in range(self.label_num):
                        mat.append([float(j) for j in next(reader)])
                    assert len(mat) == len(mat[0]) == self.label_num, 'conf mat read from csv has wrong dimension!'
                    self.__conf_mats.append((param1, param2, mat))
                except StopIteration:
                    print('Read to eof, have {} confusion matrices!'.format(len(self.__conf_mats)))
                    break

    def plotRocCurve(self, y_true, y_pred_prob, show_all=True):
        """
        Used to plot ROC curve for single model's single prediction, can show curves of all labels, as well as their
        interpolated mean
        :param y_true: n-by-1 array of multi-labels with n as number of y_test samples
        :param y_pred_prob: n-by-d array of predicted probability of a binary label with n as number of y_test samples,
                            d as labels number
        :param show_all: True if want to show ROC curves of all labels, False if only want to show the mean ROC curve
        :return: None
        """
        y_trues = label_binarize(y_true, classes=self.labels)
        assert y_true.shape[1] == self.label_num, 'Classes after binarization does not equal to {}!'.format(
            self.label_num)
        assert y_true.shape == y_pred_prob.shape, 'Multi-class matrix size not match!'
        fprs, tprs = self.rocCurves(y_trues, y_pred_prob)
        plt.figure()
        # plot ROC curve for each label
        if show_all:
            label = 0
            for fpr, tpr in zip(fprs, tprs):
                auc = self.aucByRate(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(label, auc))
                label += 1
        # get interpolated mean ROC curve and plot
        all_fpr, mean_tpr = self.interpMeanRocCurve(fprs, tprs)
        auc = self.aucByRate(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label='macro-average ROC curve (area = {0:0.2f})'.format(auc),
                 color='navy', linestyle=':', linewidth=4)
        # plot random guess line for binary prediction
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'Receiver Operating Characteristic (ROC) of {}-label Multi-class Prediction of {}'.format(self.label_num,
                                                                                                      self.algo_name))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def plotCVRocCurve(self, y_trues, y_scores, show_all=True, save_fig_path='', save_coords_path=''):
        """
        Used to plot ROC curve of single model's multiple predictions, like cross validation.
        Example Usage:
            eva = Evaluation(fer, 100)  # 100 samples per label
            model = AdaBoost()  # choose your own model
            y_train_pred, y_train_true, y_test_pred, y_test_true, scores = eva.kfoldCV(5, model, proba=True)  # choose your own fold number
            vis = Visualize(algo_name='AdaBoost')
            vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True)  # can only show mean
        :param y_trues: n-by-k multi-class true labels of k y_test, each of n samples
        :param y_scores: n-by-d-by-k multi-class prediction probabilities of k y_test, each of n samples
        :param show_all: True if want to show ROC curve for each fold, False if only want to show the mean ROC curve
        :param save_fig_path: path to save the plot as a figure
        :param save_coords_path: path to save cross-validated mean coordinates of ROC curve
        :return: None
        """
        # get k of k-fold cross validation
        K = len(y_trues)
        # store mean ROC curve for each fold
        all_fprs = []
        all_tprs = []
        plt.figure()
        for fold in range(K):
            y_true = y_trues[fold]
            y_score = y_scores[fold]
            y_true_bin = label_binarize(y_true, classes=self.labels)
            assert y_true_bin.shape[1] == self.label_num, 'Classes after binarization does not equal to {}!'.format(
                self.label_num)
            assert y_true_bin.shape == y_score.shape, 'Multi-class matrix size not match!'
            fprs, tprs = self.rocCurves(y_true_bin, y_score)
            # only care about the label-averaged ROC curve in each fold of cross validation
            all_fpr, mean_tpr = self.interpMeanRocCurve(fprs, tprs)
            all_fprs.append(all_fpr)
            all_tprs.append(mean_tpr)
            auc = self.aucByRate(all_fpr, mean_tpr)
            if show_all:
                plt.plot(all_fpr, mean_tpr, lw=2, label='ROC curve of fold {0} (area = {1:0.2f})'.format(fold, auc))
        # get interpolated averaged ROC curve of all folds
        cv_fprs, cv_tprs = self.interpMeanRocCurve(all_fprs, all_tprs)
        if save_coords_path != '':
            with open(save_coords_path, 'a') as f:
                csv_writer = csv.writer(f, dialect='excel')
                csv_writer.writerow(np.round(cv_fprs, 2))
                csv_writer.writerow(np.round(cv_tprs, 2))
        auc = self.aucByRate(cv_fprs, cv_tprs)
        plt.plot(cv_fprs, cv_tprs, label='macro-average ROC curve (area = {0:0.2f})'.format(auc),
                 color='navy', linestyle=':', linewidth=4)
        # plot random guess line for binary prediction
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) of {}-fold Cross Validation of {}'.format(K, self.algo_name))
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_fig_path != '':
            plt.savefig(save_fig_path)
        else:
            plt.show()

    def loadAndPlotRocCurve(self, roc_coords_paths=[], save_fig_path=''):
        if roc_coords_paths:
            plt.figure()
            for path in roc_coords_paths:
                # get algorithm name
                pa, fi = os.path.split(path)
                name, ext = os.path.splitext(fi)
                algorithm = name.split('_')[0]
                # plot the coordinates
                with open(path, 'r') as f:
                    csv_reader = csv.reader(f)
                    fpr = next(csv_reader)
                    tpr = next(csv_reader)
                    fpr = [float(i) for i in fpr]
                    tpr = [float(i) for i in tpr]
                    auc = self.aucByRate(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label='ROC curve of {0} (area = {1:0.2f})'.format(algorithm, auc))
            # plot random guess line for binary prediction
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) of Multiple Algorithms')
            plt.legend(loc="lower right")
            plt.tight_layout()
            if save_fig_path:
                plt.savefig(save_fig_path)
            else:
                plt.show()
        else:
            print('Path list is empty!')

    def rocCurve(self, y_true, y_score):
        """
        Get the ROC fpr and tpr coordinates of one single binary label(prediction)
        :param y_true: n-by-1 array with binary labels
        :param y_score: n-by-1 array with predicted probabilities of a single label
        :return: fpr and tpr coordinates of a single ROC curve
        """
        # y_true should have binary labels that contain 1, the other one can be 0 or -1, but does not matter
        assert len(y_true) == len(y_score), 'Size not match!'
        num_pos = sum([1 if i == 1 else 0 for i in y_true])
        num_neg = len(y_true) - num_pos
        assert num_pos > 0 and num_neg > 0, 'y_true should have both positive and negative labels!'
        x_step = 1.0 / num_neg
        y_step = 1.0 / num_pos
        fpr = [0]
        tpr = [0]
        inds = np.argsort(-y_score)  # descending order
        for index in inds:
            if y_true[index] == 1:
                tpr.append(tpr[-1] + y_step)
                fpr.append(fpr[-1])
            else:
                fpr.append(fpr[-1] + x_step)
                tpr.append(tpr[-1])
        assert len(fpr) == len(tpr), 'FPR and TPR do not match size!'
        return fpr, tpr

    def rocCurves(self, y_trues, y_scores):
        """
        Get the ROC fpr and tpr coordinates of all labels for a single prediction
        :param y_trues: n-by-d array of binary y_test labels of each label
        :param y_scores: n-by-d array of binary prediction probabilities of each label
        :return: list of fpr coordinates and list of tpr coordinates
        """
        assert y_trues.shape == y_scores.shape, 'y_trues and y_scores do not match by size!'
        assert y_trues.shape[1] == self.label_num, 'label number does not match!'
        y_trues = np.array(y_trues)
        y_scores = np.array(y_scores)
        fprs = []
        tprs = []
        for label in range(self.label_num):
            fpr, tpr = self.rocCurve(y_trues[:, label], y_scores[:, label])
            fprs.append(fpr)
            tprs.append(tpr)
        return fprs, tprs

    def aucByScore(self, y_true, y_score):
        """
        Get Area Under Curve (AUC) by a single model's single binary prediction
        :param y_true: n-by-1 array with binary labels
        :param y_score: n-by-1 array with predicted probabilities of a single label
        :return: area under the ROC curve
        """
        # y_true should have binary labels that contain 1, the other one can be 0 or -1, but does not matter
        assert len(y_true) == y_score, 'Size not match!'
        num_pos = sum([1 if i == 1 else 0 for i in y_true])
        num_neg = len(y_true) - num_pos
        assert num_pos > 0 and num_neg > 0, 'y_true should have both positive and negative labels!'
        x_step = 1.0 / num_neg
        y_step = 1.0 / num_pos
        y_sum = 0
        last_y = 0
        inds = np.argsort(-y_score)  # descending order
        for index in inds:
            if y_true[index] == 1:
                last_y += y_step
            else:
                y_sum += last_y
        assert abs(last_y - 1) < 1e-3, 'ROC curve should end up at (1,1) point!'
        res = y_sum * x_step
        assert res <= 1, 'AUC should never exceeds 1!'
        return res

    def aucByRate(self, fpr, tpr):
        """
        Get Area Under Curve (AUC) by ROC curve coordinates
        :param fpr: False Positive Rate
        :param tpr: True Positive Rate
        :return: area under the ROC curve
        """
        assert len(fpr) == len(tpr), 'Size of fpr and tpr should match!'
        assert len(fpr) > 1, 'Should have at least 2 rates!'
        # for interpolated cases where fpr interval might not be even
        res = 0
        for i in range(1, len(fpr)):
            res += tpr[i] * (fpr[i] - fpr[i - 1])
        assert res <= 1, 'AUC should never exceeds 1!'
        return res

    def interpMeanRocCurve(self, fprs, tprs):
        """
        Interpolate tpr coordinates, and get the averaged tpr
        :param fprs: list of false positive rates coordinates
        :param tprs: list of true positive rates coordinates
        :return: concatenated and sorted fpr coordinate, interpolated and averaged tpr coordinate
        """
        # Calculate macro ROC
        # First aggregate all false positive rates
        l = len(fprs)
        assert l > 0, 'fprs is empty!'
        assert l == len(tprs), 'fprs and tprs size not match!'
        all_fpr = np.unique(np.concatenate([fprs[i] for i in range(l)]))
        # Then interpolate all true positive rates at these fpr points, and get the mean
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(l):
            mean_tpr += np.interp(all_fpr, fprs[i], tprs[i])
        mean_tpr /= l
        return all_fpr, mean_tpr

    def loadMeanAccuMatOfHyperParam(self, csv_path, param_grid, fixed_params={}, load_test=True):
        if not csv_path:
            print('Should specify csv file path of your algorithm!')
            return
        if not fixed_params:
            assert len(param_grid) == 2, 'No fixed param, but only accept 2 variables right now!'

        mutable_params = {}
        # print('all ', param_grid)
        # print('fixed ', fixed_params)
        for param_key in param_grid:
            if param_key not in fixed_params:
                mutable_params[param_key] = param_grid[param_key]
                print('Mutable: ', param_key)
        if len(mutable_params) != 2:
            print('Only accept 2 variables right now!')
            print(mutable_params)
            return

        tp = 'Test' if load_test else 'Train'
        mat = []
        df = pd.read_csv(csv_path)
        groupby = df.groupby(['type'] + list(fixed_params.keys()) + list(mutable_params.keys()))
        params2 = list(mutable_params.keys())
        for param1 in mutable_params[params2[0]]:
            mean_scores = []
            for param2 in mutable_params[params2[1]]:
                query_tuple = (*(tp,), *tuple(fixed_params.values()), *(param1, param2))
                scores_df = groupby.get_group(query_tuple)
                scores = np.array(scores_df.iloc[:, -10:])
                print('x: {}, y: {}, scores: {}'.format(param1, param2, scores))
                mean_scores.append(np.mean(scores))
            mat.append(mean_scores)
        return mutable_params, np.array(mat)

    def loadCVScores(self, csv_path, best_param):
        df = pd.read_csv(csv_path)
        gb = df.groupby(['type'] + list(best_param.keys()))
        train_scores_df = gb.get_group((*('Train',), *tuple(best_param.values())))
        train_scores = np.array(train_scores_df.iloc[:, -10:])
        test_scores_df = gb.get_group((*('Test',), *tuple(best_param.values())))
        test_scores = np.array(test_scores_df.iloc[:, -10:])
        return list(train_scores), list(test_scores)

    def plotHyperParamHeatmap(self, mutable_params, mean_accu_mat, title='', save_path=''):
        plt.figure(figsize=(6.4, 4.8))
        ax = sns.heatmap(data=mean_accu_mat, annot=True)
        keys = list(mutable_params.keys())
        # ax = plt.subplot()
        ax.set_xticklabels(mutable_params[keys[1]])
        ax.set_yticklabels(mutable_params[keys[0]])
        plt.title(title)
        plt.xlabel(keys[1])
        plt.ylabel(keys[0])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plotAccuSubsetSize(self, algos, title='', save_path=''):
        assert algos, 'Algorithms should not be empty!'
        algorithms = list(algos.keys())
        plt.figure()
        for algo, pair in algos.items():
            x = []
            y = []
            for sub_size, scores in pair.items():
                x.append(sub_size)
                score_mean = np.mean(scores)
                y.append(score_mean)
            plt.plot(x, y, lw=2)
            plt.xticks(ticks=x)
        # plt.errorbar(x=x, y=y_train_mean, yerr=y_train_std)
        # plt.errorbar(x=x, y=y_test_mean, yerr=y_test_std)
        # plt.xticks(ticks=x)
        plt.title(title)
        plt.xlabel('Subset Size')
        plt.ylabel('Accuracy')
        plt.legend(algorithms)
        # plt.legend(['Training', 'Test'])
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plotAccuIter(self, method, x, train, test, save_path=''):
        plt.figure()
        plt.plot(x, train, label='train')
        plt.plot(x, test, label='test')
        plt.xlabel("Iteration Times")
        plt.ylabel("Accuracy")
        plt.title("Accuracy versus Iteration Times (Method: {})".format(method))
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plotIterVsAcc(self, method="CNN"):
        scores_train = pickle.load(open("../result/iter_vs_acc/{}_scores_train.pkl".format(method), "rb"))
        scores_test = pickle.load(open("../result/iter_vs_acc/{}_scores_test.pkl".format(method), "rb"))

        x = []
        if method == "CNN":
            x = range(1, 10000, 100)
        if method == "VGG":
            x = range(1, 600, 10)
        if method == "ResNet":
            x = range(1, 40)

        y_train = [scores_train[i] for i in x]
        y_test = [scores_test[i] for i in x]

        plt.plot(x, y_train, label="train")
        plt.plot(x, y_test, label="test")

        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy")
        plt.title("Accuracy versus Iteration Times (Method: {})".format(method))
        plt.legend()
        # plt.show()

        save_path = "../docs/report/figures/iter_vs_acc/{}.pdf".format(method)
        plt.savefig(save_path)

def tuneHyperParams():
    vis = Visualize()
    # 1. Hyper-parameter tuning of all algorithms
    #   1.1 Perceptron
    params = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 500, 1000]
    }
    # By default use the max subset
    for load_file in ['raw_pixels-subset500', 'raw_pixels+landmarks-subset500']:
        for k, v in params.items():
            for value in v:
                fixed_param = {k: value}  # only 1 fixed since 3 params in total
                para, mat = vis.loadMeanAccuMatOfHyperParam('../result/Perceptron/' + load_file + '.csv',
                                                            param_grid=params, fixed_params=fixed_param)
                vis.plotHyperParamHeatmap(params, mat,
                                          'Accuracy of Perceptron with hyperparam {} = {}'.format(k, value),
                                          '../result/Perceptron/HyperparamFigures/' + load_file + '-' + str(
                                              k) + '=' + str(value) + '.pdf')

    #   1.2. AdaBoost
    # todo

    #   1.3. SVM
    # todo


def accuVsSubsetSize():
    vis = Visualize()
    # 2. Accuracy on Training and Test set v.s. subset size
    #   2.1 Perceptron
    # Should acquire best_param from Evaluation.getBestModelAndScore
    best_param_perceptron = {'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000}
    best_param_svm = {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
    best_param_adaboost = {'n_estimators': 500, 'learning_rate': 0.1}

    # subset_to_scores = {}
    # perceptron_raw_subset_scores = {}
    # perceptron_rawlm_subset_scores = {}
    # svm_raw_subset_scores = {}
    # for root, dirs, files in os.walk('../result/Perceptron/'):
    #     files.sort(key=lambda x: int(re.findall("\d+", x)[0]))
    #     print(files)
    #     for file in files:
    #         base_name, ext = os.path.splitext(file)
    #         if ext == '.csv':
    #             print(root + file)
    #             train_scores, test_scores = vis.loadCVScores(root + file, best_param_perceptron)
    #             subset_size = int(re.findall("\d+", base_name)[0])
    #             print(subset_size)
    #             if 'landmarks' in base_name:
    #                 perceptron_raw_subset_scores[subset_size] = [train_scores, test_scores]
    #             else:
    #                 perceptron_rawlm_subset_scores[subset_size] = [train_scores, test_scores]
    # vis.plotAccuSubsetSize(subset_to_scores, 'Accuracy of Training and Test set v.s. Subset Size of Perceptron',
    #                        '../result/Perceptron/AccuracyVsSubsetSizeFigures/' + str(best_param_perceptron.values()) + '.pdf')

    # hard write scores
    perc_raw_subset_scores = {50: [0.26,0.2,0.31,0.17,0.29,0.2,0.17,0.14,0.17,0.31],
                              100: [0.3,0.19,0.21,0.29,0.24,0.26,0.16,0.26,0.14,0.26],
                              150: [0.24,0.25,0.22,0.26,0.23,0.21,0.14,0.27,0.19,0.18],
                              200: [0.27,0.21,0.24,0.22,0.2,0.24,0.21,0.15,0.16,0.24],
                              250: [0.19,0.23,0.17,0.16,0.29,0.17,0.24,0.22,0.17,0.23],
                              300: [0.31,0.27,0.23,0.19,0.24,0.3,0.24,0.29,0.23,0.22],
                              350: [0.29,0.22,0.19,0.22,0.28,0.26,0.2,0.22,0.18,0.2],
                              400: [0.3,0.23,0.2,0.27,0.24,0.21,0.25,0.2,0.24,0.28],
                              450: [0.23,0.18,0.23,0.18,0.22,0.27,0.17,0.24,0.2,0.27],
                              500: [0.3,0.24,0.24,0.3,0.25,0.21,0.16,0.29,0.27,0.18]}

    svm_raw_subset_scores = {50: [0.2,0.286,0.286,0.286,0.257,0.2,0.314,0.314,0.286,0.2],
                             100: [0.4,0.271,0.314,0.3,0.243,0.271,0.143,0.229,0.243,0.186],
                             150: [0.41,0.295,0.295,0.295,0.152,0.333,0.229,0.305,0.305,0.238],
                             200: [0.364,0.357,0.329,0.25,0.264,0.321,0.329,0.264,0.307,0.307],
                             250: [0.383,0.309,0.274,0.251,0.314,0.286,0.337,0.32,0.223,0.297],
                             300: [0.362,0.39,0.248,0.295,0.286,0.352,0.319,0.248,0.352,0.314],
                             350: [0.384,0.351,0.241,0.31,0.347,0.327,0.318,0.318,0.339,0.331],
                             400: [0.364,0.364,0.279,0.35,0.357,0.286,0.354,0.336,0.339,0.307],
                             450: [0.362,0.337,0.327,0.371,0.337,0.352,0.362,0.359,0.34,0.34],
                             500: [0.34,0.33,0.35,0.37,0.32,0.38,0.34,0.36,0.35,0.35]}

    ada_raw_subset_scores = {50: [0.17,0.23,0.29,0.2,0.34,0.11,0.11,0.2,0.2,0.11],
                             100: [0.27,0.14,0.33,0.17,0.26,0.16,0.21,0.27,0.26,0.21],
                             150: [0.24,0.29,0.26,0.21,0.3,0.21,0.19,0.22,0.23,0.24],
                             200: [0.22,0.24,0.25,0.29,0.21,0.23,0.28,0.28,0.35,0.27],
                             250: [0.26,0.29,0.22,0.22,0.28,0.27,0.3,0.24,0.19,0.26],
                             300: [0.25,0.27,0.23,0.21,0.24,0.32,0.31,0.2,0.24,0.25],
                             350: [0.25,0.25,0.21,0.27,0.31,0.29,0.22,0.24,0.25,0.29],
                             400: [0.28,0.27,0.26,0.28,0.34,0.24,0.26,0.29,0.32,0.28],
                             450: [0.28,0.27,0.26,0.33,0.25,0.27,0.29,0.33,0.26,0.37],
                             500: [0.3,0.26,0.28,0.31,0.25,0.27,0.29,0.29,0.33,0.35]}

    dt_raw_subset_scores = {50: [0.2,0.11,0.14,0.17,0.23,0.06,0.11,0.14,0.29,0.2],
                            100: [0.2,0.27,0.16,0.2,0.16,0.17,0.1,0.2,0.16,0.11],
                            150: [0.2,0.17,0.2,0.19,0.24,0.19,0.18,0.16,0.23,0.18],
                            200: [0.24,0.21,0.26,0.19,0.21,0.24,0.19,0.19,0.24,0.21],
                            250: [0.16,0.17,0.17,0.18,0.27,0.25,0.19,0.18,0.22,0.26],
                            300: [0.2,0.22,0.19,0.21,0.23,0.26,0.22,0.22,0.19,0.2],
                            350: [0.18,0.24,0.2,0.21,0.23,0.25,0.24,0.24,0.21,0.22],
                            400: [0.22,0.24,0.22,0.22,0.22,0.2,0.22,0.23,0.2,0.24],
                            450: [0.23,0.24,0.24,0.24,0.17,0.21,0.24,0.21,0.19,0.22],
                            500: [0.23,0.21,0.23,0.25,0.2,0.23,0.22,0.21,0.23,0.27]}

    algos = {'Perceptron': perc_raw_subset_scores,
             'SVM': svm_raw_subset_scores,
             'AdaBoost': ada_raw_subset_scores,
             'DecisionTree': dt_raw_subset_scores}
    vis.plotAccuSubsetSize(algos, 'Comparison between Best Accuracy and Subset Size',
                           '../docs/report/figures/accuVsSubset.pdf')

def accuVsIteration():
    vis = Visualize()
    # svm
    iter = np.linspace(50, 500+1, 10)
    train = []
    test = []
    with open('../result/SVM/iter.csv') as f:
        reader = csv.reader(f)
        for l in iter:
            tr = list(next(reader))[-10:]
            tr = [float(i) for i in tr]
            te = list(next(reader))[-10:]
            te = [float(i) for i in te]
            train.append(np.mean(tr))
            test.append(np.mean(te))
    vis.plotAccuIter('SVM', iter, train, test, '../docs/report/figures/iter_vs_acc/SVM.pdf')

    # AdaBoost
    test = [0.314, 0.321, 0.330, 0.334, 0.337, 0.338, 0.331, 0.338, 0.338, 0.344]
    train = []
    with open('../result/AdaBoost/raw_pixels-subset500.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = list(row)
            if row[0] == '0.1' and row[2] == 'Train':
                train.append(np.mean([float(i) for i in row[-10:]]))
    vis.plotAccuIter('AdaBoost', iter, train, test, '../docs/report/figures/iter_vs_acc/AdaBoost.pdf')

    # Perceptron
    train = []
    test = []
    iter = np.arange(100, 1000+1, 10)
    with open('../result/Perceptron/iter.csv') as f:
        reader = csv.reader(f)
        for l in iter:
            tr = float(list(next(reader))[-1])
            te = float(list(next(reader))[-1])
            train.append(tr)
            test.append(te)
    vis.plotAccuIter('Perceptron', iter, train, test, '../docs/report/figures/iter_vs_acc/Perceptron.pdf')

def roc():
    vis = Visualize()
    roc_csv_paths = ['../result/DecisionTree/DecisionTree_roc_rp.csv',
                     '../result/Perceptron/Perceptron_roc_rp.csv',
                     '../result/AdaBoost/AdaBoost_roc_rp.csv',
                     '../result/SVM/SVM_roc_rp.csv']
    vis.loadAndPlotRocCurve(roc_csv_paths, save_fig_path='../result/roc.pdf')


if __name__ == "__main__":

    # tuneHyperParams()
    accuVsSubsetSize()
    # accuVsIteration()
    # roc()

    # vis = Visualize()
    # vis.plotIterVsAcc(method="CNN")
    # vis.plotIterVsAcc(method="VGG")
    # vis.plotIterVsAcc(method="ResNet")


