import itertools
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import csv

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

    def __init__(self, y_predicts=None, y_tests=None, algo_name=""):
        """
        :param y_predicts: array of y_prediction, which is also an array of predicted label of each sample
        :param y_tests: array of y_true, which is also an array of true label of each sample
        """
        if y_tests is None:
            y_tests = []
        if y_predicts is None:
            y_predicts = []
        self.algo_name = algo_name
        self.labels = list(label2expression.keys())
        self.expressions = list(label2expression.values())
        self.label_num = len(self.labels)
        self.conf_mat = np.zeros([self.label_num, self.label_num])
        self.set_y(y_predicts, y_tests)
        self.__conf_mats = []

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
            plt.savefig(''.join([self.algo_name + "ConfusionMatrix.pdf"]))
            plt.show()

    def plotAccuracy(self):
        plt.figure()
        scores = [sum(y_true == y_pred) / len(y_true) for y_true, y_pred in zip(np.array(self.y_tests), np.array(self.y_predicts))]
        print(scores)
        # todo x range should be altered to specific subset sizes
        x = range(1, self.num+1)
        plt.plot(x, scores)
        plt.scatter(x, scores)
        for i in x:
            plt.text(x[i-1], scores[i-1], '%.02f' % scores[i-1])
        plt.xlabel('Sample Size')
        plt.ylabel('Accuracy')
        plt.title('Prediction Accuracy v.s. Training Sample Size')
        # plt.show()

    def saveConfMat(self, path, param1, param2):
        with open(path, 'a') as f:
            csv_writer = csv.writer(f, dialect='excel')
            csv_writer.writerow([param1, param2])
            csv_writer.writerows(self.conf_mat)

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
        assert y_true.shape[1] == self.label_num, 'Classes after binarization does not equal to {}!'.format(self.label_num)
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
        plt.title('Receiver Operating Characteristic (ROC) of {}-label Multi-class Prediction of {}'.format(self.label_num, self.algo_name))
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
            assert y_true_bin.shape[1] == self.label_num, 'Classes after binarization does not equal to {}!'.format(self.label_num)
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
                csv_writer.writerow(cv_fprs)
                csv_writer.writerow(cv_tprs)
        auc = self.aucByRate(cv_fprs, cv_tprs)
        plt.plot(cv_fprs, cv_tprs, label='macro-average ROC curve (area = {0:0.2f})'.format(auc),
                 color='navy', linestyle=':', linewidth=4)
        # plot random guess line for binary prediction
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic of {}-fold Cross Validation of {}'.format(K, self.algo_name))
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_fig_path != '':
            plt.savefig(save_fig_path)
        else:
            plt.show()

    # todo
    # show ROC curves of multiples cross-validated models in one graph

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


if __name__ == "__main__":
    # sample y of true and prediction
    # y_preds can be of the different sizes, as long as its size conforms with y_trues
    y_true1 = [0, 1, 2, 3, 4, 5, 6]
    y_pred1 = [1, 0, 2, 2, 5, 5, 5]  # 2 correct
    y_pred2 = [2, 1, 1, 3, 3, 5, 5]  # 3 correct
    y_pred3 = [0, 2, 2, 4, 4, 4, 6]  # 4 correct

    # test confusion matrix
    vis = Visualize([y_pred1, y_pred2, y_pred3], [y_true1] * 3, "Test")
    vis.plotConfusionMatrix()

    # vis.plotAccuracy()
