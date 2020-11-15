import itertools
from sklearn.metrics import confusion_matrix
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

    def __init__(self, y_predicts, y_tests, algo_name=""):
        """
        :param y_predicts: array of y_prediction, which is also an array of predicted label of each sample
        :param y_tests: array of y_true, which is also an array of true label of each sample
        """
        self.set_y(y_predicts, y_tests)
        self.algo_name = algo_name
        self.labels = list(label2expression.keys())
        self.expressions = list(label2expression.values())
        self.label_num = len(self.labels)

    def set_y(self, y_predicts, y_tests):
        self.y_predicts = y_predicts
        self.y_tests = y_tests
        assert len(y_predicts) == len(y_tests), "y test and y true size not match!"
        assert len(y_predicts) > 0, "Expect at least 1 y array, get 0!"
        assert (y_predicts[i] == y_tests[i] for i in range(len(y_predicts))), \
            "y test and y true should be of the same size in every fold!"
        self.num = len(y_predicts)
        print("Get {} Estimation(s).".format(self.num))
        self.__genConfusionMatrix()

    def __genConfusionMatrix(self):
        self.conf_mat = np.zeros([self.label_num, self.label_num])
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
