from FER2013 import FER2013
from AdaBoost import AdaBoost
from Perceptron import Perceptron
from DecisionTree import DecisionTree
from SVM import SVM
from Evaluation import Evaluation
from Visualize import Visualize

if __name__ == "__main__":
    fer = FER2013(filename="../data/subset3500.csv")
    eva = Evaluation(fer, 500)

    dt = DecisionTree()
    per = Perceptron()
    ada = AdaBoost()
    svm = SVM()

    models = {'DecisionTree': dt,
              'Perceptron': per,
              'AdaBoost': ada,
              'SVM': svm}
    params_decisiontree = {'criterion': 'entropy', 'max_depth': 16, 'min_samples_split': 4}
    params_perceptron = {'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000}
    params_adaboost = {'n_estimators': 500, 'learning_rate': 0.1}
    params_svm = {'C': 100, 'kernel': 'rbf', 'gamma': 0.01, 'max_iter': 1000}
    params = [params_decisiontree, params_perceptron, params_adaboost, params_svm]

    for model_name, param in zip(models.keys(), params):
        model = models[model_name]
        model.set_params(param)
        _, _, y_test_pred, y_test_true, _, _ = eva.kfoldCV(10, model, run_once=True)
        vis = Visualize(model_name)
        vis.set_y(y_test_pred, y_test_true)
        vis.plotConfusionMatrix(save_path='../docs/report/figures/{}ConfusionMatrix.pdf'.format(model_name))

