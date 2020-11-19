from FER2013 import FER2013
from Perceptron import Perceptron
from Evaluation import Evaluation
from Visualize import Visualize

if __name__ == "__main__":
    fer = FER2013(filename="../data/subset3500.csv")
    eva = Evaluation(fer, 500)
    # Best param
    params = {'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000}
    model = Perceptron()
    model.set_params(params)
    y_train_pred, y_train_true, y_test_pred, y_test_true, train_scores, test_scores = eva.kfoldValid(10, model,
                                                                                                     proba=True)
    vis = Visualize(algo_name='Perceptron')
    vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True,
                       save_fig_path='../result/Perceptron/Perceptron_roc_rp.pdf',
                       save_coords_path='../result/Perceptron/Perceptron_roc_rp.csv')
