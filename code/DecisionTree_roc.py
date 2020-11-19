from FER2013 import FER2013
from DecisionTree import DecisionTree
from Evaluation import Evaluation
from Visualize import Visualize

if __name__ == "__main__":
    fer = FER2013(filename="../data/subset3500.csv")
    eva = Evaluation(fer, 500)
    # Best param
    params = {
        'criterion': 'entropy',
        'max_depth': 16,
        'min_samples_split': 4
    }
    model = DecisionTree()
    model.set_params(params)
    y_train_pred, y_train_true, y_test_pred, y_test_true, train_scores, test_scores = eva.kfoldValid(10, model,
                                                                                                     proba=True)
    vis = Visualize(algo_name='DecisionTree')
    vis.plotCVRocCurve(y_test_true, y_test_pred, show_all=True,
                       save_fig_path='../result/DecisionTree/DecisionTree_roc_rp.pdf',
                       save_coords_path='../result/DecisionTree/DecisionTree_roc_rp.csv')