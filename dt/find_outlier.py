#coding:utf-8
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
# from pyod.utils.data import generate_data
# from pyod.utils.data import evaluate_print
# from pyod.utils.example import visualize
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.sos import SOS
from pyod.models.pca import PCA
from sklearn.covariance import EllipticEnvelope
import numpy as np

def knn_outlier(X_train, Y_train, contamination):
    """
    @description  :KNN离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    clf_name = 'KNN'
    clf = KNN(contamination=contamination)
    clf.fit(X_train)
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # evaluate and print the results
    return y_train_pred.tolist()


def abod_outlier(X_train, Y_train, contamination):
    """
    @description  :ABOD 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train ABOD detector
    clf_name = 'ABOD'
    clf = ABOD(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def autoEncoder_outlier(X_train, Y_train, contamination):
    """
    @description  :autoEncoder 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    clf_name = 'AutoEncoder'
    size = len(X_train[0])
    clf = AutoEncoder(hidden_neurons=[size, size * 2, size * 4, size * 2, size], contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def cblof_outlier(X_train, Y_train, contamination):
    """
    @description  :CBLOF 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train CBLOF detector
    clf_name = 'CBLOF'
    clf = CBLOF(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def hbos_outlier(X_train, Y_train, contamination):
    """
    @description  :HBOS 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train HBOS detector
    clf_name = 'HBOS'
    clf = HBOS(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def I_forest_outlier(X_train, Y_train, contamination):
    """
    @description  :I-forest 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    clf_name = 'IForest'
    clf = IForest(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def lof_outlier(X_train, Y_train, contamination):
    """
    @description  :LOF 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train LOF detector
    clf_name = 'LOF'
    clf = LOF(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def mcd_outlier(X_train, Y_train, contamination):
    """
    @description  :MCD 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train LOF detector
    clf_name = 'MCD'
    clf = MCD(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def so_gaal_outlier(X_train, Y_train, contamination):
    """
    @description  :So-gaal 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train SO_GAAL detector
    clf_name = 'SO_GAAL'
    clf = SO_GAAL(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def ocsvm_outlier(X_train, Y_train, contamination):
    """
    @description  :OCSVM 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train one_class_svm detector
    clf_name = 'OneClassSVM'
    clf = OCSVM(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def sos_outlier(X_train, Y_train, contamination):
    """
    @description  :SOS 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train SOS detector
    clf_name = 'SOS'
    clf = SOS(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def pca_outlier(X_train, Y_train, contamination):
    """
    @description  :PCA 离群分析(0 inlier 1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # train PCA detector
    clf_name = 'PCA'
    clf = PCA(contamination=contamination)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    return y_train_pred.tolist()


def ellipticEnvelope_outlier(X_train, Y_train, contamination):
    """
    @description  :EllipticEnvelope 离群分析(1 inlier -1 outlier)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    clf_name = 'EllipticEnvelope'

    X = np.array(X_train)
    clf = EllipticEnvelope(contamination=contamination)
    y_train_pred = clf.fit_predict(X)  # Returns -1 for outliers and 1 for inliers.
    # get the prediction labels and outlier scores of the training data
    # y_train_pred = clf.predict(X) # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.score_samples(X)  # raw outlier scores
    # decisions = clf.decision_function(X)

    return y_train_pred.tolist()

    # print(y_train_scores)
    # print(decisions)


def vote(X_train, Y_train, rate, app_ids):
    knn_pre = knn_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    abod_pre = abod_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    # autoEncoder_pre = autoEncoder_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)  # √
    cblof_pre = cblof_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    hbos_pre = hbos_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    I_forest_pre = I_forest_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    lof_pre = lof_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    mcd_pre = mcd_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    # so_gaal_pre = so_gaal_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)  # √
    ocsvm_pre = ocsvm_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    sos_pre = sos_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    pca_pre = pca_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)
    ellipticEnvelope_pre = ellipticEnvelope_outlier(X_train=X_train, Y_train=Y_train, contamination=0.1)

    outlier_list = []
    for i in range(len(X_train)):
        outlier = 0
        if knn_pre[i] == 1:
            outlier += 1
        if abod_pre[i] == 1:
            outlier += 1
        # if autoEncoder_pre[i] == 1:
        #     outlier += 1
        if cblof_pre[i] == 1:
            outlier += 1
        if hbos_pre[i] == 1:
            outlier += 1
        if I_forest_pre[i] == 1:
            outlier += 1
        if lof_pre[i] == 1:
            outlier += 1
        if mcd_pre[i] == 1:
            outlier += 1
        # if so_gaal_pre[i] == 1:
        #     outlier += 1
        if ocsvm_pre[i] == 1:
            outlier += 1
        if sos_pre[i] == 1:
            outlier += 1
        if pca_pre[i] == 1:
            outlier += 1
        if ellipticEnvelope_pre[i] == -1:
            outlier += 1
        if outlier > (rate * 11):
            outlier_list.append(app_ids[i])
    return outlier_list
