#/usr/vin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import logging
import CommonModules as CM
# from joblib import dump, load
#from pprint import pprint
import json, os
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
import sys
from GetApkData import GetApkData
import psutil, argparse, logging
from scipy import sparse
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices

noise_ratio = 5

def read_feature_vector(TrainMalSet, TrainGoodSet, FeatureOption):
    # creating feature vector
    TrainMalSamples = CM.ListFiles(TrainMalSet, ".data")
    TrainGoodSamples = CM.ListFiles(TrainGoodSet, ".data")
    sample_list = []
    for sample in TrainMalSamples:
        sample_list.append(sample.split('/')[-1])
    for sample in TrainGoodSamples:
        sample_list.append(sample.split('/')[-1])

    # save file name
    with open('samples_drebin_%sn.txt' % noise_ratio, 'w') as sf:
        sf.write(str(sample_list))

    Logger.info("Loaded Samples")

    FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    x_train = FeatureVectorizer.fit_transform(TrainMalSamples + TrainGoodSamples)

    # label training sets malware as 1 and goodware as 0
    Train_Mal_labels = np.ones(len(TrainMalSamples), dtype=int)
    Train_Good_labels = np.zeros(len(TrainGoodSamples), dtype=int)
    y_train = np.concatenate((Train_Mal_labels, Train_Good_labels), axis=0)

    # save feature vectors
    np.save('dt_ytrain_drebin_%sn'%noise_ratio, y_train)
    sparse.save_npz('dt_xtrain_drebin_%sn.npz'%noise_ratio, x_train)
    # sys.exit(1)

    print(x_train.shape, y_train.shape)
    return x_train, y_train


def predict_noise(x_train, y_train):
    Clf = LearningWithNoisyLabels(SVC(kernel='linear', probability=True))
    Clf.fit(x_train, y_train)

    # Use cleanlab to compute out-of-sample predicted probabilities (psx)
    psx = estimate_cv_predicted_probabilities(
        X=x_train,
        labels=y_train,
        clf=Clf.clf,
        cv_n_folds=Clf.cv_n_folds,
        seed=Clf.seed,
    )

    psx_list = []
    for pro in list(psx):
        psx_list.append(list(pro))
    with open('psx_drebin-%sn.txt'%noise_ratio, 'w') as pf:
        pf.write(str(psx_list))

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)  # the parameter "frac_noise"
    with open('drebin_labelerrorsmask-%sn.txt'%noise_ratio, 'w') as ef:
        ef.write(str(list(label_errors_mask)))


def predict_noise_2(x_train, y_train):
    # Compute out-of-sample predicted probabilities through cross-validation
    Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    Clf = GridSearchCV(SVC(kernel='linear', probability=True), Parameters, cv=5, scoring='f1', n_jobs=-1)
    SVMModels = Clf.fit(x_train, y_train)
    BestModel = SVMModels.best_estimator_
    num_crossval_folds = 5  # for efficiency; values like 5 or 10 will generally work better
    psx = cross_val_predict(
        BestModel,
        x_train,
        y_train,
        cv=num_crossval_folds,
        method="predict_proba",
    )

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)  # you can adjust the parameter "frac_noise"
    with open('drebin_labelerrorsmask-%sn.txt' % noise_ratio, 'w') as ef:
        ef.write(str(list(label_errors_mask)))


def main(Args, FeatureOption):
    MalDir = Args.maldir
    GoodDir = Args.gooddir
    NCpuCores = Args.ncpucores

    GetApkData(NCpuCores, MalDir, GoodDir)

    x_train, y_train = read_feature_vector(MalDir, GoodDir, FeatureOption)
    predict_noise(x_train, y_train)


def ParseArgs():
    Args =  argparse.ArgumentParser(description="Classification of Android Applications")
    Args.add_argument("--holdout", type= int, default= 1,
                      help="Type of Classification to be performed (0 for Random Classification and 1 for Holdout Classification")
    Args.add_argument("--maldir", default= "../../apks/train_maldir",
                      help= "Absolute path to directory containing malware apks")
    Args.add_argument("--gooddir", default= "../../apks/train_gooddir",
                      help= "Absolute path to directory containing benign apks")
    Args.add_argument("--ncpucores", type= int, default= psutil.cpu_count(),
                      help= "Number of CPUs that will be used for processing")

    return Args.parse_args()


if __name__ == "__main__":
    start = time.time()
    main(ParseArgs(), True)
    end = time.time()
    print('use time:', end - start)
