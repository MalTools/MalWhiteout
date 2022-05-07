#/usr/vin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_extraction.text import HashingVectorizer as HS
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging
import random
import CommonModules as CM
from joblib import dump, load
#from pprint import pprint
import json, os
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
import sys
from GetApkData import GetApkData
from sklearn.ensemble import RandomForestClassifier
import psutil, argparse, logging
from scipy import sparse
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices


def Predict_cleanlab(TrainMalSet, TrainGoodSet, FeatureOption, NumTopFeats):
    # step 1: creating feature vector
    Logger.debug("Loading Malware and Goodware Sample Data for training and testing")
    TrainMalSamples = CM.ListFiles(TrainMalSet, ".data")
    TrainGoodSamples = CM.ListFiles(TrainGoodSet, ".data")
    sample_list = []
    for sample in TrainMalSamples:
        sample_list.append(sample.split('/')[-1])
    for sample in TrainGoodSamples:
        sample_list.append(sample.split('/')[-1])

    # with open('new_samples_drebin_35n.txt', 'w') as sf:
    #     sf.write(str(sample_list))

    # TestMalSamples = CM.ListFiles(TestMalSet, ".data")
    # TestGoodSamples = CM.ListFiles(TestGoodSet, ".data")
    # AllTestSamples = TestMalSamples + TestGoodSamples
    Logger.info("Loaded Samples")

    FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    x_train = FeatureVectorizer.fit_transform(TrainMalSamples + TrainGoodSamples)
    # HashVectorizer = HS(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
    #                   binary=FeatureOption,n_features=500)
    # x_train = HashVectorizer.fit_transform(TrainMalSamples + TrainGoodSamples)

    # label training sets malware as 1 and goodware as 0
    Train_Mal_labels = np.ones(len(TrainMalSamples), dtype=int)
    Train_Good_labels = np.zeros(len(TrainGoodSamples), dtype=int)
    # Train_Good_labels.fill(-1)
    y_train = np.concatenate((Train_Mal_labels, Train_Good_labels), axis=0)
    # pd.DataFrame(x_train).to_csv('x_train.csv')
    # pd.DataFrame(y_train).to_csv('y_train.csv')
    # with open('x_train.txt', 'a') as f:
    #     for i in range(x_train.shape[0]):
    #         for j in range(x_train.shape[1]):
    #             f.write(str(x_train[i][j])+',')
    #         f.write('\n')

    # np.save('dt_ytrain_drebin_10n', y_train)
    # sparse.save_npz('dt_xtrain_drebin_10n.npz', x_train)
    Logger.info("Training Label array - generated")
    # sys.exit(1)
    # step 2: train the model
    Logger.info("Perform Classification with SVM Model")
    # Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    print(x_train.shape, y_train.shape)

    # T0 = time.time()
    Clf = LearningWithNoisyLabels(SVC(kernel='linear', probability=True))
    # Clf = LearningWithNoisyLabels(RandomForestClassifier(n_estimators=200))
    # Clf = GridSearchCV(SVC(kernel='linear', probability=True), Parameters, cv=5, scoring='f1', n_jobs=-1)
    Clf.fit(x_train, y_train)
    # with open('noise_mask_drebin-10%n.txt', 'w') as nmf:
    #     nmf.write(str(list(Clf.noise_mask)))
    # 输出psx
    psx = estimate_cv_predicted_probabilities(
        X=x_train,
        labels=y_train,
        clf=Clf.clf,
        cv_n_folds=Clf.cv_n_folds,
        seed=Clf.seed,
    )
    # psx_list = []
    # for pro in list(psx):
    #     psx_list.append(list(pro))
    # with open('new_psx_drebin-35n.txt', 'w') as pf:
    #     pf.write(str(psx_list))

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    # with open('new_labelerrorsmask-35n.txt', 'w') as ef:
    #     ef.write(str(list(label_errors_mask)))
    # label_errors_mask_2 = get_noise_indices(s=y_train, psx=psx, sorted_index_method='normalized_margin')
    # with open('new_labelerrorsmask_2-35n.txt', 'w') as ef:
    #     ef.write(str(list(label_errors_mask_2)))
    print(len(label_errors_mask))

def main(Args, FeatureOption):
    MalDir = Args.maldir
    GoodDir = Args.gooddir
    # test_maldir = Args.testmaldir
    # test_gooddir = Args.testgooddir
    NCpuCores = Args.ncpucores
    Model = Args.model
    NumFeatForExp = Args.numfeatforexp
    # Perform Random Classification
    TestSize = Args.testsize
    # Logger.debug("MalDir: {}, GoodDir: {}, NCpuCores: {}, TestSize: {}, FeatureOption: {}, NumFeatForExp: {}"
    #              .format(MalDir, GoodDir, NCpuCores, TestSize, FeatureOption, NumFeatForExp))
    # GetApkData(NCpuCores, MalDir, GoodDir)
    # RandomClassification(MalDir, GoodDir, TestSize, FeatureOption, Model, NumFeatForExp)

    # Predict(MalDir, GoodDir, TestSize, FeatureOption, Model, NumFeatForExp)
    Predict_cleanlab(MalDir, GoodDir, FeatureOption, NumFeatForExp)
#    Predict(MalDir, GoodDir, TestSize, NumFeatForExp, LearningWithNoisyLabels(clf=LinearSVC()))


def ParseArgs():
    Args =  argparse.ArgumentParser(description="Classification of Android Applications")
    Args.add_argument("--holdout", type= int, default= 1,
                      help="Type of Classification to be performed (0 for Random Classification and 1 for Holdout Classification")
    Args.add_argument("--maldir", default= "../../apks/train_maldir",
                      help= "Absolute path to directory containing malware apks")
    Args.add_argument("--gooddir", default= "../../apks/train_gooddir",
                      help= "Absolute path to directory containing benign apks")
    Args.add_argument("--testmaldir", default= "../../apks/test_maldir",
                      help= "Absolute path to directory containing malware apks for testing when performing Holdout Classification")
    Args.add_argument("--testgooddir", default="../../apks/test_gooddir",
                      help= "Absolute path to directory containing goodware apks for testing when performing Holdout Classification")
    Args.add_argument("--ncpucores", type= int, default= psutil.cpu_count(),
                      help= "Number of CPUs that will be used for processing")
    Args.add_argument("--testsize", type= float, default= 0.3,
                      help= "Size of the test set when split by Scikit Learn's Train Test Split module")
    Args.add_argument("--model",
                      help= "Absolute path to the saved model file(.pkl extension)")
    Args.add_argument("--numfeatforexp", type= int, default = 30,
                      help= "Number of top features to show for each test sample")
    return Args.parse_args()


if __name__ == "__main__":
    start = time.time()
    main(ParseArgs(), True)
    end = time.time()
    print('use time:', end - start)
