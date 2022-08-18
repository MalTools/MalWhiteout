# /usr/vin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import psutil
import time
import numpy as np
from GetCorpus import GetDataSet
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities

# logging level
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")

noise_ratio = 5

def read_feature_vector(train_malware_corpus, train_goodware_corpus,
               num_features_to_be_selected, feature_option):
    # Getting the txt files for apks
    Logger.debug("Loading positive and negative samples")
    train_mal_samples = glob.glob(os.path.join(train_malware_corpus, '*txt'))
    train_good_samples = glob.glob(os.path.join(train_goodware_corpus, '*txt'))

    sample_list = []
    for sample in train_mal_samples:
        sample_list.append(sample.split('/')[-1])
    for sample in train_good_samples:
        sample_list.append(sample.split('/')[-1])
    with open('samples_csbd_%sn.txt'%noise_ratio, 'w') as sf:
        sf.write(str(sample_list))

    Logger.info("All Samples loaded")

    # Creating feature vectors
    feature_vectorizer = TF(input='filename', lowercase=False, token_pattern=None,
                            tokenizer=lambda s: s.split(), binary=feature_option, dtype=np.int)
    x_train = feature_vectorizer.fit_transform(train_mal_samples + train_good_samples)

    # Label training sets malware as 1 and goodware as 0
    train_mal_labels = np.ones(len(train_mal_samples), dtype=int)
    train_good_labels = np.zeros(len(train_good_samples), dtype=int)
    y_train = np.concatenate((train_mal_labels, train_good_labels), axis=0)
    Logger.info("Training Label array - generated")

    # Doing feature selection
    features = feature_vectorizer.get_feature_names()
    Logger.info("Total number of features: {} ".format(len(features)))

    if len(features) > num_features_to_be_selected:
        # with feature selection
        Logger.info("Gonna select %s features", num_features_to_be_selected)
        fs_algo = SelectKBest(chi2, k=num_features_to_be_selected)

        x_train = fs_algo.fit_transform(x_train, y_train)
    print(x_train.shape, y_train.shape)
    return x_train, y_train


def predict_noise(x_train, y_train):
    clf = LearningWithNoisyLabels(RandomForestClassifier(n_estimators=200))
    clf.fit(x_train, y_train)

    # Use cleanlab to compute out-of-sample predicted probabilities (psx)
    psx = estimate_cv_predicted_probabilities(
        X=x_train,
        labels=y_train,
        clf=clf.clf,
        cv_n_folds=clf.cv_n_folds,
        seed=clf.seed,
    )
    psx_list = []
    for pro in list(psx):
        psx_list.append(list(pro))
    with open('psx_csbd-%sn.txt'%noise_ratio, 'w') as pf:
        pf.write(str(psx_list))
    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    with open('csbd_labelerrorsmask-%sn.txt'%noise_ratio, 'w') as ef:
        ef.write(str(list(label_errors_mask)))


def predict_noise_2(x_train, y_train):
    Parameters = {'n_estimators': [10, 50, 100, 200, 500, 1000],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}
    Clf = GridSearchCV(RandomForestClassifier(), Parameters, cv=5, scoring='f1', n_jobs=-1)
    RFmodels = Clf.fit(x_train, y_train)
    BestModel = RFmodels.best_estimator_
    num_crossval_folds = 5  # for efficiency; values like 5 or 10 will generally work better
    psx = cross_val_predict(
        BestModel,
        x_train,
        y_train,
        cv=num_crossval_folds,
        method="predict_proba",
    )

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    with open('csbd_labelerrorsmask-%sn.txt' % noise_ratio, 'w') as ef:
        ef.write(str(list(label_errors_mask)))


def main(args, feature_option):
    train_maldir = args.maldir
    train_gooddir = args.gooddir
    process_num = args.processno
    timeout = args.timeout
    GetDataSet(train_maldir, process_num, timeout)
    GetDataSet(train_gooddir, process_num, timeout)
    num_features_to_be_selected = args.numfeatures

    x_train, y_train = read_feature_vector(train_maldir, train_gooddir,
                                                  num_features_to_be_selected,
                                                  feature_option)
    predict_noise(x_train, y_train)


def parse_args():
    args = argparse.ArgumentParser("UserInput")
    args.add_argument("--maldir", default="../../apks/train_maldir",
                      help="Absolute path to directory containing malware apks")
    args.add_argument("--gooddir", default="../../apks/train_gooddir",
                      help="Absolute path to directory containing benign/goodware apks")
    args.add_argument("--processno", default=psutil.cpu_count(), type=int,
                      help="Number of processes scheduled")
    args.add_argument("--timeout", default=120, type=int,
                      help="Max number of seconds that can be used for extracting CFG signature features from an apk")
    args.add_argument("--numfeatures", default=5000, type=int,
                      help="Number of top features to select")
    return args.parse_args()


if __name__ == "__main__":
    start = time.time()
    main(parse_args(), False)
    end = time.time()
    print('use time:', end - start)


