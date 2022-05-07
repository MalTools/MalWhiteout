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


def preprocess(train_malware_corpus, train_goodware_corpus,
               num_features_to_be_selected, feature_option):
    # Step 1: Getting the malware and goodware txt files for both training and testing
    Logger.debug("Loading positive and negative samples")
    train_mal_samples = glob.glob(os.path.join(train_malware_corpus, '*txt'))
    train_good_samples = glob.glob(os.path.join(train_goodware_corpus, '*txt'))
    # with open('train-samples_csbd-20%n.txt', 'w') as tsf:
    #     tsf.write(str(train_mal_samples + train_good_samples))
    sample_list = []
    for sample in train_mal_samples:
        sample_list.append(sample.split('/')[-1])
    for sample in train_good_samples:
        sample_list.append(sample.split('/')[-1])
    with open('new_samples_csbd_35n.txt', 'w') as sf:
        sf.write(str(sample_list))

    # test_mal_samples = glob.glob(os.path.join(test_malware_corpus, '*txt'))
    # test_good_samples = glob.glob(os.path.join(test_goodware_corpus, '*txt'))
    Logger.info("All Samples loaded")

    # Step 2: Creating feature vectors
    feature_vectorizer = TF(input='filename', lowercase=False, token_pattern=None,
                            tokenizer=lambda s: s.split(), binary=feature_option, dtype=np.int)
    x_train = feature_vectorizer.fit_transform(train_mal_samples + train_good_samples)
    # x_test = feature_vectorizer.transform(test_mal_samples + test_good_samples)

    # Label training sets malware as 1 and goodware as 0
    train_mal_labels = np.ones(len(train_mal_samples), dtype=int)
    train_good_labels = np.zeros(len(train_good_samples), dtype=int)
    y_train = np.concatenate((train_mal_labels, train_good_labels), axis=0)
    Logger.info("Training Label array - generated")

    # Label testing sets malware as 1 and goodware as 0
    # test_mal_labels = np.ones(len(test_mal_samples), dtype=int)
    # test_good_labels = np.zeros(len(test_good_samples), dtype=int)
    # y_test = np.concatenate((test_mal_labels, test_good_labels), axis=0)
    # Logger.info("Testing Label array - generated")

    # Step 3: Doing feature selection
    features = feature_vectorizer.get_feature_names()
    Logger.info("Total number of features: {} ".format(len(features)))

    if len(features) > num_features_to_be_selected:
        # with feature selection
        Logger.info("Gonna select %s features", num_features_to_be_selected)
        fs_algo = SelectKBest(chi2, k=num_features_to_be_selected)

        x_train = fs_algo.fit_transform(x_train, y_train)
        # x_test = fs_algo.transform(x_test)

    return x_train, y_train


def classification(x_train, y_train, x_test, y_test, selected_model, model_parameters, n_jobs):
    # Step 4: Model selection through cross validation
    # Assuming RandomForest is the only classifier we are gonna try, we will set the n_estimators parameter as follows.
    clf = GridSearchCV(selected_model(), model_parameters, cv=5, scoring='f1', n_jobs=n_jobs)
    rf_models = clf.fit(x_train, y_train)
    best_model = rf_models.best_estimator_
    Logger.info('CV done - Best model selected: {}'.format(best_model))
    # Best model is chosen through 5-fold cross validation and stored in the variable: RFmodels

    Logger.info("Gonna perform classification with C-RandomForest")

    # Step 5: Evaluate the best model on test set
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy = ", accuracy)
    print(metrics.classification_report(y_test, y_pred, labels=[1, 0], target_names=['Malware', 'Goodware']))


def main(args, feature_option):
    train_maldir = args.maldir
    train_gooddir = args.gooddir
    test_maldir = args.testmaldir
    test_gooddir = args.testgooddir
    process_num = args.processno
    timeout = args.timeout
    # GetDataSet(train_maldir, process_num, timeout)
    # GetDataSet(train_gooddir, process_num, timeout)
    # GetDataSet(test_maldir, process_num, timeout)
    # GetDataSet(test_gooddir, process_num, timeout)
    num_features_to_be_selected = args.numfeatures

    # Logger.debug("TrainMalDir: {}, TrainGoodDir: {}, TestMalDir: {}, TestGoodDir:{} ProcessNo: {}," \
    #              "NumFeaturesToBeSelected: {}, FeatureOption: {}, TimeOut: {}".format(train_maldir, train_gooddir,
    #                                                                                   test_maldir, test_gooddir,
    #                                                                                   process_num,
    #                                                                                   num_features_to_be_selected,
    #                                                                                   feature_option,
    #                                                                                   timeout))

    x_train, y_train = preprocess(train_maldir, train_gooddir,
                                                  num_features_to_be_selected,
                                                  feature_option)

    # selected_model = ExtraTreesClassifier
    # model_parameters = {'n_estimators': [10, 50, 100, 200, 500, 1000],
    #                     'bootstrap': [True, False],
    #                     'criterion': ['gini', 'entropy']}
    # n_jobs = -1  # 设置为-1，则表示将电脑中的cpu全部用上
    # classification(x_train, y_train, x_test, y_test, selected_model, model_parameters, n_jobs)

    # selected_model_with_cleanlab = (lambda f: lambda: LearningWithNoisyLabels(f()))(RandomForestClassifier)
    # cleanlab_model_parameters = {'clf__n_estimators': [10, 50, 100, 200, 500, 1000],
    #                               'clf__bootstrap': [True, False],
    #                               'clf__criterion': ['gini', 'entropy']}
    # cleanlab_n_jobs = 1  # cleanlab 不能多核跑，只能单核
    # classification(x_train, y_train, x_test, y_test, selected_model_with_cleanlab, cleanlab_model_parameters,
    #                cleanlab_n_jobs)

    # get_noise_mask(x_train, y_train, selected_model_with_cleanlab, cleanlab_model_parameters)
    print(x_train.shape, y_train.shape)

    # 只输出噪声 不需要跑分类了
    clf = LearningWithNoisyLabels(RandomForestClassifier(n_estimators=200))
    clf.fit(x_train, y_train)
    # 输出noise_mask
    # with open('noise_mask_csbd-20%n.txt', 'w') as nmf:
    #     nmf.write(str(list(clf.noise_mask)))

    # 输出psx(概率)
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
    with open('new_psx_csbd-35n.txt', 'w') as pf:
        pf.write(str(psx_list))
    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    with open('new_labelerrorsmask-35n.txt', 'w') as ef:
        ef.write(str(list(label_errors_mask)))
    label_errors_mask_2 = get_noise_indices(s=y_train, psx=psx, sorted_index_method='normalized_margin')
    with open('new_labelerrorsmask_2-35n.txt', 'w') as ef:
        ef.write(str(list(label_errors_mask_2)))


def parse_args():
    args = argparse.ArgumentParser("UserInput")
    args.add_argument("--maldir", default="../../apks/train_maldir",
                      help="Absolute path to directory containing malware apks")
    args.add_argument("--gooddir", default="../../apks/train_gooddir",
                      help="Absolute path to directory containing benign/goodware apks")
    args.add_argument("--testmaldir", default="../../apks/test_maldir",
                      help="Absolute path to directory containing malware apks for testing(for Holdout Classification)")
    args.add_argument("--testgooddir", default="../../apks/test_gooddir",
                      help="Absolute path to directory containing benign/goodware apks for testing(for Holdout Classification)")
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


