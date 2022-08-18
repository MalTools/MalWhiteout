from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import csv
from itertools import islice
import argparse
import time

noise_ratio = 5

def feature_extraction(file):
    vectors = []
    labels = []
    samples = []
    with open(file, 'r') as f:
        csv_data = csv.reader(f)
        for line in islice(csv_data, 1, None):
            sample_name = line[0]
            vector = [float(i) for i in line[1:-1]]
            label = int(float(line[-1]))
            vectors.append(vector)
            labels.append(label)
            samples.append(sample_name)
    with open('samples_malscan_%sn.txt'%noise_ratio, 'w') as sf:
        sf.write(str(samples))
    return vectors, labels

def degree_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'degree-with-%sn.csv'%noise_ratio
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def katz_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'katz-with-%sn.csv'%noise_ratio
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def closeness_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'closeness-with-%sn.csv'%noise_ratio
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def harmonic_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'harmonic-with-%sn.csv'%noise_ratio
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels


def random_features(vectors, labels):
    Vec_Lab = []
    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]

def predict_noise(vectors, labels):
    x_train = np.array(vectors)
    y_train = np.array(labels)
    print(x_train.shape, y_train.shape)

    # RandomForest
    clf = LearningWithNoisyLabels(RandomForestClassifier(n_estimators=200))
    # KNN_1
    # clf = LearningWithNoisyLabels(KNeighborsClassifier(n_neighbors=1))
    # KNN_3
    # clf = LearningWithNoisyLabels(KNeighborsClassifier(n_neighbors=3))

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
    with open('new_psx_malscan-{}n.txt'.format(noise_ratio), 'w') as pf:
        pf.write(str(psx_list))

    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    with open('new_labelerrorsmask-{}n.txt'.format(noise_ratio), 'w') as ef:
        ef.write(str(list(label_errors_mask)))


def predict_noise_2(vectors, labels):
    x_train = np.array(vectors, dtype=np.float32)
    y_train = np.array(labels)
    print(x_train.shape, y_train.shape)
    # Compute out-of-sample predicted probabilities through cross-validation
    # RandomForest
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
    with open('new_labelerrorsmask-{}n.txt'.format(noise_ratio), 'w') as ef:
        ef.write(str(list(label_errors_mask)))


def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains feature_CSV.', required=True)
    # parser.add_argument('-o', '--output', help='The path of output.', required=True)
    parser.add_argument('-t', '--type', help='The type of centrality: degree, closeness, harmonic, katz', required=True)

    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    feature_dir = args.dir
    # out_put = args.output
    type = args.type

    if feature_dir[-1] == '/':
        feature_dir = feature_dir
    else:
        feature_dir += '/'

    if type == 'degree':
        degree_vectors, degree_labels = degree_centrality_feature(feature_dir)
        predict_noise(degree_vectors, degree_labels)

    elif type == 'harmonic':
        harmonic_vectors, harmonic_labels = harmonic_centrality_feature(feature_dir)
        # classification(harmonic_vectors, harmonic_labels, 35)
        predict_noise(harmonic_vectors, harmonic_labels)
    elif type == 'katz':
        katz_vectors, katz_labels = katz_centrality_feature(feature_dir)
        predict_noise(katz_vectors, katz_labels)

    elif type == 'closeness':
        closeness_vectors, closeness_labels = closeness_centrality_feature(feature_dir)
        predict_noise(closeness_vectors, closeness_labels)

    else:
        print('Error Centrality Type!')

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('use time:', end-start)