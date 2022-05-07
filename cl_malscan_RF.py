from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
import csv
from itertools import islice
import argparse
import time

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
    with open('new_samples_malscan_35n.txt', 'w') as sf:
        sf.write(str(samples))

    return vectors, labels

def degree_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'degree-with-10n.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def katz_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'katz-with-10n.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def closeness_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'closeness-with-30n.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def harmonic_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'harmonic-with-35n.csv'
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

def classification(vectors, labels, noise):
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
    with open('new_psx_malscan-{}n.txt'.format(noise), 'w') as pf:
        pf.write(str(psx_list))
    label_errors_mask = get_noise_indices(s=y_train, psx=psx)
    with open('new_labelerrorsmask-{}n.txt'.format(noise), 'w') as ef:
        ef.write(str(list(label_errors_mask)))
    label_errors_mask_2 = get_noise_indices(s=y_train, psx=psx, sorted_index_method='normalized_margin')
    with open('new_labelerrorsmask_2-{}n.txt'.format(noise), 'w') as ef:
        ef.write(str(list(label_errors_mask_2)))

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
        classification(degree_vectors, degree_labels, 10)

    elif type == 'harmonic':
        harmonic_vectors, harmonic_labels = harmonic_centrality_feature(feature_dir)
        classification(harmonic_vectors, harmonic_labels, 35)

    elif type == 'katz':
        katz_vectors, katz_labels = katz_centrality_feature(feature_dir)
        classification(katz_vectors, katz_labels, 10)

    elif type == 'closeness':
        closeness_vectors, closeness_labels = closeness_centrality_feature(feature_dir)
        classification(closeness_vectors, closeness_labels, 30)

    else:
        print('Error Centrality Type!')

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('use time:', end-start)