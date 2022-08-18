import csv
import numpy as np
import time
from collections import Counter
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities

noise_rate = '10n'
start = time.time()
with open('results/%s/allsamples.txt'%noise_rate, 'r') as f:
    all_sample = eval(f.read())
print('len(all_sample)=',len(all_sample))

sample_label = {}  # orginal labels which may be incorrect; goodware->0, malware->1
cert_apks = {}
common_keys = ['61ed377e85d386a8dfee6b864bd85b0bfaa5af81', '27196e386b875e76adf700e7ea84e4c6eee33dfa', '5b368cff2da2686996bc95eac190eaa4f5630fe5']

with open('app-relation-{}.csv'.format(noise_rate), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'apkname':
            continue
        sample_label[row[0]] = int(row[2])
        cert = row[1]
        if cert in common_keys:
            continue
        if cert not in cert_apks.keys():
            cert_apks[cert] = [row[0]]
        else:
            cert_apks[cert].append(row[0])

sample_result = {}  # s:[0,1,0,1,0,1]
csbd_result = {}
drebin_result = {}
malscan_result = {}

with open('results/{}/samples_csbd_{}.txt'.format(noise_rate,noise_rate), 'r') as f:
    sample_csbd = f.read()
    sample_csbd = eval(sample_csbd)
with open('results/{}/samples_drebin_{}.txt'.format(noise_rate, noise_rate), 'r') as f:
    sample_drebin = f.read()
    sample_drebin = eval(sample_drebin)
with open('results/{}/samples_malscan_{}.txt'.format(noise_rate, noise_rate), 'r') as f:
    sample_malscan = f.read()
    sample_malscan = eval(sample_malscan)

with open('results/{}/psx_csbd-{}.txt'.format(noise_rate, noise_rate), 'r') as f:
    psx_csbd = f.read()
    psx_csbd = eval(psx_csbd)
with open('results/{}/psx_drebin-{}.txt'.format(noise_rate, noise_rate), 'r') as f:
    psx_drebin = f.read()
    psx_drebin = eval(psx_drebin)
with open('results/{}/psx_malscan-{}.txt'.format(noise_rate, noise_rate), 'r') as f:
    psx_malscan = f.read()
    psx_malscan = eval(psx_malscan)


for i in range(len(sample_csbd)):
    sample_name = sample_csbd[i].split('/')[-1][:-3]+'apk'
    csbd_result[sample_name] = psx_csbd[i]
for i in range(len(sample_drebin)):
    sample_name = sample_drebin[i].split('/')[-1][:-4]+'apk'
    drebin_result[sample_name] = psx_drebin[i]
for i in range(len(sample_malscan)):
    sample_name = sample_malscan[i].split('/')[-1][:-4]+'apk'
    malscan_result[sample_name] = psx_malscan[i]

# "sample_label_new" contains samples used for ensemble learning, i.e., samples have psx from three malware detection models.
sample_label_new = {}

c = 0
# Use three models for ensemble learning
# for k in csbd_result.keys():
#     if k in drebin_result.keys() and k in malscan_result.keys():
#         c += 1
#         tuple_6 = csbd_result[k]+drebin_result[k]+malscan_result[k]
#         sample_result[k] = tuple_6
#         sample_label_new[k] = sample_label[k]  # "sample_label" contains all the smaples

# You can also choose two of the three models for ensemble, e.g., use drebin and malscan
print('Use two models (drebin and malscan) for ensemble.')
for k in drebin_result.keys():
    if k in malscan_result.keys():
        c += 1
        tuple_4 = drebin_result[k]+malscan_result[k]
        if k in sample_label.keys():
            sample_result[k] = tuple_4
            sample_label_new[k] = sample_label[k]  # sample_label包含全部的样本
print('How many samples are included in ensemble:', c)

# "nose_label" is a dict recording whether each sample is a true noise.(noise->1, non-noise->0)
with open('results/{}/noiselabel.txt'.format(noise_rate), 'r') as f:
    noise_label = eval(f.read())
noise_count = 0
for v in noise_label.values():
    if v == 1:
        noise_count += 1

noise_label_list = []
for sam in sample_result.keys():
    noise_label_list.append(noise_label[sam])
for s in all_sample:
    if s not in sample_result.keys():  # samples not in "sample_label_new"
        noise_label_list.append(noise_label[s])

def cl_predict():
    noise_predic = {}

    X_train_data = list(sample_result.values())
    train_noisy_labels = list(sample_label_new.values())

    X_train_data = np.array(X_train_data)
    train_noisy_labels = np.array(train_noisy_labels)

    # Wrap around LogisticRegression classifier.
    lnl = LearningWithNoisyLabels(clf=LogisticRegression())
    lnl.fit(X=X_train_data, s=train_noisy_labels)

    noise_predict = []
    noise_predict_count = 0
    psx = estimate_cv_predicted_probabilities(
        X=X_train_data,
        labels=train_noisy_labels,
        clf=lnl.clf,
        cv_n_folds=lnl.cv_n_folds,
        seed=lnl.seed,
    )
    label_errors_mask = get_noise_indices(s=train_noisy_labels, psx=psx)

    for n in list(label_errors_mask):
        if n == True:
            noise_predict.append(1)
            noise_predict_count += 1
        else:
            noise_predict.append(0)
    samples = list(sample_result.keys())
    for i in range(len(samples)):
        noise_predic[samples[i]] = noise_predict[i]

    drebin_sample_label = {}
    with open('results/{}/drebin_labelerrorsmask-{}.txt'.format(noise_rate, noise_rate), 'r') as f:
        labelmask_drebin = eval(f.read())
    for i in range(len(sample_drebin)):
        sample_name = sample_drebin[i].split('/')[-1][:-4] + 'apk'
        if labelmask_drebin[i] is True:
            drebin_sample_label[sample_name] = 1
        else:
            drebin_sample_label[sample_name] = 0

    nofeature = 0
    for s in all_sample:
        if s not in sample_result.keys():  # samples not in "sample_label_new"
            # You can use the predict result of drebin if available (since drebin can extract most of the samples features).
            if s in drebin_sample_label.keys():
                noise_predict.append(drebin_sample_label[s])
                noise_predic[s] = drebin_sample_label[s]
                if drebin_sample_label[s] == 1:
                    noise_predict_count += 1
            else:
                nofeature += 1
                noise_predict.append(0)
                noise_predic[s] = 0

    print('No feature extracted by all of the models:', nofeature)
    print('How many true noises:', noise_count)
    print('len(noise_predict):', len(noise_predict))
    print('How many noises estimated by ensemble learning:', noise_predict_count)

    prec = precision_score(noise_label_list, noise_predict, average='binary', pos_label=1)
    rec = recall_score(noise_label_list, noise_predict, average='binary', pos_label=1)
    f1 = f1_score(noise_label_list, noise_predict, average='binary', pos_label=1)
    acc = accuracy_score(noise_label_list, noise_predict)
    print('precision:', prec)
    print('recall:', rec)
    print('f1:', f1)
    print('Accuracy:', acc)

    TP1 = 0
    FP1 = 0
    for i in range(len(noise_predict)):
        if noise_predict[i] == 1:
            if noise_label_list[i] == 1:
                TP1 += 1
            else:
                FP1 += 1
    print('TP1=', TP1, 'FP1=', FP1)
    end = time.time()
    print('use time:', end - start)
    return noise_predic


def app_relation_calibration():
    noise_predic = cl_predict()
    # the following is noise correction through app relation
    # group apks with the same cert:
    noise_predic_2 = {}
    for k, v in noise_predic.items():
        noise_predic_2[k] = v
    for k, v in cert_apks.items():
        # A cert corresponds to multiple apks
        if len(v) > 1:
            labels = [sample_label[apk] for apk in v]
            c = Counter(labels)
            malcount = c[1]
            # Malware samples account for more than 2/3
            if malcount > len(v) / 3 * 2:
                for apk in v:
                    if apk not in noise_predic.keys():
                        continue
                    # If the original label is 0, take it as noise.
                    if sample_label[apk] == 0:
                        noise_predic_2[apk] = 1
                    else:
                        noise_predic_2[apk] = 0
            # Malware samples account for less than 1/3
            if malcount < len(v) / 3:
                for apk in v:
                    if apk not in noise_predic.keys():
                        continue
                    # If the original label is 1, take it as noise.
                    if sample_label[apk] == 1:
                        noise_predic_2[apk] = 1
                    else:
                        noise_predic_2[apk] = 0

    noise_predic_count2 = 0
    for np in noise_predic_2.values():
        if np == 1:
            noise_predic_count2 += 1

    print('How many noises after app relation calibration:', noise_predic_count2)
    prec = precision_score(noise_label_list, list(noise_predic_2.values()), average='binary', pos_label=1)
    rec = recall_score(noise_label_list, list(noise_predic_2.values()), average='binary', pos_label=1)
    f1 = f1_score(noise_label_list, list(noise_predic_2.values()), average='binary', pos_label=1)
    acc = accuracy_score(noise_label_list, list(noise_predic_2.values()))
    print('precision:', prec)
    print('recall:', rec)
    print('f1:', f1)
    print('Accuracy:', acc)

    TP2 = 0
    FP2 = 0
    for i in range(len(list(noise_predic_2.values()))):
        if list(noise_predic_2.values())[i] == 1:
            if noise_label_list[i] == 1:
                TP2 += 1
            else:
                FP2 += 1
    print('TP2=', TP2, 'FP2=', FP2)
    end = time.time()
    print('use time:', end - start)

if __name__ == '__main__':
    noise_predic = cl_predict()
    app_relation_calibration()

