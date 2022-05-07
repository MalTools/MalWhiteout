import csv
import numpy as np
import time
from collections import Counter
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities

noise_rate = 35
start_1 = time.time()
with open('results/%snoise/allsamples.txt'%noise_rate, 'r') as f:
    all_sample = eval(f.read())
print('len(all_sample)=',len(all_sample))
sample_label = {}  # 原始标签（不一定正确）goodware为0 malware为1
cert_apks = {}
with open('app-relation-new-{}%noise.csv'.format(noise_rate), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'apkname':
            continue
        sample_label[row[0]] = int(row[2])
        cert = row[1]
        if cert not in cert_apks.keys():
            cert_apks[cert] = [row[0]]
        else:
            cert_apks[cert].append(row[0])

sample_result = {}  # s:[0,1,0,1,0,1]
csbd_result = {}
drebin_result = {}
malscan_result = {}
mama_result = {}
noise_predic = {}
with open('results/csbd/{}noise/new_samples_csbd_{}n.txt'.format(noise_rate,noise_rate), 'r') as f:
    sample_csbd = f.read()
    sample_csbd = eval(sample_csbd)
with open('results/drebin/{}noise/new_samples_drebin_{}n.txt'.format(noise_rate, noise_rate), 'r') as f:
    sample_drebin = f.read()
    sample_drebin = eval(sample_drebin)
with open('results/malscan/{}noise/new_samples_malscan_{}n.txt'.format(noise_rate, noise_rate), 'r') as f:
    sample_malscan = f.read()
    sample_malscan = eval(sample_malscan)
# with open('results/mamadroid/{}noise/new_samples_mamadroid_{}n.txt'.format(noise_rate, noise_rate), 'r') as f:
#     sample_mama = f.read()
#     sample_mama = eval(sample_mama)

with open('results/csbd/{}noise/new_psx_csbd-{}n.txt'.format(noise_rate, noise_rate), 'r') as f:
    psx_csbd = f.read()
    psx_csbd = eval(psx_csbd)
with open('results/drebin/{}noise/new_psx_drebin-{}n.txt'.format(noise_rate, noise_rate), 'r') as f:
    psx_drebin = f.read()
    psx_drebin = eval(psx_drebin)
with open('results/malscan/{}noise/new_psx_malscan-{}n.txt'.format(noise_rate, noise_rate), 'r') as f:
    psx_malscan = f.read()
    psx_malscan = eval(psx_malscan)
# with open('results/mamadroid/{}noise/new_psx_mamadroid-{}n.txt'.format(noise_rate,noise_rate), 'r') as f:
#     psx_mama = f.read()
#     psx_mama = eval(psx_mama)
# print(len(sample_csbd))
for i in range(len(sample_csbd)):
    sample_name = sample_csbd[i].split('/')[-1][:-3]+'apk'
    csbd_result[sample_name] = psx_csbd[i]


for i in range(len(sample_drebin)):
    sample_name = sample_drebin[i].split('/')[-1][:-4]+'apk'
    drebin_result[sample_name] = psx_drebin[i]

for i in range(len(sample_malscan)):
    sample_name = sample_malscan[i].split('/')[-1][:-4]+'apk'
    malscan_result[sample_name] = psx_malscan[i]

# for i in range(len(sample_mama)):
#     sample_name = sample_mama[i].split('/')[-1][:-3]+'apk'
#     mama_result[sample_name] = psx_mama[i]

sample_label_new = {}  # sample_label_new包含用来做集成学习检测的样本，即在三个malware detection model中都有psx生成的样本
# print(len(csbd_result.keys()))
c = 0
for k in csbd_result.keys():
    # print(k)
    if k in drebin_result.keys() and k in malscan_result.keys():
        c += 1
        tuple_6 = csbd_result[k]+drebin_result[k]+malscan_result[k]
        sample_result[k] = tuple_6
        sample_label_new[k] = sample_label[k]  # sample_label包含全部的样本
print(c)
X_train_data = list(sample_result.values())
train_noisy_labels = list(sample_label_new.values())

X_train_data = np.array(X_train_data)
train_noisy_labels = np.array(train_noisy_labels)
# X_test = np.array(X_test)

# Wrap around any classifier. Yup, you can use sklearn/pyTorch/Tensorflow/FastText/etc.
lnl = LearningWithNoisyLabels(clf=LogisticRegression())
lnl.fit(X=X_train_data, s=train_noisy_labels)
# Estimate the predictions you would have gotten by training with *no* label errors.
# print(len(lnl.noise_mask))

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
# for n in list(lnl.noise_mask):
for n in list(label_errors_mask):
    if n == True:
        noise_predict.append(1)
        noise_predict_count += 1
    else:
        noise_predict.append(0)
samples = list(sample_result.keys())
for i in range(len(samples)):
    noise_predic[samples[i]] = noise_predict[i]

with open('results/{}noise/noiselabel.txt'.format(noise_rate), 'r') as f:
    noise_label = eval(f.read())  # 真实的噪声标签 是噪声为1 不是为0
noise_label_list = []
for sam in sample_result.keys():
    noise_label_list.append(noise_label[sam])
noise_count = 0
for v in noise_label.values():
    if v == 1:
        noise_count += 1
# print('len(noise_label_list)', len(noise_label_list))
for s in all_sample:
    if s not in sample_result.keys():  # 把不在sample_label_nwe中的样本补上，均预测为非噪声
        noise_predict.append(0)
        noise_label_list.append(noise_label[s])
        noise_predic[s] = 0

print('真实有多少个噪声noise_count=', noise_count)
print('len(noise_predict)=', len(noise_predict))
print('集成学习找出多少个噪声noise_predict_count=', noise_predict_count)
print('len(noise_label_list)', len(noise_label_list))

prec=precision_score(noise_label_list, noise_predict, average='binary', pos_label=1)
rec=recall_score(noise_label_list, noise_predict, average='binary', pos_label=1)
f1=f1_score(noise_label_list, noise_predict, average='binary', pos_label=1)
acc=accuracy_score(noise_label_list, noise_predict)
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
end_1 = time.time()
print('use time:', end_1-start_1)
# the following is noise correction through app relation
# group apks with the same cert:
noise_predic_2 = {}
for k,v in noise_predic.items():
    noise_predic_2[k] = v
for k,v in cert_apks.items():
    if len(v) > 1:  # 同一个cert有多个apk
        labels = [sample_label[apk] for apk in v]
        c = Counter(labels)
        malcount = c[1]
        if malcount > len(v)/3*2:  # 标签为mal的apk占多一半 --> 多于2/3 >len(v)/3*2
            for apk in v:
                if sample_label[apk] == 0:  # 如果原来label是0，就是噪声
                    noise_predic_2[apk] = 1
                else:
                    noise_predic_2[apk] = 0
        if malcount < len(v)/3:  # <len(v)/3
            for apk in v:
                if sample_label[apk] == 1:
                    noise_predic_2[apk] = 1
                else:
                    noise_predic_2[apk] = 0
# noise_predic_3 = {}
# for k in noise_predic.keys():
#     if noise_predic[k] == 1 and noise_predic_2[k] == 1:
#         noise_predic_3[k] = 1
#     else:
#         noise_predic_3[k] = 0
noise_predic_count2 = 0
for np in noise_predic.values():
    if np == 1:
        noise_predic_count2 += 1
print('app relation校正后噪声数量noise_predic_count2=', noise_predic_count2)
prec=precision_score(noise_label_list, list(noise_predic_2.values()), average='binary', pos_label=1)
rec=recall_score(noise_label_list, list(noise_predic_2.values()), average='binary', pos_label=1)
f1=f1_score(noise_label_list, list(noise_predic_2.values()), average='binary', pos_label=1)
acc=accuracy_score(noise_label_list, list(noise_predic_2.values()))
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
end_2 = time.time()
print('use time:', end_2 - start_1)
# 下面是将噪声标签反转 测试各个malware detection model的效果
# for k,v in noise_predic.items():
#     if v == 1:  # 说明该样本是噪声，将其样本标签进行反转
#         if sample_label[k] == 0:
#             sample_label[k] = 1
#         if sample_label[k] == 1:
#             sample_label[k] = 0
# with open('revised_sample_label_10.txt', 'w') as wf:
#     wf.write(str(sample_label))