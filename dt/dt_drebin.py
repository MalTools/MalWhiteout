import numpy as np
import torch
import csv
import random
import time
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  # 保证sess.run()能够正常运行
from utils.common import get_config
from dt.find_outlier import vote
from dt.pytorchtools import EarlyStopping as es
from dt.mlp_model import MLP, MLP_DATASET
import sys
from scipy import sparse
# from sklearn.decomposition import PCA
# app_vec = {}  # 每个app的特征向量
# app_label_origin = {}  # 每个app的原始label, gooddir:0, malware:1
# app_label_new = {}  # 每一轮 每个app的label

# x_train, y_train, sample_list = Predict_cleanlab(MalDir, GoodDir, True, NumFeatForExp)
with open('dt/dt_samples-20n.txt', 'r') as f:
    sample_list = eval(f.read())

y_train = np.load('dt/dt_ytrain_20n.npy')
x_train = sparse.load_npz('dt/dt_xtrain_20n.npz')
x_train = x_train.toarray()
print(type(x_train), type(y_train))
print('x_train.shape=', x_train.shape, 'y_train.shape', y_train.shape)
# pca = PCA(n_components=1000)
# low_x_train = pca.fit_transform(x_train)
# print(low_x_train.shape)
print("***********get (x_train, y_train, sample_list)*************")

input_num = x_train.shape[1]
print('input_num=', input_num)
app_vec= dict(zip(sample_list, list(x_train)))
app_label_origin= dict(zip(sample_list, list(y_train)))
app_label_new = dict(zip(sample_list, list(y_train)))
# with open('dt/noise_predict_my25_time_nopca/dt-noise-predict-142.txt') as f:
#     nnn = eval(f.read())
# app_label_new = {}
# for k,v in app_label_origin.items():
#     if nnn[k] == 0:
#         app_label_new[k] = v
#     else:
#         if v == 1:
#             app_label_new[k] = 0
#         else:
#             app_label_new[k] = 1
print(len(app_vec), len(app_label_origin))
print('**************data OK*****************************')

def mutate_label(app_name):
    if app_label_new[app_name] == 0:
        return 1
    if app_label_new[app_name] == 1:
        return 0

def train_2(app_feature, ds_idx):
    # filter_warnings()
    # seed_everything(config.seed)
    train_data = MLP_DATASET.get_data(app_feature, app_label_new)
    train_dataset = MLP_DATASET(train_data)
    config = get_config('CWE119', 'stack')
    model = MLP(config, ds_idx=ds_idx, input=input_num)
    print("###################dataset len########", len(train_dataset))
    model.fit(train_dataset=train_dataset)
    # import copy
    # loss_dict = copy.deepcopy(model.loss_dict)
    # del model
    return model.loss_dict

def concatenate_data(wds, dds):
    """
    @description  :获取离群训练所需要的数据
    @param  :
    wds ws中ds的loss vector
    dds ds中ds的loos vector

    @Returns  : 对于ds中的每个app 拼接后的loss vector
    """
    X_train = []
    Y_train = []
    # ds_flipped = []
    ds_ids = []
    for app in dds.keys():
        x = wds[app]
        x.extend(dds[app])
        X_train.append(x)
        Y_train.append(app_label_new[app])
        ds_ids.append(app)

    return X_train, Y_train, ds_ids

def differential_training_one(ws, downsample_count, vote_rate):
    """
    @description  : train once df
    ---------
    @param:
    -------
    @Returns :
    -------
    """
    # ds = np.random.choice(ws, downsample_count, replace=False)
    ds = {}
    sample_keys = random.sample(list(ws.keys()), downsample_count)
    for k in sample_keys:
        ds[k] = ws[k]

    # wds_loss = train(ws)
    # dds_loss = train(ds)
    wds_loss = train_2(ws, sample_keys)
    dds_loss = train_2(ds, sample_keys)

    # log
    # w_value_length = []
    # for v in wds_loss.values():
    #     w_value_length.append(len(v))
    # d_value_length = []
    # for v in dds_loss.values():
    #     d_value_length.append(len(v))
    X_train, Y_train, app_ids = concatenate_data(wds_loss, dds_loss)
    outlier_list = vote(X_train, Y_train, vote_rate, app_ids)

    for app in ds.keys():
        random_x = random.randint(0, 1)
        if app in outlier_list and random_x > 0.5:  # 以50%的概率翻转标签
            # app_flip[app] += 0.5
            app_label_new[app] = mutate_label(app)
            # if app_label_new[app] != app_label_origin[app]:
            #     noise_predic[app] = 1
        # else:
        #     if app_flip[app] > 0:
        #         app_flip[app] -= 0.5
    # print('********len(outlier_list) / len(ds)=', len(outlier_list) / len(ds))

    noise_cnt = 0
    tot = 0
    for app in app_label_origin.keys():
        tot += 1
        if app_label_origin[app] != app_label_new[app]:
            noise_cnt += 1
    return noise_cnt / tot

    # return len(outlier_list) / len(ds)

def differential_training():
    ws = app_vec
    early_stopping = es(patience=7, verbose=False, delta=0.001)
    noise_ratio_1 = differential_training_one(ws, downsample_count=int(len(app_vec) * 0.06),
                                            vote_rate=0.7)  # int(len(app_vec)*0.06)

    for iter in range(1, 200):
        print('@@@@@@@@@iteration =', iter)
        # iter_start = time.time()
        noise_ratio_2 = differential_training_one(ws, downsample_count=int(len(app_vec)*0.06), vote_rate=0.7)  # int(len(app_vec)*0.06)
        # iter += 1
        noise_ratio_loss = noise_ratio_2 - noise_ratio_1
        early_stopping(abs(noise_ratio_loss), None)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Differential training early stopping,", "iter=", iter)
            # 结束模型训练
            break
        noise_ratio_1 = noise_ratio_2
        # iter_end = time.time()
        # use_second = iter_end - iter_start
        # with open('iter-time-my5.txt', 'a') as f:
        #     f.write('iter %s: %s minutes\n' % (iter,  round((use_second/60),4)))
        # for k, v in app_flip.items():
        #     if v > 0.5:  # app_flip>0.5 认为是噪声
        #         noise_predic[k] = 1
        # with open('dt/app_flip/af-%s.txt'%iter, 'w') as f:
        #     f.write(str(app_flip))
        noise_predic = {}
        for app in app_label_origin.keys():
            if app_label_origin[app] == app_label_new[app]:
                noise_predic[app] = 0
            else:
                noise_predic[app] = 1
        with open('dt/noise_predict_my20_time_nopca/dt-noise-predict-%s.txt'%iter, 'w') as wf:
            wf.write(str(noise_predic))



if __name__ == '__main__':
    start = time.time()
    # with open('dt/dt-noise-predict-178.txt', 'r') as f:
    #     eval(f.read())
    # train(app_vec)
    differential_training()
    end = time.time()
    print('use time (hours) =', (end-start)/3600)
