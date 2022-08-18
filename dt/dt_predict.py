import numpy as np
import torch
import csv
import random
import time
from utils.common import get_config
from dt.find_outlier import vote
from dt.pytorchtools import EarlyStopping as es
from dt.mlp_model import MLP, MLP_DATASET
import sys
from scipy import sparse
from sklearn.decomposition import PCA


with open('dt/dt_samples-5n.txt', 'r') as f:
    sample_list = eval(f.read())

y_train = np.load('dt/dt_ytrain_5n.npy')
x_train = sparse.load_npz('dt/dt_xtrain_5n.npz')
x_train = x_train.toarray()
print(type(x_train), type(y_train))
print('x_train.shape=', x_train.shape, 'y_train.shape', y_train.shape)
pca = PCA(n_components=1000)
low_x_train = pca.fit_transform(x_train)
print(low_x_train.shape)
print("***********get (x_train, y_train, sample_list)*************")

input_num = low_x_train.shape[1]
print('input_num=', input_num)
app_vec= dict(zip(sample_list, list(low_x_train)))

app_label_origin= dict(zip(sample_list, list(y_train))) # original labels; gooddir:0, malware:1
app_label_new = dict(zip(sample_list, list(y_train))) # labels after each iteration
print(len(app_vec), len(app_label_origin))
print('**************data prepared*****************************')

def mutate_label(app_name):
    if app_label_new[app_name] == 0:
        return 1
    if app_label_new[app_name] == 1:
        return 0

def train(app_feature, ds_idx):
    # filter_warnings()
    # seed_everything(config.seed)
    train_data = MLP_DATASET.get_data(app_feature, app_label_new)
    train_dataset = MLP_DATASET(train_data)
    config = get_config(model='config')
    model = MLP(config, ds_idx=ds_idx, input=input_num)
    print("###################dataset length########", len(train_dataset))
    model.fit(train_dataset=train_dataset)
    # import copy
    # loss_dict = copy.deepcopy(model.loss_dict)
    # del model
    return model.loss_dict

def concatenate_data(wds, dds):
    """
    @description  :Get data for outlier analysis
    @param  :
    wds: loss vectors of ds-samples in ws
    dds: loos vectors of ds-samples in ds

    @Returns  : the concatenated loss vector for each app in DS
    """
    X_train = []
    Y_train = []
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

    wds_loss = train(ws, sample_keys)
    dds_loss = train(ds, sample_keys)

    X_train, Y_train, app_ids = concatenate_data(wds_loss, dds_loss)
    outlier_list = vote(X_train, Y_train, vote_rate, app_ids)

    for app in ds.keys():
        random_x = random.randint(0, 1)
        # Flip the label with a 50% probability
        if app in outlier_list and random_x > 0.5:

            app_label_new[app] = mutate_label(app)

    noise_cnt = 0
    tot = 0
    for app in app_label_origin.keys():
        tot += 1
        if app_label_origin[app] != app_label_new[app]:
            noise_cnt += 1
    return noise_cnt / tot

def differential_training():
    ws = app_vec
    early_stopping = es(patience=7, verbose=False, delta=0.0005)
    noise_ratio_1 = differential_training_one(ws, downsample_count=int(len(app_vec) * 0.06),
                                            vote_rate=0.7)

    for iter in range(1, 200):
        print('@@@@@@@@@iteration =', iter)
        noise_ratio_2 = differential_training_one(ws, downsample_count=int(len(app_vec)*0.06), vote_rate=0.7)
        noise_ratio_loss = noise_ratio_2 - noise_ratio_1
        early_stopping(abs(noise_ratio_loss), None)
        # Stopping Criterion
        if early_stopping.early_stop:
            print("Differential training early stopping,", "iter=", iter)
            # End the model training
            break
        noise_ratio_1 = noise_ratio_2

        noise_predic = {}
        for app in app_label_origin.keys():
            if app_label_origin[app] == app_label_new[app]:
                noise_predic[app] = 0
            else:
                noise_predic[app] = 1
        with open('dt/noise_predict/dt-noise-predict-%s.txt'%iter, 'w') as wf:
            wf.write(str(noise_predic))


if __name__ == '__main__':
    start = time.time()
    differential_training()
    end = time.time()
    print('use time (hours) =', (end-start)/3600)
