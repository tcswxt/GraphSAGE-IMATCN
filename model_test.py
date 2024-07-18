# -*- coding:utf-8 -*-
import os
import sys
from itertools import chain
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from get_data import setup_seed
from model_train import device
from models import GraphSAGE_IMATCN
import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
import statsmodels.api as sm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({"font.size": 28})  # 此处必须添加此句代码方可改变标题字体大小
plt.rcParams['axes.unicode_minus'] = False

setup_seed(123)

def test(args, Dte, scaler, path):
    print('loading models...')

    model = GraphSAGE_IMATCN(args).to(device)
    model.load_state_dict(torch.load(path + '/GraphSAGE-IMATCN/models/GraphSAGE-IMATCN.pkl')['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for graph in tqdm(Dte):
        graph = graph.to(device)
        _pred, targets = model(graph)
        targets = np.array(targets.data.tolist())
        for i in range(args.input_size):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        for i in range(_pred.shape[0]):
            pred = _pred[i]
            pred = list(chain.from_iterable(pred.data.tolist()))
            preds[i].extend(pred)

    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T

    error = []
    for i in range(len(ys)):
        error.append(preds[i] - ys[i])

    print(preds.shape)
    mses, rmses, maes, sees = [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        if ind == 7:
            print('--------------------------------')
            # print('第', str(ind + 1), '个变量:')
            print('r2:', get_r2(y, pred))
            print('rmse:', get_rmse(y, pred))
            print('mae:', get_mae(y, pred))
            print('see:', calculate_standard_error(y, pred))
            mses.append(get_mse(y, pred))
            rmses.append(get_rmse(y, pred))
            maes.append(get_mae(y, pred))
            sees.append(calculate_standard_error(y, pred))
            print('--------------------------------')


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def calculate_standard_error(x, y):
    # 添加截距项
    x = sm.add_constant(x)
    # 拟合线性回归模型
    model = sm.OLS(y, x)
    results = model.fit()
    # 获取估计标准误差
    standard_error = np.sqrt(results.mse_resid)
    return standard_error
