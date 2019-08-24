#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 20:26
__author__ = 'wangwenhui'

import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from neupy import algorithms



def read_data():
    """读取数据"""
    train_path = "bas_grnn/bikedata/train.csv"
    test_path="bas_grnn/bikedata/test.csv"
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data=pd.read_csv(test_path,encoding='utf-8')
    return train_data,test_data


def feature_engineering():
    """特征工程，归一化"""
    train_data,test_data = read_data()
    train_data.drop('id',axis=1,inplace=True)
    test_data.drop('id',axis=1,inplace=True)
    train_data= pd.DataFrame(preprocessing.minmax_scale(train_data))
    return train_data,test_data

def get_k_fold_data(k, i, x, y):
    """k折交叉验证"""
    assert k > 1
    fold_size = x.shape[0] // k
    x_valid = x[i * fold_size:(i + 1) * fold_size]
    y_valid = y[i * fold_size:(i + 1) * fold_size]
    if i == 0:
        x_train = x[(i + 1) * fold_size:]
        y_train = y[(i + 1) * fold_size:]
    elif i == k - 1:
        x_train = x[:i * fold_size]
        y_train = y[:i * fold_size]
    else:
        x_train = pd.concat([x[:i * fold_size], x[(i + 1) * fold_size:]])
        y_train = pd.concat([y[:i * fold_size], y[(i + 1) * fold_size:]])
    return x_train, y_train, x_valid, y_valid


def train_grnn(g,k=4):
    """训练模型"""
    train_data,test_data=feature_engineering()
    x=train_data.iloc[:,:-1]
    y=train_data.iloc[:,-1]
    rmse_list=[]
    for i in range(k):
        x_train,y_train,x_test,y_test=get_k_fold_data(k,i,x,y)
        gn=algorithms.GRNN(std=g)
        gn.train(x_train,y_train)
        y_predicted=gn.predict(x_test)
        rmse=math.sqrt(np.mean((y_predicted-np.array(y_test))**2))
        rmse_list.append(rmse)
    return sum(rmse_list)/len(rmse_list)


def test_grnn(g):
    """预测模型"""
    train_data,test_data=feature_engineering()
    gn=algorithms.GRNN(std=g)
    x=train_data.iloc[:,:-1]
    y=train_data.iloc[:,-1]
    gn.train(x,y)
    y_predicted=gn.predict(test_data)
    return y_predicted


if __name__ == '__main__':
    pass