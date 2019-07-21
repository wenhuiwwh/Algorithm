#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/21 9:09
# __author__ = 'wangwenhui'

from sklearn import preprocessing
from neupy import algorithms,environment
import pandas as pd



file_name="backfill_pipeline"

def read_csv():
    df=pd.read_csv("E:\Algorithm\gwo_grnn\data\%s.csv"%file_name,header=None,skiprows=1,index_col=0,engine="python")
    norm_eigen=preprocessing.minmax_scale(df.iloc[:,0:10])
    norm_target=preprocessing.minmax_scale(df.iloc[:,10])
    return df,norm_eigen,norm_target


def train_model(g):
    """训练模型
    Parameters:
    ----------
    g: 待优化的光滑因子
    return: 返回的是预测值与真实值
    """
    environment.reproducible()
    df,norm_eigen,norm_target=read_csv()
    x_train=norm_eigen[:10]
    y_train=norm_target[:10]
    x_test=norm_eigen[10:]
    y_test=norm_target[10:]
    gn=algorithms.GRNN(std=g)
    gn.train(x_train,y_train)
    y_predicted=gn.predict(x_train)
    return y_predicted,y_train


def test_model(g):
    '''验证学习后的PSO_GRNN结果
    Parameters:
    ----------
    g: 优化好的光滑因子
    return: 返回的是预测值
    '''
    df,norm_eigen,norm_target=read_csv()
    x_train=norm_eigen[:10]
    y_train=norm_target[:10]
    x_test=norm_eigen[10:]
    y_test=norm_target[10:]
    gn=algorithms.GRNN(std=g,verbose=False)
    gn.train(x_train,y_train)
    y_predicted=gn.predict(x_test)
    normalize_value=y_predicted*(df.iloc[:,10].max()-df.iloc[:,10].min())+df.iloc[:,10].min()
    return normalize_value