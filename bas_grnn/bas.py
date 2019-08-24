#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 10:17
#__author__ = 'wangwenhui'
import numpy as np
from bas_grnn.test import Michalewicz,Goldsteiin_Price
from bas_grnn.grnn import train_grnn



def fitness(xbest):
    """适应度函数"""
    f=train_grnn(xbest)
    return f


def bas(ub,lb,eta=0.95,c=5,step=1,n=200,k=1,eps=1e-8):
    """初始化参数
    Parameters
    ----------
    ub:搜索空间的上界
    lb:搜索空间的下界
    eta: 步长精度
    c:常数
    step: 天牛步长
    n: 最大迭代次数
    k: 维度
   """
    x=np.random.uniform(lb,ub,size=(1,k)).flatten()
    xbest=x
    fbest=fitness(xbest)
    for i in range(n):

        # 若搜索位置超过了搜索区间，则重新回到搜索空间
        for i in range(x.shape[0]):
            if x[i]>ub:
                x[i]=ub
            if x[i]<lb:
                x[i]=lb

        # 天牛两须之间的距离
        d0=step/c
        # 天牛朝向的随机向量
        dirs = np.random.uniform(-1, 1,k)
        dirs = dirs / (np.linalg.norm(dirs)+eps)
        # 天牛左须的位置
        xleft=x+dirs*d0/2
        fleft=fitness(xleft)
        # 天牛右须的位置
        xright=x-dirs*d0/2
        fright=fitness(xright)
        # 判断向左移动还是向右移动
        x=x-step*dirs*np.sign(fleft-fright)
        f=fitness(x)
        if f<fbest:
            xbest=x
            fbest=f
        step*=eta
    return xbest,fbest

