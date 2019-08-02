#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 10:16
__author__ = 'wangwenhui'

import numpy as np
from gaXaj.XAJ import XAJ
from sce_uaXaj import sce_ua

def cceua(s, area,xx0, sf, bl, bu, icall, df):
    '''
    这是在单纯形中生成新点的子程序
    :param s:按目标函数值排好序的单纯形，即需要率定的参数列表
    :param sf: 目标函数值
    :param bl: 待率定参数的下界
    :param bu: 待率定参数的上界
    :param icall:
    :param maxn:
    :return:
    '''
    '''单纯形中生成新点的子程序'''
    nps, nopt = s.shape
    n = nps
    m = nopt
    alpha = 1.0
    beta = 0.5
    # 记录最好与最差的点
    bestx = s[0]
    bestfx = sf[0]
    worstx = s[-1]
    worstfx = sf[-1]
    # 计算单纯形质心（去掉最差点）
    ce = s[:n - 1, :].mean(axis=0)
    # 计算反射点
    snew = ce + alpha * (ce - worstx)
    # 检查是否出界
    ibound = 0
    s1 = snew - bl
    idx = np.nonzero(s1 < 0)
    if np.array(idx).size != 0:
        ibound = 1
    s1 = bu - snew
    idx = np.nonzero(s1 < 0)
    if np.array(idx).size != 0:
        ibound = 2
    if ibound >= 1:
        snew = bl + np.random.random((1, m)) * (bu - bl)
    df['Qp'] = XAJ(np.append(snew, [area]), xx0, df)
    fnew = sce_ua.caltarget(df['Qp'], df['Qr'])
    icall = icall + 1
    # 如果反射点失败，尝试收缩点
    if fnew > worstfx:
        snew = worstx + beta * (ce - worstx)
        df['Qp'] = XAJ(np.append(snew, [area]), xx0, df)
        fnew = sce_ua.caltarget(df['Qp'], df['Qr'])
        icall = icall + 1
        # 反射点和收缩点都失败了，尝试随机点
        if fnew > worstfx:
            snew = bl + np.random.random((1, nopt)) * (bu - bl)
            df['Qp'] = XAJ(np.append(snew, [area]), xx0, df)
            fnew = sce_ua.caltarget(df['Qp'], df['Qr'])
            icall = icall + 1
    return snew, fnew, icall
