#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/5/12 10:16
__author__ = 'wangwenhui'

import numpy as np
import math
import random
from sce_ua_xaj import cce_ua
from ga_xaj.XAJ import XAJ


def caltarget(predict, acutal):
    '''目标函数'''
    rmse = math.sqrt(np.mean((predict - acutal) ** 2))
    return rmse



class SceUa:

    def __init__(self, df, xx, area, xx0, bl, bu, maxn, kstop, pcento, peps, ngs, iseed, iniflg):
        '''
        :param xx: 待优化参数的初始值
        :param bl: 待优化参数的下限
        :param bu: 待优化参数的上限
        :param maxn: 进化过程中函数最大调用次数
        :param kstop: 集中之前最大演化数
        :param pcento: 允许在kstop循环前收敛的百分比变化
        :param peps:
        :param ngs: 参与进化的复合形数（即有多少个子复合形）
        :param iseed: 生成随机数种子
        :param iniflg: 初始参数矩阵的标志
        '''
        self.xx = xx
        self.area = area
        self.xx0 = xx0
        self.df = df
        self.bl = bl
        self.bu = bu
        self.maxn = maxn
        self.kstop = kstop
        self.pcento = pcento
        self.peps = peps
        self.ngs = ngs
        self.iseed = iseed
        self.iniflg = iniflg

    def gnrng(self, x, bound):
        '''计算参数的标准化几何范围'''
        gnrng = np.exp(np.mean(np.log((x.max(axis=0) - x.min(axis=0)) / bound)))
        return gnrng

    def sce_ua(self):
        global BESTX, BESTF, ICALL, PX, PF
        # 点的维度
        nopt = len(self.xx)
        # 每个复合形顶点数
        npg = 2 * nopt + 1
        # 每个子复合形顶点数
        nps = nopt + 1
        # 每个复合形进化的迭代数
        nspl = npg
        # 进化过程复合形的最小数目
        # mings = self.ngs
        # 复合形总的点数
        npt = npg * self.ngs
        # 计算上下界之差
        bound = self.bu - self.bl
        x = np.zeros((npt, nopt))
        for i in range(npt):
            x[i] = self.bl + np.random.random((1, nopt)) * bound
        # 是否包含xx
        if self.iniflg == 1:
            x[0] = self.xx
        # 计算函数值
        icall = 0
        xf = np.zeros(npt)
        for i in range(npt):
            self.df['Qp'] = XAJ(np.append(x[i], [self.area]), self.xx0, self.df)
            xf[i] = caltarget(self.df['Qp'], self.df['Qr'])
            icall += 1
        # idx为由小到大排序的索引，xf为排序后的结果
        idx = xf.argsort()
        xf.sort()
        x = x[idx]
        # 记录最好最差的点
        bestx = x[0]
        bestfx = xf[0]
        worstx = x[-1]
        worstfx = xf[-1]
        BESTX = bestx
        BESTF = bestfx
        ICALL = icall
        # 计算标准差
        # xnstd = x.std(axis=0)
        # 计算收敛指标
        gnrng = self.gnrng(x, bound)
        print("The Initial Loop: 0")
        print("BESTFX: ", bestfx)
        print("BESTX: ", bestx)
        print("WORSTF: ", worstfx)
        print("WORSTX:", worstx)

        # 检查是否收敛
        if icall >= self.maxn:
            print("优化搜索因最大数量的限制而终止试验")
            print(self.maxn)
            print("已经超过，搜索是停在试验数：")
            print(icall)
            print("初始循环！")
        if gnrng < self.peps:
            print("种群已聚到一个预先设定的小参数空间")

        # 开始进化循环
        nloop = 0
        criter = []
        criter_change = 1e5
        # 存储每次迭代的最优值与最差值
        bestfx_list = []
        while icall < self.maxn and gnrng > self.peps and criter_change > self.pcento:
            nloop += 1
            # 循环复合形(子形)
            for igs in range(self.ngs):
                # 将群体划分为复合形（子群体）
                k1 = np.arange(npg)
                k2 = k1 * self.ngs + igs
                cx = np.zeros((len(k1), nopt))
                cx[k1] = x[k2]
                cf = np.zeros(len(k1))
                cf[k1] = xf[k2]
                # 针对nspl步骤演化子复合形
                for loop in range(nspl):
                    # 根据线性概率分布对复数进行采样来选择单纯形
                    lcs = [0]
                    for k3 in range(1, nps):
                        for iter in range(1000):
                            lpos = math.floor(
                                npg + 0.5 - math.sqrt((npg + 0.5) ** 2 - npg * (npg + 1) * random.random()))
                            # idx = np.nonzero(lcs[:k3] == lpos)
                            # if np.array(idx).size == 0:
                            if lpos not in lcs[:k3]:
                                break
                        lcs.append(lpos)
                    lcs.sort()
                    # 构建单纯形
                    # s = np.zeros((nps, nopt))
                    s = cx[lcs]
                    sf = cf[lcs]
                    snew, fnew, icall = cce_ua.cceua(s, self.area, self.xx0, sf, self.bl, self.bu, icall, self.df)
                    # 用新点替换单纯形中的最差点
                    s[nps - 1] = snew
                    sf[nps - 1] = fnew
                    # 将单纯形替换为复合形
                    cx[lcs] = s
                    cf[lcs] = sf
                    # 单纯形排序
                    idx = cf.argsort()
                    cf.sort()
                    cx = cx[idx]
                # 将复合形替换为群体
                x[k2] = cx[k1]
                xf[k2] = cf[k1]

            idx = xf.argsort()
            xf.sort()
            x = x[idx]
            PX = x
            PF = xf
            # 记录最好和最差点
            bestx = x[0]
            bestfx = xf[0]
            print(bestx)
            worstx = x[-1]
            worstfx = xf[-1]
            BESTX = np.vstack((BESTX, bestx))
            BESTF = np.vstack((BESTF, bestfx))
            ICALL = np.vstack((ICALL, icall))
            # 计算每个参数的标准差
            xnstd = x.std(axis=0)
            gnrng = self.gnrng(x, bound)
            print("Evolultion Loop:%s,- Trial- %s" % (nloop, icall))
            print("BESTFX: ", bestfx)
            print("BESTX: ", bestx)
            print("WORSTF: ", worstfx)
            print("WORSTX: ", worstx)
            bestfx_list.append(bestfx)
            # 检查是否收敛
            if icall > self.maxn:
                print("%s已达最大调用次数: " % self.maxn)
            if gnrng < self.peps:
                print("种群已进化到了一个预料中的小范围")
            criter.append(bestfx)
            if nloop >= self.kstop:
                criter_change = abs(criter[nloop - 1] - criter[nloop - self.kstop]) * 100
                criter_change = criter_change / (
                        sum([abs(i) for i in criter[nloop - self.kstop:]]) / len(criter[nloop - self.kstop:]))
                if criter_change < self.pcento:
                    print("已非常接近最优点，循环次数为:{}，阈值下界为:{:.2f}%".format(self.kstop, self.pcento))
                    print("已基于目标功能标准实现了收敛!")

        print("计算次数:%s" % icall)
        print("归一化几何指数:%s" % gnrng)
        print("最好点较上个点进化程度为:{:.2f},阈值下界为:{:.2f}%".format(self.kstop, criter_change))
        return icall,bestx, bestfx_list

if __name__ == '__main__':
    pass
