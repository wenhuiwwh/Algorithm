#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 13:53
__author__ = 'wangwenhui'

import pickle
import numpy as np
import pandas as pd
import os
import math
from sce_uaXaj.sce_ua import SceUa
from matplotlib import pyplot as plt
import matplotlib as mpl
from gaXaj.XAJ import XAJ, QJcal

mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.figsize'] = (12.0, 4.8)



class XinAnRiver:
    def __init__(self, file_name, df, area, xx0, icall, bestx, bestfx_list):
        '''
        :param file_name: 要读取的文件名
        :param df: 读成数据框形式
        :param area: xx的第16个参数
        :param xx0: 模型初始值
        :param icall: 模型的计算次数
        :param bestx: 最优参数
        :param bestfx_list: 历次计算的最优值RMSE列表
        '''
        self.file_name = file_name
        self.df = df
        self.area = area
        self.xx0 = xx0
        self.icall = icall
        self.bestx = bestx
        self.bestfx_list = bestfx_list

    def save_result(self):
        '''存储每次运算结果的文件名，计算次数，RMSE和率定的参数值'''
        best_df = pd.DataFrame()
        best_df['file_name'] = [self.file_name]
        best_df['icall'] = [self.icall]
        best_df['RMSE'] = self.bestfx_list[-1]
        best_df['K'] = [self.bestx[0]]
        best_df['WUM'] = [self.bestx[1]]
        best_df['WLM'] = [self.bestx[2]]
        best_df['C'] = [self.bestx[3]]
        best_df['WDM'] = [self.bestx[4]]
        best_df['B'] = [self.bestx[5]]
        best_df['IMP'] = [self.bestx[6]]
        best_df['SM'] = [self.bestx[7]]
        best_df['EX'] = [self.bestx[8]]
        best_df['KG'] = [self.bestx[9]]
        best_df['KSS'] = [self.bestx[10]]
        best_df['KKG'] = [self.bestx[11]]
        best_df['KKSS'] = [self.bestx[12]]
        best_df['CS'] = [self.bestx[13]]
        best_df['L'] = [self.bestx[14]]
        best_df['U'] = [self.area]
        if os.path.exists('result/param.csv'):
            best_df.to_csv('result/param.csv', mode='a+', header=0, index=0)
        else:
            best_df.to_csv('result/param.csv', index=0)
        return None

    def draw_rmse(self):
        '''绘制目标函数(RMSE)随迭代次数的变化情况曲线图'''
        x = range(len(self.bestfx_list))
        best_y = self.bestfx_list
        plt.title("Iterative graph of SCE_UA")
        plt.plot(x, best_y, color='blue', label='best_value', lw=1.0)
        plt.xlabel('iteration number')
        plt.ylabel('RMSE value')
        plt.xlim(xmin=0)
        plt.tick_params(direction='in')
        plt.legend(fontsize=8, edgecolor='black', loc='best')
        figure_fig = plt.gcf()
        plt.rcParams['savefig.dpi'] = 800
        figure_fig.savefig('result/%s_rmse.png'%self.file_name, format='png')
        return None

    def draw_value(self):
        '''绘制预测值与真实值之间的对比图'''
        self.df['Qp'] = XAJ(np.append(self.bestx, self.area), self.xx0, self.df)
        self.df['Qp'].plot(color='red', label='predict_value', lw=1.0)
        self.df['Qr'].plot(color='blue', label='real_value', lw=1.0)
        plt.title('Compare View')
        plt.tick_params(direction='in')
        plt.legend(fontsize=8, edgecolor='black', loc='best')
        figure_fig = plt.gcf()
        plt.rcParams['savefig.dpi'] = 800
        figure_fig.savefig('result/%s_value.png'%self.file_name, format='png')
        plt.show()
        return None

    def draw_re(self):
        '''绘制相对误差'''
        self.df['Qp'] = XAJ(np.append(self.bestx, self.area), self.xx0, self.df)
        re_value = np.abs(self.df['Qp'] - self.df['Qr']) / self.df['Qr']
        re_value.plot(color='green', label='re_value', lw=1.0)
        plt.title("Relative Value")
        plt.tick_params(direction='in')
        plt.legend(fontsize=8, edgecolor='black', loc='best')
        figure_fig = plt.gcf()
        plt.rcParams['savefig.dpi'] = 800
        figure_fig.savefig('result/%s_re.png'%self.file_name, format='png')
        plt.show()
        return None


def validation():
    '''用纳子峡数据验证结果'''
    xx0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    params = pd.read_csv('result/param.csv')
    idx = params['RMSE'].idxmin()
    opt_param = params.iloc[idx, 3:]
    df = pickle.load(open(r'F:\chenqing\meteData\Qr_4.pkl', 'rb'), encoding="iso-8859-1")
    df['Qp'] = XAJ(opt_param, xx0, df)
    rmse = math.sqrt(np.mean(df['inflow'] - df['Qp']) ** 2)
    return rmse


def main():
    file_name = 'Qr_5.pkl'
    df = pickle.load(open(r'F:\chenqing\meteData\%s' % file_name, 'rb'), encoding="iso-8859-1")
    xx = np.array([0.85, 20, 80, 0.18, 125, 0.2, 0.05, 15, 1.2, 0.5, 0.4, 0.995, 0.9, 0.5, 0])
    area = [38.6, 109.8, 126]
    xx0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    bl = np.array([0.1, 1, 1, 0.05, 1, 0.05, 0.005, 2, 0.1, 0.1, 0.1, 0.9, 0.5, 0.1, 0])
    bu = np.array([3, 30, 100, 50, 100, 2, 0.2, 80, 3, 0.6, 0.2, 0.999, 0.99, 0.9, 7])
    maxn = 10000
    kstop = 10
    pcento = 0.1
    peps = 0.001
    ngs = 2
    iseed = -1
    iniflg = 0
    sceua = SceUa(df, xx, area[2], xx0, bl, bu, maxn, kstop, pcento, peps, ngs, iseed, iniflg)
    icall, bestx, bestfx_list = sceua.sce_ua()
    xaj = XinAnRiver(file_name, df, area[2], xx0, icall, bestx, bestfx_list)
    # xaj.save_result()
    # xaj.draw_rmse()
    # xaj.draw_value()
    xaj.draw_re()
    return None


if __name__ == '__main__':
    main()
    # print(validation())
