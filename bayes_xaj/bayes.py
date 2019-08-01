#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 16:25
__author__ = 'wangwenhui'

import pickle
import numpy as np
import pandas as pd
import math
import os
from ga_xaj.xaj import XAJ
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp, partial



class Bayes:
    def __init__(self, file_name, xx0, area, init_points=5, n_iter=25):
        """初始化参数
        Parameters
        ----------
        file_name: 文件名
        xx0: 模型初始值
        area: xx的第16个参数
        init_points: 随机初始点的个数
        n_iter: 最大迭代次数
       """
        self.file_name = file_name
        self.xx0 = xx0
        self.area = area
        self.init_points = init_points
        self.n_iter = n_iter

    def read_pkl(self):
        """读取文件"""
        df = pickle.load(open('/data/%s'%self.file_name, 'rb'), encoding="iso-8859-1")
        return df

    def bayes(self, K, WUM, WLM, C, WDM, B, IMP, SM, EX, KG, KSS, KKG, KKSS, CS, L):
        """bayes_opt模块方法"""
        df = self.read_pkl()
        param_list = [K, WUM, WLM, C, WDM, B, IMP, SM, EX, KG, KSS, KKG, KKSS, CS, L]
        param_list.append(126)
        df['Qp'] = XAJ(param_list, self.xx0, df)
        value = -math.sqrt(np.mean((df['Qp'] - df['Qr']) ** 2))
        return value

    def run_bayes(self):
        """运行bayes方法"""
        param_list = {'K': (0.1, 3), 'WUM': (1, 30), 'WLM': (1, 100), 'C': (0.05, 50), 'WDM': (1, 100), 'B': (0.05, 2),
                      'IMP': (0.005, 0.2), 'SM': (2, 80), 'EX': (0.1, 3), 'KG': (0.1, 0.6), 'KSS': (0.1, 0.2),
                      'KKG': (0.9, 0.999), 'KKSS': (0.5, 0.99), 'CS': (0.1, 0.9), 'L': (0, 7)}
        xaj_bayes = BayesianOptimization(self.bayes, param_list)
        xaj_bayes.maximize(self.init_points,self.n_iter)
        opt_target=xaj_bayes.max['target']
        opt_params=xaj_bayes.max['params']
        print(xaj_bayes.max)
        return opt_target,opt_params

    def save_params(self):
        """存储参数"""
        opt_target,opt_params=self.run_bayes()
        best_df=pd.DataFrame()
        best_df['file_name']=[self.file_name]
        best_df['init_points']=[self.init_points]
        best_df['n_iter']=[self.n_iter]
        best_df['target']=[-opt_target]
        best_df=pd.concat([best_df,pd.DataFrame(opt_params,index=[0])],axis=1)
        best_df['area']=self.area
        if os.path.exists("result/params.csv"):
            best_df.to_csv('result/params.csv',mode='a+',header=0,index=0)
        else:
            best_df.to_csv('result/params.csv',index=0)
        return None

    def percept(self, args):
        """hyperopt模块方法，定义目标函数，此法暂时不通"""
        df = self.read_pkl()
        param_list = [args["K"], args["WUM"], args["WLM"], args["C"], args["WDM"], args["B"], args["IMP"], args["SM"],
                      args["EX"], args["KG"], args["KSS"], args["KKG"], args["KKSS"], args["CS"], args["L"],38.6]
        df['Qp'] = XAJ(param_list, self.xx0, df)
        rmse_value = -math.sqrt(np.mean((df['Qp'] - df['Qr']) ** 2))
        return rmse_value

    def run_percept(self):
        """定义域空间
        choice: 类别变量
        quniform: 离散均匀(整数间隔均匀)
        uniform: 连续均匀(对数下均匀分布)
        tpe: 优化算法
        partial: 指定搜索算法tpe的参数
        """
        space = {"K": hp.uniform("K", 0.1, 3), "WUM": hp.uniform("WUM", 1, 30), "WLM": hp.uniform("WLM", 1, 100),
                 "C": hp.uniform("C", 0.05, 50), "WDM": hp.uniform("WDM", 1, 100), "B": hp.uniform("B", 0.05, 2),
                 "IMP": hp.uniform("IMP", 0.005, 0.2), "SM": hp.uniform("SM", 2, 80), "EX": hp.uniform("EX", 0.1, 3),
                 "KG": hp.uniform("KG", 0.1, 0.6), "KSS": hp.uniform("KSS", 0.1, 0.2),
                 "KKSS": hp.uniform("KKSS", 0.5, 0.99),
                 "CS": hp.uniform("CS", 0.1, 0.9), "L": hp.uniform("L", 0, 7)}
        bayesopt=partial(tpe.suggest,n_startup_jobs=10)
        best = fmin(self.percept, space, bayesopt, max_evals=100)
        print(best)
        print(self.percept(best))
        return best


def main():
    file_name = "Qr_2.pkl"
    xx0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    area = [38.6, 109.8, 126]
    init_points =45
    n_iter = 100
    baye = Bayes(file_name, xx0, area[0],init_points,n_iter)
    baye.save_params()
    # baye.run_percept()
    return None


if __name__ == '__main__':
    main()
