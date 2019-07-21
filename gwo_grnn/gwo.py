#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/21 8:31
# __author__ = 'wangwenhui'

import numpy as np
import numpy.random as rd
from gwo_grnn.grnn import train_model
import math



def fitness(g):
    '''适应度函数选为均方根误差'''
    y_predicted, y_test=train_model(g)
    y_test=np.array(y_test)
    rmse=math.sqrt(np.mean(y_predicted-y_test)**2)
    return rmse


def gwo(dim, lb, ub, SearchAgents_no, Max_iteration):
    # 初始化灰狼的位置
    Alpha_pos = np.zeros(dim)
    Beta_pos = np.zeros(dim)
    Delta_pos = np.zeros(dim)
    # 初始化Alpha狼的目标函数值
    Alpha_score = float("inf")
    Beta_score = float("inf")
    Delta_score = float("inf")
    # 初始化首次搜索位置
    Positions = np.dot(rd.rand(SearchAgents_no, dim), (ub - lb)) + lb
    iterations = []
    rmse = []

    # 主循环
    index_iteration = 0
    while index_iteration < Max_iteration:
        # 遍历每个狼
        for i in range(Positions.shape[0]):
            # 若搜索位置超过了搜索空间，需要重新回到搜索空间
            for j in range(Positions.shape[1]):
                Flag4ub = Positions[i, j] > ub
                Flag4lb = Positions[i, j] < lb
                # 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，则回到最大值边界
                if Flag4ub:
                    Positions[i, j] = ub
                if Flag4lb:
                    Positions[i, j] = lb

            # 如果目标函数值小于Alpha狼的目标函数值
            if fitness(Positions[i]) < Alpha_score:
                # 则将Alpha狼的目标函数值更新为最优目标函数值
                Alpha_score = fitness(Positions[i])
                # 同时将Alpha狼的位置更新为最优位置
                Alpha_pos = Positions[i]
            # 如果目标函数值介于Alpha狼和Beta狼的目标函数值之间
            if Alpha_score < fitness(Positions[i]) < Beta_score:
                # 则将Beta狼的目标函数值更新为最优目标函数值
                Beta_score = fitness(Positions[i])
                Beta_pos = Positions[i]
            # 如果目标函数值介于Beta狼和Delta狼的目标函数值之间
            if fitness(Positions[i]) > Alpha_score and Beta_score < fitness(Positions[i]) < Delta_score:
                # 则将Delta狼的目标函数值更新为最优目标函数值
                Delta_score = fitness(Positions[i])
                Delta_pos = Positions[i]

        # 对每一次迭代，计算相应的a值
        a = 2 - index_iteration * (2 / Max_iteration)

        # 遍历每个狼
        for i in range(Positions.shape[0]):
            # 遍历每个维度
            for j in range(Positions.shape[1]):
                # 包围猎物，位置更新
                r1 = rd.random(1)
                r2 = rd.random(1)
                # 计算系数A
                A1 = 2 * a * r1 - a
                # 计算系数C
                C1 = 2 * r2

                # Alpha狼位置更新
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = rd.random(1)
                r2 = rd.random(1)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                # Beta狼位置更新
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                # Delta狼位置更新
                r1 = rd.random(1)
                r2 = rd.random(1)

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                # Delta狼位置更新
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                # 位置更新
                Positions[i, j] = (X1 + X2 + X3) / 3

        index_iteration = index_iteration + 1
        iterations.append(index_iteration)
        rmse.append(fitness(Alpha_pos))
        # print("------------------------迭代次数-----------------------------"+str(index_iteration))
        # print(Positions)
        # print("the best g for fitness: "+str(Alpha_pos))
        # print("rmse: "+str(fitness(Alpha_pos)))

    best_g=Alpha_pos

    return best_g,rmse

