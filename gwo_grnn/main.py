#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/21 15:35
__author__ = 'wangwenhui'

from gwo_grnn.gwo import gwo
from gwo_grnn.grnn import test_model

class Main:
    def __init__(self,SearchAgents_no,Max_iteration,dim,lb,ub):
        self.SearchAgents_no=SearchAgents_no
        self.Max_iteration=Max_iteration
        self.dim=dim
        self.lb=lb
        self.ub=ub

def main():
    """设置gwo的参数"""
    # 种群数
    SearhAgents_no=100
    # 最大迭代次数
    Max_iteration=200
    # 需要优化的光滑因子参数维度
    dim=1
    # 参数取值下界
    lb=0.1
    # 参数取值上界
    ub=3
    # 调用灰狼优化算法函数
    best_g,min_value=gwo(dim,lb,ub,SearhAgents_no,Max_iteration)
    print("-------------结果显示------------------")
    print("the best parameter is: "+str(best_g))
    print("the min value is: "+str(min_value[-1]))
    predicted=test_model((best_g))
    print("predicted for backfill pipeline risk: %s"%predicted)
    return best_g


if __name__ == '__main__':
    main()