#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25 Aug 2016
__author__ = 'czrzchao'

from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import pickle
from ga_xaj.ga import calobjValue,calfitValue,selection,crossover,mutation,best,geneEncoding
from ga_xaj.xaj import  XAJ
import pandas as pd
import os

mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False
print('math.sqrt(np.mean((yc - y0) ** 2)') # 目标函数



class GA:
    def __init__(self,file_name,pop_size,t_max,max_value,min_value,chrom_length,pc,pm,xx0):
        '''初始化参数
        Parameters
        ----------
        file_name:文件名
        pop_size: 种群数目
        max: 最大遗传代数
        max_value: 基因中允许出现的最大值
        max_value: 基因中允许出现的最小值
        n_value: 参数维度
        chrom_length: 染色体长度
        pc: 交配概率
        pm: 变异概率
        xx0: 初始值
        '''
        self.file_name=file_name
        self.pop_size=pop_size
        self.t_max=t_max
        self.max_value=max_value
        self.min_value=min_value
        self.n_value=len(self.max_value)
        self.chrom_length=chrom_length
        self.pc=pc
        self.pm=pm
        self.xx0=xx0

    # 计算2进制序列代表的数值
    def b2d(self, best_fit, best_individual):
        tn=[]
        tn.append(best_fit)
        for i in range(self.n_value):
            t = 0
            for j in range(self.chrom_length):
                t += best_individual[j + i * self.chrom_length] * (math.pow(2,j))
            t = t * (self.max_value[i] - self.min_value[i]) / (math.pow(2, self.chrom_length) - 1) + self.min_value[i]
            tn.append(t)
        return tn

    def calculate(self,df):
        # 存储每一代的最优解，N个二元组
        results = [[]]
        # pop = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for i in range(pop_size)]
        pop = geneEncoding(self.pop_size, self.chrom_length*self.n_value)
        for i in range(self.t_max):
            # 个体评价
            obj_value = calobjValue(pop, self.chrom_length, self.max_value,self.min_value,self.n_value,df,self.xx0)
            # 淘汰
            fit_value = calfitValue(obj_value)
            # 第一个存储最优的基因, 第二个存储最优解
            best_individual, best_fit = best(pop, fit_value)
            results.append(self.b2d(best_fit,best_individual))
            # 选择
            selection(pop, fit_value)
            # 交叉
            crossover(pop, self.pc)
            # 变异
            mutation(pop, self.pm)
        return results[1:]

    def draw_rmse(self, x, y):
        """绘制目标函数(RMSE)随遗传代数的变化情况曲线图"""
        plt.title("The optimal value")
        plt.plot(x, y, color='blue', label='GA', lw=1.0)
        plt.xlabel('iteration number')
        plt.ylabel('value')
        plt.xlim(xmin=0)
        # 设置坐标刻度朝里
        plt.tick_params(direction='in')
        plt.legend(fontsize=8, edgecolor='black', loc='best')
        figure_fig = plt.gcf()
        plt.rcParams['savefig.dpi'] = 800
        figure_fig.savefig('result/%s_rmse.png'%self.file_name, format='png')
        plt.show()
        return None

    def draw_value(self,df):
        """绘制预测值与真实值之间的对比图"""
        df['Qp'].plot(color='red',label='predict_value',lw=1.0)
        df['Qr'].plot(color='blue', label='real_value', lw=1.0)
        plt.title("Compare View")
        plt.tick_params(direction='in')
        plt.legend(fontsize=8, edgecolor='black', loc='best')
        figure_fig = plt.gcf()
        plt.rcParams['savefig.dpi'] = 800
        figure_fig.savefig('result/%s_value.png'%self.file_name, format='png')
        plt.show()
        return None

    def save_param(self,parameter_value,rmse_result):
        """存储每次运算结果的种群数，最大遗传次数,文件名，RMSE和率定的参数"""
        best_df = pd.DataFrame()
        best_df['pop_size'], best_df['t_max'] = [self.pop_size], [self.t_max]
        best_df['file_name']=[self.file_name]
        best_df['RMSE']=[rmse_result]
        best_df['K']=[parameter_value[0]]
        best_df['WUM']=[parameter_value[1]]
        best_df['WLM']=[parameter_value[2]]
        best_df['C']=[parameter_value[3]]
        best_df['WDM']=[parameter_value[4]]
        best_df['B']=[parameter_value[5]]
        best_df['IMP']=[parameter_value[6]]
        best_df['SM']=[parameter_value[7]]
        best_df['EX']=[parameter_value[8]]
        best_df['KG']=[parameter_value[9]]
        best_df['KSS']=[parameter_value[10]]
        best_df['KKG']=[parameter_value[11]]
        best_df['KKSS']=[parameter_value[12]]
        best_df['CS']=[parameter_value[13]]
        best_df['L']=[parameter_value[14]]
        best_df['U']=[parameter_value[15]]
        if os.path.exists('result/param.csv'):
           best_df.to_csv('result/param.csv', mode='a+', header=0, index=0)
        else:
            best_df.to_csv('result/param.csv', index=0)
        return None


def main():
    """参数含义说明
    Parameters
    ----------
    0.1-3，    K：蒸散发能力折算系数
    1-30，     WUM：为上层蓄水容量
    1-100，    WLM：下层蓄水容量
    0.05-50，  C：深层蒸散发系数
    1-100，    WDM：深层流域蓄水容量
    0.05-2，   B：蓄水容量曲线的方次
    0.005-0.2，IMP：不透水面积占全流域面积之比
    2-80，     SM：流域平均自由水蓄水容量
    0.1-3，    EX：自由水蓄水容量曲线指数
    0.1-0.6，  KG：自由水蓄水库对地下径流出流系数
    0.1-0.2，  KSS：自由水蓄水库对壤中流的出流系数
    0.9-0.999，KKG：地下水库的消退系数
    0.5-0.99， KKSS：壤中流水库的消退系数
    0.1-0.9，  CS：河网蓄水消退系数
    0-7,       L：滞时
    """
    # 设定参数
    pop_size = 50
    t_max=500
    min_value = [0.1, 1,  1, 0.05, 1, 0.05, 0.005, 2, 0.1, 0.1, 0.1,  0.9,  0.5,  0.1, 0]
    max_value = [3,  30, 100, 50, 100,  2,   0.2,  80, 3,  0.6, 0.2, 0.999, 0.99, 0.9, 7]
    chrom_length = 15
    pc = 0.8
    pm = 0.15
    xx0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    area = [38.6, 109.8, 126]
    file_name='Qr_5.pkl'
    # 生成rmse图片
    ga=GA(file_name,pop_size,t_max,max_value,min_value,chrom_length,pc,pm,xx0)
    df=pickle.load(open(r'F:\chenqing\meteData\%s' % file_name,'rb'),encoding="iso-8859-1")
    results = ga.calculate(df)
    rmse_list=[results[i][0] for i in range(len(results))]
    ga.draw_rmse(range(len(rmse_list)), rmse_list)
    # 生成预测值与真实值对比图
    results.sort(reverse=True)
    parameter_value=results[-1][1:]
    parameter_value.append(area[2])
    predict_value=XAJ(parameter_value,xx0,df)
    df['Qp'] = predict_value
    ga.draw_value(df)
    # 存储率定的参数
    print("rmse_result:",results[-1][0])
    ga.save_param(parameter_value,results[-1][0])


if __name__ == '__main__':
    main()
