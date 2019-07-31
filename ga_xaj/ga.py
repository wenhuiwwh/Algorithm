import math
import random
import numpy as np
from ga_xaj.xaj import XAJ


#目标函数
def caltarget(yc, y0):
    rmse = math.sqrt(np.mean((yc - y0) ** 2))
    return rmse

# 编码：随机生成序列 
def geneEncoding(pop_size, chrom_length):
    """
    :param pop_size: 种群数量
    :param chrom_length: 染色体长度
    :return: 种群值
    """
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)

    return pop[1:]


# 解码并计算值
def decodechrom(pop, chrom_length,n_value):
    temp = []
    for i in range(len(pop)):
        temp0 = []
        for k in range(n_value):
            t = 0
            for j in range(chrom_length):
                t += pop[i][j+k*chrom_length] * (math.pow(2, j))
            temp0.append(t)
        temp.append(temp0)
    return temp


def calobjValue(pop, chrom_length, max_value,min_value,n_value,df,xx0):
    obj_value = []

    temp1 = decodechrom(pop, chrom_length,n_value)
    for i in range(len(temp1)):
        xx = []
        for k in range(n_value):
            x0=temp1[i][k]*(max_value[k]-min_value[k])/(math.pow(2,chrom_length)-1)+min_value[k]
            xx.append(x0)
        xx.append(126)
        y=XAJ(xx,xx0,df)
        df['Qp']=y
        yt=caltarget(df['Qp'],df['Qr'])
        obj_value.append(yt)
    return obj_value


def calfitValue(obj_value):
    """淘汰（去除负值）"""
    fit_value = []
    c_max=200
    for i in range(len(obj_value)):
        if (c_max-obj_value[i] > 0):
            temp = c_max - obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


def best(pop, fit_value):
    """找出最优解的基因编码和最优解"""
    px = len(pop)
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, px):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# 选择
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total

def cumsum(fit_value):
    for i in range(len(fit_value) - 2, -1, -1):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j += 1
        fit_value[i] = t
        fit_value[len(fit_value) - 1] = 1

def selection(pop, fit_value):
    newfit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # 计算累计概率
    cumsum(newfit_value)
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    # 转轮盘选择法
    while newin < pop_len:
        if (ms[newin] < newfit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin += 1
        else:
            fitin += 1
    pop = newpop
    return pop


def crossover(pop, pc):
    """交配"""
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2
    return pop


def mutation(pop, pm):
    """基因突变"""
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
    return pop


if __name__ == '__main__':
    pass
