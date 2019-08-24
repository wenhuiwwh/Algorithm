#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 10:17
__author__ = 'wangwenhui'

import numpy as np
import math



def Michalewicz(x, m=10):
    """
    describe url:https://www.sfu.ca/~ssurjano/michal.html
    The function is usually evaluated on the hypercube xi ∈ [0, π], for all i = 1, …, d
    """
    d = len(x)
    f = 0
    for i in range(0, d):
        value = math.sin(x[i]) * (math.sin((i + 1) * x[i] ** 2 / math.pi)) ** (2 * m)
        f += value
    return -f


def Goldsteiin_Price(x):
    """
    describe url:http://www.sfu.ca/~ssurjano/goldpr.html
    The function is usually evaluated on the square xi ∈ [-2, 2], for all i = 1, 2.
    """
    f1 = (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))
    f2 = (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    return f1 * f2


def main():
    michalewicz_x = [2.20, 1.57]
    mic_value=Michalewicz(michalewicz_x)
    goldstein_x = [0.001870855, -0.996496153]
    gold_value = Goldsteiin_Price(goldstein_x)
    print("mic_best_value: %s"%mic_value)  # best_value: -1.80114
    print("golg_best_value: %s"%gold_value) # best_value: 3.004756




if __name__ == '__main__':
    main()