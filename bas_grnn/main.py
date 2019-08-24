#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 10:17
__author__ = 'wangwenhui'

import pandas as pd
from bas_grnn.grnn import test_grnn
from bas_grnn.bas import bas



# def test_main():
#     ub = 2
#     lb = -2
#     xbest, fbest = bas(ub, lb, k=2)
#     print("xbest: %s" % xbest)
#     print("fbest: %s" % fbest)
#     return None


def grnn_main():
    ub=2
    lb=0
    submit_path="bas_grnn/bikedata/sample_submit.csv"
    submit_data=pd.read_csv(submit_path,encoding='utf-8')
    best_g,rmse=bas(ub,lb,k=1)
    y_predicted=test_grnn(best_g)
    # 输出预测结果至my_XGB_prediction.csv
    submit_data['y'] = y_predicted
    submit_data.to_csv('my_XGB_prediction.csv', index=False)
    print("预测结果生成成功！")
    return None



if __name__ == '__main__':
    grnn_main()