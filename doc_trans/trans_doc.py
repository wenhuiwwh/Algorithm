#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/8/10 15:58
__author__ = 'wangwenhui'

import os
import struct
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time


def awx_to_txt():
    """将awx雷达数据文件转换成txt数据文件"""
    read_path = 'doc_trans/data/201709060700_B03.awx'
    write_path = 'doc_trans/result/copy_awx.txt'

    with open(read_path, 'rb') as f:
        byte = f.read()
        # struct.unpack返回的是一个元祖
        data_ncols, = struct.unpack('h', byte[20:22])
        info, = struct.unpack('h', byte[22:24])
        data_nrows, = struct.unpack('h', byte[24:26])
        lat_end = struct.unpack('h', byte[72:74])[0] / 100
        lat_start = struct.unpack('h', byte[74:76])[0] / 100
        lon_start = struct.unpack('h', byte[76:78])[0] / 100
        lon_end = struct.unpack('h', byte[78:80])[0] / 100
        lon_interval = (lat_end - lat_start) / data_nrows
        # 经纬度间隔相差一点点
        lat_interval = (lon_end - lon_start) / data_ncols
        content = struct.unpack('%dB' % (len(byte) - info * data_ncols), byte[info * data_ncols:])
        content = pd.DataFrame(np.array(content).reshape(data_nrows, data_ncols))
        index = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']
        data = [data_ncols, data_nrows, lon_start, lat_start, lon_interval, -9999]
        summary = dict(zip(index, data))
        f.close()

    with open(write_path, 'w') as f1:
        for key in summary:
            f1.writelines(str(key) + " " + str(summary[key]) + '\n')
        content.to_csv(write_path, header=0, index=0, sep=' ', mode='a')
    print("将awx格式转换为nc格式文件成功！")
    return None


def awx_to_nc():
    """将awx雷达数据文件转换为nc数据文件"""
    read_path = 'doc_trans/data/201709060700_B03.awx'
    write_path = 'doc_trans/result/copy_awx.nc'

    with open(read_path, 'rb') as f:
        byte = f.read()
        data_ncols, = struct.unpack('h', byte[20:22])
        info, = struct.unpack('h', byte[22:24])
        data_nrows, = struct.unpack('h', byte[24:26])
        lat_end = struct.unpack('h', byte[72:74])[0] / 100
        lat_start = struct.unpack('h', byte[74:76])[0] / 100
        lon_start = struct.unpack('h', byte[76:78])[0] / 100
        lon_end = struct.unpack('h', byte[78:80])[0] / 100
        content = struct.unpack('%dB' % (len(byte) - info * data_ncols), byte[info * data_ncols:])
        attr_data = np.array(content).reshape(data_nrows, data_ncols)
        lon = np.linspace(lon_start, lon_end, data_ncols)
        lat = np.linspace(lat_start, lat_end, data_nrows)

    ds = xr.Dataset()
    ds.coords['lon'] = lon
    ds.coords['lat'] = lat
    ds['var'] = (('lat', 'lon'), attr_data)
    ds.to_netcdf(write_path, format='NETCDF3_64BIT')
    print("将awx格式转换为nc格式文件成功！")
    return None


def zeros_to_txt():
    """将000格式文件转为txt格式文件"""
    read_path = 'doc_trans/data/ddd.000'
    write_path = 'doc_trans/result/copy_000.txt'

    zeros_data = pd.read_csv(read_path, header=None, skiprows=1, nrows=1, encoding='gbk', sep=" ")
    data_ncols = int(zeros_data.loc[0, 12])
    data_nrows = int(zeros_data.loc[0, 13])
    lon_start = float(zeros_data.loc[0, 8])
    lat_start = float(zeros_data.loc[0, 10])
    cell_size = float(zeros_data.loc[0, 6])
    data = [data_ncols, data_nrows, lon_start, lat_start, cell_size, -9999]
    index = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']
    summary = dict(zip(index, data))
    with open(write_path, 'w') as f:
        for key in summary:
            f.writelines(key + " " + str(summary[key]) + '\n')
    content = pd.read_csv(read_path, header=None, skiprows=2, sep=' ', dtype=float)
    # 逐行排序
    content.sort_index(axis=0, ascending=False, inplace=True)
    content.to_csv(write_path, index=0, header=0, sep=' ', mode='a')
    print('000格式转为txt格式文件成功！')
    return None


def zeros_to_dat():
    """将000格式文件转为dat格式文件"""
    read_path = 'doc_trans/data/ddd.000'
    write_path = 'doc_trans/result/copy_000.dat'
    zeros_data = pd.read_csv(read_path, nrows=1, encoding='gbk', sep='\t')
    copy_data = pd.read_csv(read_path, header=None, skiprows=1, nrows=1, encoding='gbk', sep=' ')
    header = "".join(zeros_data.columns.values)
    title = pd.Series(header)
    time = pd.DataFrame(np.array(copy_data.iloc[0, 0:4]).reshape(1, 4))
    three_row = [str('{:g}'.format(i)) for i in list(copy_data.iloc[0, 6:12])]
    three_row.insert(0, '3')
    three_row.insert(1, '0')
    four_row = [str('{:g}'.format(i)) for i in list(copy_data.iloc[0, 12:14])]
    title.to_csv(write_path, index=False, mode='w')
    time.to_csv(write_path, index=0, header=0, float_format='%.0f', sep=' ', mode='a')
    with open(write_path, 'a') as f:
        for i in (three_row, four_row):
            f.writelines(' '.join(i) + '\n')
    content = pd.read_csv(read_path, header=None, skiprows=2, sep=' ', dtype=float)
    content.to_csv(write_path, index=0, header=0, sep=' ', mode='a')
    print("000格式文件转为dat格式文件成功！")
    return None


def dat_to_zeros():
    """将dat格式文件转为000格式文件"""
    read_path = 'doc_trans/data/16102000.003.ZJOCF.Pr03.dat'
    write_path = 'doc_trans/result/copy_dat.000'
    content_data = pd.read_csv(read_path, header=None, skiprows=4, encoding='gbk', sep='\s+')
    info_list = []
    with open(read_path, 'r') as f:
        for i in range(4):
            info_list.append(f.readline().strip().split(" "))
    info_list[1].extend(['0', '999'])
    summary = [info_list[0], info_list[1] + info_list[2][2:] + info_list[3][:2]]
    summary[1].extend(['0.6', '0.0', '6.2', '1.0', '0.0'])
    with open(write_path, 'w') as f:
        for i in summary:
            f.writelines(" ".join(i) + '\n')
    content_data.to_csv(write_path, header=0, index=0, sep=' ', mode='a')
    print("dat格式转为000格式文件成功！")
    return None


def dat_to_txt():
    """将dat格式文件转为txt格式文件"""
    read_path = 'doc_trans/data/16102000.003.ZJOCF.Pr03.dat'
    write_path = 'doc_trans/result/copy_dat.txt'
    content_data = pd.read_csv(read_path, header=None, skiprows=4, sep='\s+')
    content_data.sort_index(axis=0, ascending=False, inplace=True)
    info_list = []
    with open(read_path, 'r') as f:
        next(f)
        next(f)
        for i in range(2):
            info_list.append(f.readline().strip().split(" "))
    index = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']
    summary = [info_list[1][0], info_list[1][1], info_list[0][4], info_list[0][6], info_list[0][2], -9999]
    dicts = dict(zip(index, summary))
    with open(write_path, 'w') as f:
        for key in dicts:
            f.writelines(key + " " + str(dicts[key]) + '\n')
    content_data.to_csv(write_path, index=0, header=0, mode='a', sep=' ')
    print("将dat格式文件转为txt格式文件成功")
    return None


def txt_to_dat():
    """将txt格式文件转为dat格式文件"""
    read_path = 'doc_trans/data/depth10.txt'
    write_path = 'doc_trans/result/copy_txt.dat'
    header_data = pd.read_csv(read_path, header=None, index_col=0, nrows=6, sep='\s+')
    content = pd.read_csv(read_path, header=None, skiprows=6, sep='\s+')
    content.sort_index(axis=0, ascending=False, inplace=True)
    summary = [str(i) for i in np.array(header_data.iloc[:, 0])]
    lon_end = str(float(summary[2]) + float(summary[0]) * float(summary[4]))
    lat_end = str(float(summary[3]) + float(summary[1]) * float(summary[4]))
    one_data = ['diamond 4']
    date = str(time.strftime('%Y %m %d %H', time.localtime(time.time()))).split()
    three_data = ['3'] + ['0'] + summary[4].split() * 2 + summary[2].split() + lon_end.split() + summary[
        3].split() + lat_end.split()
    four_data = [str(int(float(summary[0]))), str(int(float(summary[1])))]
    four_data.extend(['5', '5', '300', '1', '0'])
    summary = [one_data, date, three_data, four_data]
    with open(write_path, 'w') as f:
        for i in summary:
            f.writelines(" ".join(i) + "\n")
    content.to_csv(write_path, index=0, header=0, sep=' ', mode='a')
    print("将txt格式文件转为dat格式文件成功！")
    return None


def txt_to_zeros():
    """将txt格式文件转为000格式文件"""
    read_path = 'doc_trans/data/depth10.txt'
    write_path = 'doc_trans/result/copy_txt.000'
    header_data = pd.read_csv(read_path, header=None, index_col=0, nrows=6, sep='\s+')
    content = pd.read_csv(read_path, header=None, skiprows=6, sep='\s+')
    content.sort_index(axis=0, ascending=False, inplace=True)
    summary = [str(i) for i in np.array(header_data.iloc[:, 0])]
    lon_end = str(float(summary[2]) + float(summary[0]) * float(summary[4]))
    lat_end = str(float(summary[3]) + float(summary[1]) * float(summary[4]))
    one_data = ['diamond 4']
    date = str(time.strftime('%Y %m %d %H', time.localtime(time.time()))).split()
    two_data = date + str(int(float(summary[5]))).split() + summary[4].split() * 2 + summary[
        2].split() + lon_end.split() + summary[3].split() + lat_end.split() + str(int(float(summary[0]))).split() + str(
        int(float(summary[1]))).split()
    two_data.extend(['0.6', '0.0', '6.2', '1.0', '0.0'])
    with open(write_path,'w') as f:
        for i in (one_data,two_data):
            f.writelines(" ".join(i)+"\n")
    content.to_csv(write_path,index=0,header=0,sep=' ',mode='a')
    print("将txt格式文件转为000格式文件成功！")
    return None


if __name__ == '__main__':
    # awx_to_txt()
    # awx_to_nc()
    # zeros_to_txt()
    # zeros_to_dat()
    # dat_to_zeros()
    # dat_to_txt()
    # txt_to_dat()
    txt_to_zeros()
