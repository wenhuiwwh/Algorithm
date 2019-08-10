#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/8/4 15:26
__author__ = 'wangwenhui'
import numpy as np
import pandas as pd
import time
import datetime as dt
import xarray as xr



def zeros_to_nc():
    """读取000格式文件，并转换为nc格式文件"""
    read_path="doc_trans/data/ddd.000"
    write_path="doc_trans/result/copy_000.nc"
    zeros_data=pd.read_csv(read_path,header=None,skiprows=1,nrows=1,encoding='gbk',sep=' ')
    real_data=pd.read_csv(read_path,header=None,skiprows=2,sep=' ')
    real_data=np.array(real_data)

    year=int(zeros_data.loc[0,0])
    month=int(zeros_data.loc[0,1])
    day=int(zeros_data.loc[0,2])
    minute=int(zeros_data.loc[0,3])
    second=int(zeros_data.loc[0,4])
    log_interval=float(zeros_data.loc[0,6])
    lat_interval=float(zeros_data.loc[0,7])
    log_start=float(zeros_data.loc[0,8])
    log_end=float(zeros_data.loc[0,9])
    lat_start=float(zeros_data.loc[0,10])
    lat_end=float(zeros_data.loc[0,11])
    data_cols=int(zeros_data.loc[0,12])
    data_rows=int(zeros_data.loc[0,13])
    date=dt.datetime(year,month,day,minute,second)
    log=np.arange(log_start,log_end+log_interval,log_interval)
    lat=np.arange(lat_start,lat_end+lat_interval,lat_interval)

    ds=xr.Dataset()
    ds.coords['lon']=log
    ds['lon'].attrs['units']='degrees_east'
    ds['lon'].attrs['long_time']='Longitude'
    ds.coords['lat']=lat
    ds['lat'].attrs['units']='degrees_north'
    ds.coords['time']=np.array([date])
    ds['time'].attrs['long_time']='Time(UTC)'
    ds['var']=(('time','lat','lon'),real_data.reshape((1,data_rows,data_cols)))
    ds.to_netcdf(write_path,format='NETCDF3_64BIT')
    print("将000格式文件转为nc格式文件成功！")
    return ds


def txt_to_nc():
    """将txt格式文件转化为nc格式文件"""
    date=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    read_path='doc_trans/data/depth10.txt'
    write_path='doc_trans/result/copy_txt.nc'
    txt_data=pd.read_csv(read_path,header=None,index_col=0,nrows=6,sep='\s+')
    real_data=pd.read_csv(read_path,header=None,skiprows=6,sep='\s+')
    real_data.sort_index(axis=0,ascending=False,inplace=True)
    real_data=np.array(real_data)
    info=[str(i) for i in np.array(txt_data.iloc[:,0])]
    data_cols=int(float(info[0]))
    data_rows=int(float(info[1]))
    log_end=float(info[2])+float(info[0])*float(info[4])
    lat_end=float(info[3])+float(info[1])*float(info[4])
    lon=np.linspace(float(info[2]),log_end,data_cols)
    lat=np.arange(float(info[3]),lat_end,float(info[4]))

    ds=xr.Dataset()
    ds.coords['lon']=lon
    ds.coords['lat']=lat
    ds.coords['time']=np.array([date])
    ds['time'].attrs['long_time']='Time(UTC)'
    ds['var']=(('time','lat','lon'),real_data.reshape((1,data_rows,data_cols)))
    ds.to_netcdf(write_path,format='NETCDF3_64BIT')
    print("将txt格式文件转化为nc格式文件成功！")
    return None


def dat_to_nc():
    """将dat格式文件转化为nc格式文件"""
    read_path="doc_trans/data/16102000.003.ZJOCF.Pr03.dat"
    write_path="doc_trans/result/copy_dat.nc"
    two_data=pd.read_csv(read_path,header=None,skiprows=1,nrows=1,sep='\s+')
    three_data=pd.read_csv(read_path,header=None,skiprows=2,nrows=1,sep='\s+')
    four_data=pd.read_csv(read_path,header=None,skiprows=3,nrows=1,sep='\s+')
    real_data=pd.read_csv(read_path,header=None,skiprows=4,sep='\s+')
    # 将多维数据展开为1维数据
    two_data=np.array(two_data).flatten()
    three_data=np.array(three_data).flatten()
    four_data=np.array(four_data).flatten()
    real_data=np.array(real_data)
    list_time=[]
    for i in two_data:
        list_time.append(str(i))
    date=dt.datetime.strptime(" ".join(list_time),"%Y %m %d %H")
    data_cols=int(four_data[0])
    data_rows=int(four_data[1])
    lon=np.linspace(three_data[4],three_data[5],data_cols)
    lat=np.linspace(three_data[6],three_data[7],data_rows)

    ds=xr.Dataset()
    ds.coords['lon']=lon
    ds.coords['lat']=lat
    ds.coords['time']=np.array([str(date)])
    ds['time'].attrs['long_time']='Time(UTC)'
    ds['var']=(('time','lat','lon'),real_data.reshape((1,data_rows,data_cols)))
    ds.to_netcdf(write_path,format='NETCDF3_64BIT')
    print("将dat格式转换为nc格式文件成功！")
    return None




if __name__ == '__main__':
    # zeros_to_nc()
    # txt_to_nc()
    dat_to_nc()