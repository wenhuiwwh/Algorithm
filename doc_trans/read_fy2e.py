import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import xarray as xr
from PIL import Image


def trans_gray(bt, band):
    '''将亮度或反照率转化为灰度值的函数'''
    if band == 1:
        return int(255 * (bt - 315.0) / (187.0 - 315.0))
    elif band == 2:
        return int(255 * (bt - 315.0) / (187.0 - 315.0))
    elif band == 3:
        return int(255 * (bt - 315.0) / (188.0 - 315.0))
    elif band == 4:
        return int(255 * (bt - 318.0) / (233.0 - 318.0))
    else:
        return int(255 * (bt - 0) / (1.00 - 0))


def read_fy2e(band):
    '''读取fy2e格式文件并转为png图片格式'''
    read_path = 'doc_trans/data/FY2E_2017_06_01_00_17_E_PJ3.gpf'
    write_path = 'doc_trans/result/copy_fy2e%d.png' % (band)
    # Read data header parameters
    with open(read_path, 'rb') as f:
        fileid, = struct.unpack('2s', f.read(2))
        version, = struct.unpack('<h', f.read(2))
        satid, = struct.unpack('<h', f.read(2))
        year, = struct.unpack('<h', f.read(2))
        month, = struct.unpack('<h', f.read(2))
        day, = struct.unpack('<h', f.read(2))
        hour, = struct.unpack('<h', f.read(2))
        minute, = struct.unpack('<h', f.read(2))
        chnums, = struct.unpack('<h', f.read(2))  # 通道数
        pjtype, = struct.unpack('<h', f.read(2))
        width, = struct.unpack('<h', f.read(2))  # 投影头文件的数据宽度
        height, = struct.unpack('<h', f.read(2))  # 投影头文件的数据高度
        clonres, = struct.unpack('<f', f.read(4))
        clatres, = struct.unpack('<f', f.read(4))
        stdlat1, = struct.unpack('<f', f.read(4))
        stdlat2, = struct.unpack('<f', f.read(4))
        earthr, = struct.unpack('<f', f.read(4))
        minlat, = struct.unpack('<f', f.read(4))
        maxlat, = struct.unpack('<f', f.read(4))
        minlon, = struct.unpack('<f', f.read(4))
        maxlon, = struct.unpack('<f', f.read(4))
        ltlat, = struct.unpack('<f', f.read(4))
        ltlon, = struct.unpack('<f', f.read(4))
        rtlat, = struct.unpack('<f', f.read(4))
        rtlon, = struct.unpack('<f', f.read(4))
        lblat, = struct.unpack('<f', f.read(4))
        lblon, = struct.unpack('<f', f.read(4))
        rblat, = struct.unpack('<f', f.read(4))
        rblon, = struct.unpack('<f', f.read(4))
        stdlon, = struct.unpack('<f', f.read(4))
        centerlon, = struct.unpack('<f', f.read(4))
        centerlat, = struct.unpack('<f', f.read(4))
        chindex = []  # 通道索引
        for i in range(chnums):
            chindex.append(struct.unpack('b', f.read(1))[0])
        f.read(128 - chnums)
        plonres, = struct.unpack('<f', f.read(4))
        platres, = struct.unpack('<f', f.read(4))
        f.read(1808)
        infrared1 = struct.unpack('%dl' % (1024), f.read(1024 * 4))
        infrared2 = struct.unpack('%dl' % (1024), f.read(1024 *4))
        infrared3 = struct.unpack('%dl' % (1024), f.read(1024 * 4))
        infrared4 = struct.unpack('%dl' % (1024), f.read(1024 * 4))
        infrared = (infrared1, infrared2, infrared3, infrared4)
        light = struct.unpack('%dl' % (1024), f.read(1024*4))
        f.close()

    nskip = 2048 + 32768
    for i in range(1,band):  # 计算跳过的投影数据头+定标值+通道数据n所占的字符数
        nskip += 2 * width * height

    with open(read_path, 'rb') as fn:
        fn.read(nskip)
        value = list()
        if band < 5:  # 读的是红1,红2，红3，红4
            # content = struct.unpack('%dH' % (height * width), byte[nskip:nskip+2*height*width])
            content = struct.unpack('%dH' % (height * width), fn.read(2 * height * width))
            for i in content:
                value.append(infrared[band-1][i] / 1000)
        else:  # 读的是可见光
            content = struct.unpack('%dB' % (height * width), fn.read(height * width))
            for i in content:
                value.append(light[i] / 100000000)
        f.close()

    grayvalue = []
    for i in value:
        grayvalue.append(trans_gray(i, band))
    gray_matrix = np.array(grayvalue).reshape(height, width)
    d1,d2=gray_matrix.shape
    gray_matrix=gray_matrix.reshape(d1,d2)
    fy2eImage = Image.fromarray(gray_matrix)
    fy2eImage.save(write_path)
    return '将fy2e格式文件转为png图片格式成功！'



def fy2e_to_nc(band):
    '''将fy2e格式文件转为nc格式文件'''
    read_path = 'meteorological_data/FY2E_2017_06_01_00_17_E_PJ3.gpf'
    write_path = 'meteorological_data/copy_fy2e%d.nc' % (band)
    # Read data header parameters
    with open(read_path, 'rb') as f:
        fileid, = struct.unpack('2s', f.read(2))
        version, = struct.unpack('<h', f.read(2))
        satid, = struct.unpack('<h', f.read(2))
        year, = struct.unpack('<h', f.read(2))
        month, = struct.unpack('<h', f.read(2))
        day, = struct.unpack('<h', f.read(2))
        hour, = struct.unpack('<h', f.read(2))
        minute, = struct.unpack('<h', f.read(2))
        chnums, = struct.unpack('<h', f.read(2))  # 通道数
        pjtype, = struct.unpack('<h', f.read(2))
        width, = struct.unpack('<h', f.read(2))  # 投影头文件的数据宽度
        height, = struct.unpack('<h', f.read(2))  # 投影头文件的数据高度
        clonres, = struct.unpack('<f', f.read(4))
        clatres, = struct.unpack('<f', f.read(4))
        stdlat1, = struct.unpack('<f', f.read(4))
        stdlat2, = struct.unpack('<f', f.read(4))
        earthr, = struct.unpack('<f', f.read(4))
        minlat, = struct.unpack('<f', f.read(4))
        maxlat, = struct.unpack('<f', f.read(4))
        minlon, = struct.unpack('<f', f.read(4))
        maxlon, = struct.unpack('<f', f.read(4))
        ltlat, = struct.unpack('<f', f.read(4))
        ltlon, = struct.unpack('<f', f.read(4))
        rtlat, = struct.unpack('<f', f.read(4))
        rtlon, = struct.unpack('<f', f.read(4))
        lblat, = struct.unpack('<f', f.read(4))
        lblon, = struct.unpack('<f', f.read(4))
        rblat, = struct.unpack('<f', f.read(4))
        rblon, = struct.unpack('<f', f.read(4))
        stdlon, = struct.unpack('<f', f.read(4))
        centerlon, = struct.unpack('<f', f.read(4))
        centerlat, = struct.unpack('<f', f.read(4))
        chindex = []  # 通道索引
        for i in range(chnums):
            chindex.append(struct.unpack('b', f.read(1))[0])
        f.read(128 - chnums)
        plonres, = struct.unpack('<f', f.read(4))
        platres, = struct.unpack('<f', f.read(4))
        f.read(1808)
        infrared1 = struct.unpack('%dl' % (1024), f.read(1024 * 4))  # 红外1
        infrared2 = struct.unpack('%dl' % (1024), f.read(1024 * 4))  # 红外2
        infrared3 = struct.unpack('%dl' % (1024), f.read(1024 * 4))  # 红外3
        infrared4 = struct.unpack('%dl' % (1024), f.read(1024 * 4))  # 红外4
        infrared = (infrared1, infrared2, infrared3, infrared4)
        light = struct.unpack('%dl' % (1024), f.read(1024*4))  # 可见光
        f.close()

    nskip = 2048 + 32768
    for i in range(1,band):  # 计算跳过的投影数据头+定标值+通道数据n所占的字符数
        nskip += 2 * width * height

    realvalue = list()
    with open(read_path, 'rb') as fn:
        fn.read(nskip)
        if band < 5:  # 读的是红1,红2，红3，红4
            content = struct.unpack('%dH' % (height * width), fn.read(2 * height * width))
            for i in content:
                realvalue.append(infrared[band - 1][i] / 1000)
        else:  # 读的是可见光
            content = struct.unpack('%dB' % (height * width), fn.read(height * width))
            for i in content:
                realvalue.append(light[i] / 100000000)
        f.close()

    lon = np.linspace(ltlon, rtlon, width)
    lat = np.linspace(ltlat,lblat, height)
    cs = xr.Dataset({'prec': (['latitude', 'longitude'], np.array(realvalue).reshape(height, width))},
                    coords={'longitude': lon, 'latitude': lat})
    cs.to_netcdf(write_path, format='NETCDF3_64BIT')
    return '将fy2e格式文件转为nc格式文件成功！'


if __name__ == '__main__':
    '''参数为设定获取哪个通道的图像'''
    print(read_fy2e(1))
    # print(fy2e_to_nc(5))
