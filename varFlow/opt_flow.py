#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


def draw_flow(img, flow, step=16):
    '''作出光流图'''
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    '''获取第t帧图片和flow，获得第t+1时刻图片'''
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def main():
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0
    # cam = video.create_capture(fn)
    # red返回的是一个值，prev返回的是一个图像
    # ret, prev = cam.read()
    read_path = r'D:\PycharmProjects\bigdata\stage_03\minist'
    image1 = 'minist_0.png'
    # 默认为1,1为加载彩色图
    prev = cv.imread(os.path.join(read_path, image1))
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = True
    # ret, img = cam.read()
    image1 = 'minist_1.png'
    gray = cv.imread(os.path.join(read_path, image1))
    cur_glitch = gray.copy()
    # 从BGR颜色空间转换到灰度空间，数据类型和位深与源图像一致
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # prevgray = gray
    cv.imshow('flow', draw_flow(gray, flow))
    cv.imwrite(r'D:\PycharmProjects\bigdata\stage_03\flowimage\draw.png', draw_flow(gray, flow))

    if show_hsv:
        cv.imshow('flow HSV', draw_hsv(flow))
        cv.imwrite(r'D:\PycharmProjects\bigdata\stage_03\flowimage\hsv.png', draw_hsv(flow))

    if show_glitch:
        cur_glitch = warp_flow(cur_glitch, flow)
        cv.imshow('glitch', cur_glitch)
        pic = cv.cvtColor(cur_glitch, cv.COLOR_BGR2GRAY)
        plt.imshow(pic)
        plt.savefig(r'D:\PycharmProjects\bigdata\stage_03\flowimage\pic.png')  # 外推数据
        plt.show()
    # 图片显示着，直到你按下任意一个键，才被关掉
    cv.waitKey(0)
    # 结束所有窗口
    cv.destroyAllWindows()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
