# -*- coding: utf-8 -*-
__author__ = 'wangwenhui'

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



class OptFlow:
    """windows下光流法的案例代码"""

    def __init__(self, read_path, filename1, filenam2, save_image):
        self.read_path = read_path
        self.image1 = os.path.join(self.read_path, filename1)
        self.image2 = os.path.join(self.read_path, filenam2)
        self.save_image = save_image

    def flow_pyrlk(self):
        """'通过金字塔Lucas_kanade（基于梯度方法）光流法计算某些点集的光流（稀疏光流）"""
        image1 = cv2.imread(self.image1)
        image2 = cv2.imread(self.image2)
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        # 金字塔Lucas kandade光流法的参数
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # 创建随机颜色
        color = np.random.randint(0, 255, (100, 3))
        # 找到第一张灰度图
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # 获取图像中的角点，返回到p0中
        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
        # 创建一个蒙版用来画轨迹
        mask = np.zeros_like(image1)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
        # 选择好的跟踪点
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # 画出轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            cv2.circle(image2, (a, b), 2, color[i].tolist(), -1)

        img = cv2.add(image2, mask)
        # cv2.imwrite(self.save_image,img)
        plt.imshow(img)
        plt.show()
        return None

    def flow_farneback(self):
        """用Gunnar Farnback的算法计算稠密光流（即图像上所有像素点的光流都计算出来）"""
        img1 = cv2.imread(self.image1)
        img2 = cv2.imread(self.image2)
        hsv = np.zeros_like(img1)
        hsv[..., 1] = 255
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 6, 15, 3, 5, 1.2, 0)
        '''
        gray1: 输入前一帧图像;
        gray2: 输入后一帧图像;
        None: 输出的光流;
        0.5: 金字塔上下两层的尺度关系;
        6: 金字塔层数;
        15: 均值窗口大小，越大越能denoise并且能检测快速移动目标，但会引起模糊运动区域;
        3: 迭代次数;
        5: 像素领域范围大小，一般为5、7等;
        1.2: 高斯标准差，一般为1-1.5（函数处理中需要高斯分布权重）;
        0: 计算方法，包括2种
        '''
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        hsv[..., 0] = ang
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # plt.subplot(131),plt.imshow(img1)
        # plt.subplot(132),plt.imshow(img2)
        # plt.subplot(133),plt.imshow(rgb)
        # cv2.imwrite(self.save_image,rgb)
        # plt.show()
        return flow

    def warp_flow(self, img, flow):
        """获取第t帧图片和flow，获得第t+1时刻图片"""
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        # 重映射，双线性插值
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res


def main():
    read_path = r'F:\chenqing\varFlow\picture'
    image1 = 'MOSAICHREF000.20161231.160000.png'
    image2 = 'MOSAICHREF000.20161231.161000.png'
    save_path = r'F:\chenqing\varFlow\flowimage\forecast_MOSAICHREF000.20161231.162000.png'
    fp = OptFlow(read_path, image1, image2, save_path)
    flow = fp.flow_farneback()
    img2 = cv2.imread(os.path.join(read_path, image2))
    cur_glitch = img2.copy()
    cur_glitch = fp.warp_flow(cur_glitch, flow)
    plt.imshow(cur_glitch)
    plt.savefig(save_path)
    plt.show()
    return None


if __name__ == '__main__':
    main()
    # # 批量处理
    # for image_index in range(1,len(list_dir)):
    #     save_image=os.path.join(save_path,"minist_%d.png"%(image_index))
    #     fp=OptFlow(read_path, list_dir[image_index - 1], list_dir[image_index], save_image)
    #     fp.flow_pyrlk()
