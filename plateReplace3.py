# -*- coding: utf-8 -*-

import cv2
import numpy as np

minPlateRatio = 1.5 # 车牌最小比例
maxPlateRatio = 3 # 车牌最大比例

# 图像处理
def imageProcess(gray):
    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (7, 7), 0, 0, cv2.BORDER_DEFAULT)
    
    # Sobel算子，X方向求梯度
    # sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 0, 1, ksize=3))
    # return sobel
    # 二值化
    ret, binary = cv2.threshold(gaussian, 150, 255, cv2.THRESH_BINARY)

    # 对二值化后的图像进行闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)

    # 再通过腐蚀->膨胀 去掉比较小的噪点
    erosion = cv2.erode(closed, None, iterations=2)
    dilation = cv2.dilate(erosion, None, iterations=2)

    # 返回最终图像
    return dilation

# 找到符合车牌形状的矩形
def findPlateNumberRegion(img):
    region = []

    # 查找外框轮廓
    contours_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print("contours lenth is :%s" % (len(contours)))
    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓面积
        area = cv2.contourArea(cnt)

        # 面积小的忽略
        if area < 1000 or area > 5000:
            continue

        # 转换成对应的矩形（最小）
        rect = cv2.minAreaRect(cnt)
        print("rect is:%s" % {rect})

        # 根据矩形转成box类型，并int化
        box = np.int32(cv2.boxPoints(rect))

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # print("the height&width is: %(height)s %(width)s" % {'height':height,'width':width})

        # 正常情况车牌长高比在1.5-3之间,那种两行的有可能小于3，这里不考虑
        ratio = float(width) / float(height)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        print("%(height)s %(width)s matched!" % {'height':height,'width':width})

        # 符合条件，加入到轮廓集合
        region.append(box)
    return region

def detect(img, replaceImg):
  # 转化成灰度图
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 形态学变换的处理
  dilation = imageProcess(gray)

  # 查找车牌区域
  region = findPlateNumberRegion(dilation)
  if len(region) == 0:
    return img

  # 默认取第一个
  box = region[0]

  #在原图画出轮廓
  cv2.drawContours(img, region, 0, (0, 255, 0), 2)
  return img
  # 找出box四个角的x点，y点，构成数组并排序
  ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
  xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
  ys_sorted_index = np.argsort(ys)
  xs_sorted_index = np.argsort(xs)

  # 取最小的x，y 和最大的x，y 构成切割矩形对角线
  min_x = box[xs_sorted_index[0], 0]
  max_x = box[xs_sorted_index[3], 0]
  min_y = box[ys_sorted_index[0], 1]
  max_y = box[ys_sorted_index[3], 1]

  return img


if __name__ == '__main__':
        imagePath = './img/1.jpg' # 图片路径
        img = cv2.imread(imagePath)

        replaceImgPath = './img/replace.jpg' # 替换图片路径
        replaceImg = cv2.imread(replaceImgPath)

        img = detect(img, replaceImg)
        cv2.imshow("img",img)
        cv2.waitKey(0)