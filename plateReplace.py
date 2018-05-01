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
    ret, binary = cv2.threshold(gaussian, 130, 255, cv2.THRESH_BINARY)

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
    rectlist = []
    sizelist = []

    # 查找外框轮廓
    contours_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print("contours lenth is :%s" % (len(contours)))
    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓面积
        area = cv2.contourArea(cnt)

        # 面积太大太小的忽略
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
        print("the height&width is: %(height)s %(width)s" % {'height':height,'width':width})

        # 正常情况车牌长高比在1.5-3之间
        ratio = float(width) / float(height)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        print("%(height)s %(width)s matched!" % {'height':height,'width':width})

        # 符合条件，加入到轮廓集合
        rectlist.append(rect)
        region.append(box)
        sizelist.append([height, width])
    return region,rectlist,sizelist

def detect(img, replaceImg):
  imgLogo = img.copy()

  # 转化成灰度图
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 形态学变换的处理
  dilation = imageProcess(gray)

  # 查找车牌区域
  region, rectlist, sizelist = findPlateNumberRegion(dilation)
  if len(region) == 0:
    return img,img

  # 默认取第一个
  box = region[0]
  rect = rectlist[0]
  size = sizelist[0]
  size_height = size[0]
  size_width = size[1]

  # 在原图画出轮廓
  cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

  # 混合LOGO到牌照的位置
  start_x = box[1, 1]
  start_y = box[1, 0]
  imageROI = imgLogo[start_x:start_x+size_height, start_y:start_y+size_width]
  replaceImgBk = cv2.resize(replaceImg, (size_width,size_height), interpolation = cv2.INTER_CUBIC)

  cv2.addWeighted(imageROI, 0, replaceImgBk, 1, 0, imageROI)

  return img, imgLogo


if __name__ == '__main__':
        imagePath = './img/1.jpg' # 图片路径
        img = cv2.imread(imagePath)

        replaceImgPath = './img/replace.jpg' # 替换图片路径
        replaceImg = cv2.imread(replaceImgPath)

        img, imgLogo = detect(img, replaceImg)

        cv2.imwrite('./img/1_position.jpg', img)
        cv2.imwrite('./img/1_logo.jpg', imgLogo)

        cv2.imshow("img",img)
        cv2.waitKey(0)