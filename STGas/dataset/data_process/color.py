import numpy as np
import cv2
import random


# 调整亮度
def random_brightness(img_list, brightness_delta, brightness_prob=0.5):
    if random.random() < brightness_prob:
        brig = random.uniform(-brightness_delta, brightness_delta)
        for i in range(len(img_list)):
            img_list[i] += brig
    return img_list


# 调整对比度
def random_contrast(img_list, contrast_lower, contrast_upper, contrast_prob=0.5):
    if random.random() < contrast_prob:
        cont = random.uniform(contrast_lower, contrast_upper)
        for i in range(len(img_list)):
            img_list[i] *= cont
    return img_list


# 调整饱和度
def random_saturation(img_list, saturation_lower, saturation_upper, saturation_prob=0.5):
    if random.random() < saturation_prob:
        satu = random.uniform(saturation_lower, saturation_upper)
        for i in range(len(img_list)):
            hsv = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] *= satu
            img_list[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_list


# 随机调整色调
def random_hue(img_list, hue_delta, hue_prob=0.5):
    if random.random() < hue_prob:
        hue = random.uniform(-hue_delta, hue_delta)
        for i in range(len(img_list)):
            hsv = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] += hue
            img_list[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_list


def high_reserve(img, ksize, sigm):
    img = img * 1.0
    gauss_out = cv2.GaussianBlur(img, (ksize, ksize), sigm)
    img_out = img - gauss_out + 128
    img_out = img_out / 255.0
    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    return img_out


def usm(img, number):
    blur_img = cv2.GaussianBlur(img, (0, 0), number)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


def Overlay(target, blend):
    mask = blend < 0.5
    img = 2 * target * blend * mask + (1 - mask) * (1 - 2 * (1 - target) * (1 - blend))
    return img


def img_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    1.分离背景层和细节层:
    """
    r = 3  # 值越小,细节越多
    eps1 = 1e-5  # 值越小,细节越多,但噪声大
    gray = image.astype(np.float32)
    input = image.astype(np.float32)
    # 引导滤波
    low_img = cv2.ximgproc.guidedFilter(gray, input, r, (eps1 * 255 * 255))
    low_img = np.clip(low_img, 0, 255).astype(np.uint8)
    high_img = image - low_img

    """
    2.背景层:
    """
    adaptive_clip = np.clip(15 * (np.std(low_img) / 30), 5, 20)  # 根据背景层噪声水平调整
    clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(5, 5))
    low_img = clahe.apply(low_img)
    """
    3.对细节层进行细节增强和噪声抑制:
        非线性锐化(拉普拉斯算子,USM), 边缘增强(Sobel,Canny), 总变分去噪(TVD)
        双边滤波, Wiener去噪, 非局部均值去噪, 自适应双边滤波
    """
    # # 自适应锐化
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    high_img = cv2.filter2D(high_img, -1, kernel)
    adaptive_clip = np.clip(15 * (np.std(high_img) / 30), 5, 25)
    clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(16, 16))
    high_img = clahe.apply(high_img)

    high_img = high_img.astype(np.float32)
    high_img = 255 * (high_img / 255) ** 0.7
    high_img = high_img.astype(np.uint8)
    """
    5.合成输出图像
    """
    blended = cv2.addWeighted(low_img, 0.8, high_img, 0.2, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


# 归一化
def normalize(img_list, param):
    mean, std = param
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img_list = [(img - mean) / std for img in img_list]
    return img_list


class ImgColor:
    def __init__(self, brightness, contrast, hue, saturation, normalize):
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.normalize = normalize

    def img_process(self, img_list, mode):
        img_list = [img.astype(np.float32) for img in img_list]
        if mode == "train":
            if random.random() > 0.5:
                img_list = random_brightness(img_list, self.brightness)
                img_list = random_contrast(img_list, self.contrast[0], self.contrast[1])
                img_list = random_saturation(img_list, self.saturation[0], self.saturation[1])
                img_list = random_hue(img_list, self.hue)
            else:
                img_list = random_brightness(img_list, self.brightness)
                img_list = random_saturation(img_list, self.saturation[0], self.saturation[1])
                img_list = random_hue(img_list, self.hue)
                img_list = random_contrast(img_list, self.contrast[0], self.contrast[1])
        img_list = [img / 255 for img in img_list]
        # 对测试集只进行归一化操作
        img_list = normalize(img_list, self.normalize)
        return img_list
