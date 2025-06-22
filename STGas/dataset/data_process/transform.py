import numpy as np
import cv2
import random
import math
from .box_sample import crop_image
from . import batch_samplers


# 翻转
def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


# 透视变换
def get_perspective_matrix(perspective=0.0):
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    return P


# 旋转变换
def get_rotation_matrix(degree=0.0):
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


# 缩放变换
def get_scale_matrix(ratio=(1, 1)):
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


# 拉伸变换
def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


# 裁剪
def get_shear_matrix(degree):
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # x shear (deg)
    Sh[1, 0] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # y shear (deg)
    return Sh


# 平移操作
def get_translate_matrix(translate, width, height):
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
    return T


# 调整尺寸
def get_resize_matrix(raw_shape, dst_shape):
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    Rs[0, 0] *= d_w / r_w
    Rs[1, 1] *= d_h / r_h
    return Rs


# 对bbox也变换
def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def reshape_bbox(bbox, old_shape, dst_shape):
    res_bbox = []
    o_w, o_h = old_shape
    d_w, d_h = dst_shape
    scale_x = d_w / o_w
    scale_y = d_h / o_h
    for box in bbox:
        x_min, y_min, x_max, y_max = box
        res_bbox.append([x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y])
    return np.array(res_bbox).astype(np.float32)


class ImgTransform:
    """
    调整大小、随机视角、随机比例、随机拉伸、随机旋转、随机剪切、随机平移、随机翻转、亮度调整、对比度调整、饱和度调整

    Args:
        perspective: 透视变换因子.
        scale: 随机比例.
        stretch: 宽度和高度拉伸比范围.
        rotation: 随机旋转度.
        shear: 随机剪切度.
        translate: 随机翻译比率.
        flip: 随机翻转概率.
    """

    def __init__(self, perspective, scale, stretch, rotation, shear, translate, flip):
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate

    def img_process(self, img_list, bbox_list, dst_shape, mode):
        height, width = img_list[0].shape[:2]

        if mode == "train":
            C = np.eye(3)
            C[0, 2] = -width / 2
            C[1, 2] = -height / 2

            # 透视变换
            P = get_perspective_matrix(self.perspective)
            C = P @ C

            # 缩放变换
            Scl = get_scale_matrix(self.scale_ratio)
            C = Scl @ C

            # 拉伸变换
            Str = get_stretch_matrix(*self.stretch_ratio)
            C = Str @ C

            # 旋转变换
            R = get_rotation_matrix(self.rotation_degree)
            C = R @ C

            # 裁剪
            Sh = get_shear_matrix(self.shear_degree)
            C = Sh @ C

            # 翻转
            F = get_flip_matrix(self.flip_prob)
            C = F @ C

            # 平移
            T = get_translate_matrix(self.translate_ratio, width, height)
            C = T @ C

            ResizeM = get_resize_matrix((width, height), dst_shape)
            C = ResizeM @ C  # warp_matrix

            res_img_list = [cv2.warpPerspective(img, C, dsize=tuple(dst_shape)) for img in img_list]
            res_bbox_list = [warp_boxes(bbox, C, dst_shape[0], dst_shape[1]) for bbox in bbox_list]

            return res_img_list, res_bbox_list, C

        res_img_list = [cv2.resize(img, tuple(dst_shape), interpolation=cv2.INTER_LINEAR) for img in img_list]
        res_bbox_list = [reshape_bbox(bbox, (width, height), tuple(dst_shape)) for bbox in bbox_list]
        return res_img_list, res_bbox_list, None

# import os
#
#
# def draw_box_to_img(image, pos):
#     pos = pos.astype(int).tolist()[0]
#     top_left_corner = (pos[0], pos[1])
#     bottom_right_corner = (pos[2], pos[3])
#     rect_color = (0, 0, 255)
#     rect_thickness = 3
#     image = cv2.rectangle(image, top_left_corner, bottom_right_corner, rect_color, rect_thickness)
#     return image
#
#
# if __name__ == '__main__':
#     img_path = "D:\\tpw\\tpw_graduate\\dataset\\IOD-Video\\Frames"
#     file_name = "TrueLeakedGas/367_experiment_static_clear/00013.png"
#     gt_box = np.array([[115, 175, 305, 236]]).astype(np.float32)
#     image_path = str(os.path.join(img_path, file_name))
#     image = cv2.imread(image_path)
#     # image = draw_box_to_img(image, gt_box)
#     # cv2.imshow("img1", image)
#
#     temp = ImgTransform(perspective=0.0, scale=[0.6, 1.4], stretch=[[0.8, 1.2], [0.8, 1.2]], rotation=0, shear=0,
#                         translate=0.2,
#                         flip=0.5)
#
#     image, gt_box, C = temp.img_process([image], [gt_box], (320, 320), "train")
#
#     box_image = draw_box_to_img(image[0], gt_box[0])
#
#     cv2.imshow("img1", box_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
