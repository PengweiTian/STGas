from . import expand_param, mean_values, batch_samplers
from .color import ImgColor
from .transform import ImgTransform
from .box_sample import apply_expand, crop_image


class Pipeline:
    def __init__(self, mode, cfg):
        self.mode = mode
        self.transform_tool = ImgTransform(**cfg.transform)
        self.color_tool = ImgColor(**cfg.color)

    def multi_img_process(self, multi_img, multi_bbox, dst_shape):
        multi_img = self.color_tool.img_process(multi_img, self.mode)
        img_list, bbox_list, warp_matrix = self.transform_tool.img_process(multi_img, multi_bbox, dst_shape, self.mode)
        return img_list, bbox_list, warp_matrix
