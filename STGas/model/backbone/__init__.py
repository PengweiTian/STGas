import copy

from .STGas import STGas
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from .ghostnet import GhostNet
from .efficientnet_lite import EfficientNetLite
from .mobilevit import MobileViT
from .repvit import repvit_m1_1


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "STGas":
        return STGas(**backbone_cfg)
    elif name == "ResNet18":
        return ResNet(**backbone_cfg)
    elif name == "ShuffleNetv2":
        return ShuffleNetV2(**backbone_cfg)
    elif name == "MobileNetv3":
        return MobileNetV3_Large(**backbone_cfg)
    elif name == "GhostNet":
        return GhostNet(**backbone_cfg)
    elif name == "EfficientNet":
        return EfficientNetLite(**backbone_cfg)
    elif name == "MobileViT":
        return MobileViT(**backbone_cfg)
    elif name == "RepViT":
        return repvit_m1_1(**backbone_cfg)
    else:
        raise NotImplementedError
