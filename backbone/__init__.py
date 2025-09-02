import copy

from .STGas import STGas


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "STGas":
        return STGas(**backbone_cfg)
    else:
        raise NotImplementedError
