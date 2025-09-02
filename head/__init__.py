import copy

from .nanodet_plus_head import NanoDetPlusHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    else:
        raise NotImplementedError
