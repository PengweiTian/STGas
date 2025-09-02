import copy

from .ghost_pan import GhostPAN


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop("name")
    if name == "GhostPAN":
        return GhostPAN(**neck_cfg)
    else:
        raise NotImplementedError
