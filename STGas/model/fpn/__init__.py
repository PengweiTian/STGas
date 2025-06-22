import copy

from .ghost_pan import GhostPAN


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop("name")
    if name == "GhostPAN":
        return GhostPAN(**fpn_cfg)
    else:
        raise NotImplementedError
