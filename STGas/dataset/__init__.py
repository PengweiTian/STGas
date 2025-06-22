import copy
from .IODVideoDataset import IODVideoDataset


# 选择数据集
def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    if name == "IODVideo":
        return IODVideoDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")
