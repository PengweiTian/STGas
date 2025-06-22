from .CTDFF import CTDFF
from .TDN import TDN


def build_temporal_backbone(name, frame_seg):
    if name == "CTDFF":
        return CTDFF(frame_seg)
    elif name == "TDN":
        return TDN(frame_seg)
    else:
        raise NotImplementedError
