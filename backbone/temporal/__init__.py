from .CTDFF import CTDFF


def build_temporal_backbone(name, frame_seg):
    if name == "CTDFF":
        return CTDFF(frame_seg)
    else:
        raise NotImplementedError
