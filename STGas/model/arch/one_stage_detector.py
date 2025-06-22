import time

import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


# 一阶段检测网络: YOLO,SSD,NanoDet
class OneStageDetector(nn.Module):
    def __init__(self, backbone_cfg, fpn_cfg, head_cfg):
        super(OneStageDetector, self).__init__()
        self.temporal = backbone_cfg.temporal

        self.backbone = build_backbone(backbone_cfg)
        self.fpn = build_fpn(fpn_cfg)
        self.head = build_head(head_cfg)
        self.epoch = 0

    def forward(self, x):
        if self.temporal in ["TSM","S3D","MSNet"]:
            x = torch.cat(x, dim=0)
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)
        return x

    def inference(self, x):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            predict = self(x["img"])
            torch.cuda.synchronize()
            time2 = time.time()
            print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")
            results = self.head.post_process(predict, x)
            torch.cuda.synchronize()
            print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")
        return results

    def forward_train(self, batch):
        predict = self(batch["img"])
        loss, loss_states = self.head.loss(predict, batch)
        return predict, loss, loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch
