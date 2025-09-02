import torch.nn as nn

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck(cfg.neck)
        self.head = build_head(cfg.head)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def train_loss(self, batch):
        images = [item.cuda() for item in batch["img"]]
        predict = self(images)
        loss, loss_states = self.head.loss(predict, batch)
        return predict, loss, loss_states

    def post_process(self, predict, batch):
        results = self.head.post_process(predict, batch)
        return results

    def inference(self, batch):
        images = [item.cuda() for item in batch["img"]]
        predict = self(images)
        results = self.head.post_process(predict, batch)
        return results
