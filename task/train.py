import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
from thop import profile
import time

from STGas.evaluator import build_evaluator
from STGas.dataset.collate import batch_collate
from STGas.dataset import build_dataset
from STGas.util import cfg, load_config, STGasLightningLogger
from STGas.trainer import TrainingTask

import warnings

warnings.filterwarnings("ignore")


# 设置随机种子，确保每次运行得到相同的结果
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    set_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    # 创建日志
    logger = STGasLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)

    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")

    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=batch_collate,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=batch_collate,
        drop_last=True,
    )

    logger.info(f"Train data: len={len(train_dataloader)}, batch_size={cfg.device.batchsize_per_gpu}")
    logger.info(f"Val data: len={len(val_dataloader)}, batch_size={cfg.device.batchsize_per_gpu}")

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = task.model
    model = model.to(device)
    inputs = [torch.randn(16, 3, 256, 256).to(device)] * 10

    macs, params = profile(model, inputs=(inputs,))
    params_m = params / 1e6
    gflops = macs * 2 / 1e9  # 转换为GFLOPs
    print(f"模型GFLOPs: {gflops:.2f} GFLOPs, 参数量：{params_m:.2f}M")

    # 计算FPS
    model.eval()
    inputs = [torch.randn(1, 3, 256, 256).to(device)] * 10

    with torch.no_grad():
        for _ in range(10):
            model(inputs)
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            model(inputs)
            torch.cuda.synchronize()

    end_time = time.time()
    fps = (100) / (end_time - start_time)
    print(f"模型FPS: {fps:.2f}")

    # trainer = pl.Trainer(
    #     default_root_dir=cfg.save_dir,
    #     max_epochs=cfg.schedule.total_epochs,
    #     gpus=cfg.device.gpu_ids,
    #     check_val_every_n_epoch=cfg.schedule.val_intervals,
    #     accelerator="gpu",
    #     log_every_n_steps=cfg.log.interval,
    #     num_sanity_val_steps=0,
    #     resume_from_checkpoint=None,
    #     callbacks=[ProgressBar(refresh_rate=0)],
    #     logger=logger,
    #     benchmark=True,
    #     gradient_clip_val=cfg.get("grad_clip", 0.0),
    # )
    #
    # trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
