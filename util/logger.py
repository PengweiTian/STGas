import logging
import os
import time

import numpy as np
from termcolor import colored
import fsspec
from fsspec.implementations.local import AbstractFileSystem, LocalFileSystem

from .path import mkdir


class Logger:
    def __init__(self, local_rank, save_dir="./", use_tensorboard=True):
        mkdir(local_rank, save_dir)
        self.rank = local_rank
        fmt = (
                colored("[%(name)s]", "magenta", attrs=["bold"])
                + colored("[%(asctime)s]", "blue")
                + colored("%(levelname)s:", "green")
                + colored("%(message)s", "white")
        )
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(save_dir, "logs.txt"),
            filemode="w",
        )
        self.log_dir = os.path.join(save_dir, "logs")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                ) from None
            if self.rank < 1:
                logging.info(
                    "Using Tensorboard, logs will be saved in {}".format(self.log_dir)
                )
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    def scalar_summary(self, tag, phase, value, step):
        if self.rank < 1:
            self.writer.add_scalars(tag, {phase: value}, step)


class MovingAverage(object):
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val):
        self.reset()
        self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class CustomLogger:
    def __init__(self, save_dir="./", **kwargs):
        super().__init__()
        self._name = "Model"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(save_dir, f"logs-{self._version}")

        self._fs = get_filesystem(save_dir)
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self._init_logger()

        self._experiment = None
        self._kwargs = kwargs

    def _init_logger(self):
        self.logger = logging.getLogger(name=self._name)
        self.logger.setLevel(logging.INFO)

        # create file handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        # set file formatter
        f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
        file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # set console formatter
        c_fmt = (
                colored("[%(name)s]", "magenta", attrs=["bold"])
                + colored("[%(asctime)s]", "blue")
                + colored("%(levelname)s:", "green")
                + colored("%(message)s", "white")
        )
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, string):
        self.logger.info(string)

    def log(self, string):
        self.logger.info(string)

    def dump_cfg(self, cfg_node):
        with open(os.path.join(self.log_dir, "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)

    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")

    def log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")


def get_filesystem(path):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    # use local filesystem
    return LocalFileSystem()
