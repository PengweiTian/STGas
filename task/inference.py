import os
import queue
import argparse

import time

import numpy as np
import cv2
import torch

from STGas.dataset.data_process.pipeline import Pipeline
from STGas.dataset.data_process.color import img_process
from STGas.model.arch import build_model
from STGas.util import (
    load_model_weight,
    STGasLightningLogger,
    cfg,
    load_config,
    mkdir,
)

import warnings

warnings.filterwarnings("ignore")


class Predictor:
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline("test", cfg.data.val.pipeline)

    def inference(self, img_list):
        key_img = img_list[len(img_list) // 2]
        key_img = cv2.resize(key_img, (256, 256))
        height, width = img_list[0].shape[:2]
        img_info = {"height": [height], "width": [width], "id": [0]}
        # img_list = [img_process(item) for item in img_list]
        # img_list = [img.astype(np.float32) for img in img_list]
        # img_data = [cv2.resize(img, tuple(self.cfg.data.val.input_size), interpolation=cv2.INTER_LINEAR) for img in img_list]
        img_data, _, warp_matrix = self.pipeline.multi_img_process(img_list, [], self.cfg.data.val.input_size)
        img_data = [torch.from_numpy(img_data[idx].transpose(2, 0, 1)).unsqueeze(0).contiguous().to(self.device)
                    for idx in range(len(img_data))]

        meta = dict(img_info=img_info, img=img_data, warp_matrix=[warp_matrix])
        with torch.no_grad():
            results = self.model.inference(meta)

        # 绘制检测框与标签
        res = self.model.head.show_result(key_img, results[0], self.cfg.class_names, score_thres=0.35, show=False)
        return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    local_rank = -1

    cfg.defrost()
    cfg.save_dir = os.path.join(cfg.save_dir, "inference_logger")
    mkdir(local_rank, cfg.save_dir)

    logger = STGasLightningLogger(cfg.save_dir)

    detector = Predictor(cfg, args.model, logger)

    # 推理
    frame_seg = 4
    frame_seg += 2  # STGas
    # frame_seg += 4  # TDN
    video_name = "491_valve_dynamic_clear.mp4"
    # video_name = "202_experiment_dynamic_clear.mp4"
    # video_name = "370_pipeline_dynamic_vague.mp4"
    # video_name = "546_cylinder_dynamic_clear.mp4"
    video_path = "D:\\tpw\\tpw_graduate\\dataset\\IOD-Video\\IOD-Video_mp4"
    file_path = str(os.path.join(video_path, video_name))
    frame_queue = queue.Queue(maxsize=frame_seg)
    cap = cv2.VideoCapture(file_path)
    # out = cv2.VideoWriter('simple.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (320, 320))
    # out = cv2.VideoWriter('difficult.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (320, 320))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            # 检测
            frame_list = [item for item in frame_queue.queue]
            # 计算FPS
            detect_start_time = time.time()
            detect_frame = detector.inference(frame_list)
            detect_time = time.time() - detect_start_time
            fps = 1 / detect_time
            print(f"FPS={fps}")
            cv2.imshow('Video Frame', detect_frame)
            # out.write(detect_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            frame_queue.get()
        frame_queue.put(frame)
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
