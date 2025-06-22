import os
import json
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path

from pycocotools.coco import COCO
from .data_process.pipeline import Pipeline
from .data_process.color import img_process


class IODVideoDataset(Dataset):

    def __init__(self, img_path, ann_path, coco_ann_path, valid_coco_ann_path, input_size, down_ratio, frame_seg,
                 pipeline, mode, model):
        super().__init__()
        self.frame_seg = frame_seg
        self.mode = mode
        self.IODVideo = IODVideo(img_path, ann_path, coco_ann_path, valid_coco_ann_path, input_size, down_ratio,
                                 frame_seg, pipeline, mode, model)
        # k=1,bs=1,train_len=94220
        # k=1,bs=1,val_len=46197
        self.data_list = self.IODVideo.get_data_list()  # (video_id,frame_id)==>(video_id,frame_id+k)

        self.coco_api = self.IODVideo.coco_api
        self.cat_ids = self.IODVideo.cat_ids
        self.class_names = self.IODVideo.class_names

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.IODVideo.get_data_item(self.data_list[index])


# IODVideo数据处理
class IODVideo:
    def __init__(self, img_path, ann_path, coco_ann_path, valid_coco_ann_path, input_size, down_ratio, frame_seg,
                 pipeline_cfg, mode, model):
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.down_ratio = down_ratio
        self.frame_seg = frame_seg
        self.mode = mode
        # 读取数据
        self.video_data = json.load(open(ann_path, 'r', encoding='utf-8'))["video_dict"]

        items = list(self.video_data.items())
        random.shuffle(items)
        self.video_data = dict(items)
        self.video_list = list(self.video_data.keys())

        # 加载数据预处理工具
        self.pipeline = Pipeline(mode, pipeline_cfg)

        self.valid_coco_ann_path = valid_coco_ann_path
        # 加载COCO
        self.coco_api = COCO(coco_ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
        self.img2id = {img["file_name"]: idx for idx, img in self.coco_api.imgs.items()}

        self.extra_count = 0
        if model == "CTDFF":
            self.extra_count = 2  # t-1,t,t+1
        elif model == "TDN":
            self.extra_count = 4  # t-2,t-1,t,t+1,t+2
        self.frames_count = frame_seg + self.extra_count

    def get_data_list(self):
        data_list = []
        for idx, (video, frames) in enumerate(self.video_data.items()):
            i = 0
            while i < len(frames) - self.frames_count + 1:
                data_list.append((idx, i))
                i += self.frames_count
        self.reset_coco_api(data_list)
        return data_list

    def get_data_item(self, idx_data):
        video_name = self.video_list[idx_data[0]]
        frame_start = idx_data[1]
        # frame_end = idx_data[1] + self.frame_seg
        frame_end = idx_data[1] + self.frames_count
        frame_data_list = self.video_data[video_name][frame_start:frame_end]

        # 数据预处理
        process_multi_img, process_multi_bbox, warp_matrix = self.data_process(frame_data_list)

        result_data = []
        for idx, frame_data in enumerate(frame_data_list):
            img_info = frame_data["image"]
            img_info["id"] = self.img2id[img_info["file_name"]]

            img_data = torch.from_numpy(process_multi_img[idx].transpose(2, 0, 1))
            bbox_data = process_multi_bbox[idx][0]  # 单实例

            item_data = {"img": img_data, "img_info": img_info, "gt_bbox": bbox_data,
                         "gt_label": frame_data["annotation"]["category"], "warp_matrix": warp_matrix}
            result_data.append(item_data)

        return result_data

    def data_process(self, frame_list):
        process_data = {"img": [], "bbox": []}
        for frame_data in frame_list:
            img_data = self.read_img_file(frame_data["image"]["file_name"])
            bbox_data = np.array([frame_data["annotation"]["bbox"]]).astype(np.float32)  # 单实例
            process_data["img"].append(img_data)
            process_data["bbox"].append(bbox_data)

        # 对连续k帧使用相同的数据预处理
        return self.pipeline.multi_img_process(process_data["img"], process_data["bbox"], self.input_size)

    def read_img_file(self, file_name):
        image_path = str(os.path.join(self.img_path, file_name))
        img = cv2.imread(image_path).astype(np.float32)
        img = img_process(img).astype(np.float32)
        return img

    # 根据Valid重新生成COCO
    def reset_coco_api(self, data_list):
        result = {'images': [], 'type': "detect", 'annotations': [], 'categories': []}
        result["categories"].append({
            "id": 1,
            "name": "gas",
            "supercategory": 'none'
        })
        temp_count = 0
        for idx, item in enumerate(data_list):
            video_name = self.video_list[item[0]]
            # 挑取关键检测帧
            frame_start = item[1] + (self.extra_count // 2)
            frame_end = item[1] + self.frames_count - (self.extra_count // 2)

            key_frames = self.video_data[video_name][frame_start:frame_end]
            for key_frame in key_frames:
                img_info = key_frame["image"]
                img_info["id"] = self.img2id[img_info["file_name"]]
                bbox = key_frame["annotation"]["bbox"]
                coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                annotation_info = {
                    "id": temp_count,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                    "ignore": 0,
                    "image_id": img_info["id"],
                    "bbox": coco_bbox,
                    "category_id": 1
                }
                result["images"].append(img_info)
                result["annotations"].append(annotation_info)
                temp_count += 1
        # Windows
        root_path = '\\'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split("\\")[:-1])
        save_path = root_path + "\\data\\IODVideo\\valid_coco_eval.json"

        # Linux
        # root_path = Path(__file__).resolve().parents[2]  # 向上两级目录
        # save_path = root_path / "data" / "IODVideo" / "valid_coco_eval.json"

        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

        self.coco_api = COCO(self.valid_coco_ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
