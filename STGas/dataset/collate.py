import numpy as np
import torch

"""
multi_frame_count=3时，batch_size=1
batch: [
    [
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'}
    ]
]
batch_size=3
batch: [
    [
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'}
    ],
    [
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'}
    ],
    [
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'},
        {'img','img_info','gt_bbox','gt_label'}
    ]
]

multi_frame_count=3时
目标batch:
    {
        "img":[tensor(bs*c*h*w),tensor(bs*c*h*w),tensor(bs*c*h*w)],
        "img_info":{"file_name":[],"width":[],"height":[]}len=bs的[每个bs的key_frames的img_info]
        "gt_bbox":len=bs的[每个bs的key_frames的gt_bbox]==>[[],[],..]
        "gt_label":len=bs的[每个bs的key_frames的gt_label]==>[[],[],..]
        "warp_matrix":len=bs的[[],[],...]
    }
"""


def batch_collate(batch):
    result = {"img": [], "img_info": {'file_name': [], "height": [], "width": [], "id": []},
              "gt_bbox": [], "gt_label": [], "warp_matrix": []}

    k = len(batch[0])
    multi_frame_list = [[frame_series[idx]['img'] for frame_series in batch] for idx in range(k)]
    batch_img_tensor_list = [torch.stack(item, dim=0) for item in multi_frame_list]
    result["img"] = batch_img_tensor_list

    for frame_series in batch:
        for frame in frame_series[1:-1]:
            result["img_info"]["file_name"].append(frame["img_info"]["file_name"])
            result["img_info"]["height"].append(frame["img_info"]["height"])
            result["img_info"]["width"].append(frame["img_info"]["width"])
            result["img_info"]["id"].append(frame["img_info"]["id"])
            # 只有一个目标
            result["gt_bbox"].append([frame["gt_bbox"]])
            result["gt_label"].append([frame["gt_label"]])
            result["warp_matrix"].append(frame["warp_matrix"])

    result["gt_bbox"] = np.array(result["gt_bbox"]).astype(np.float32)
    result["gt_label"] = np.array(result["gt_label"]).astype(np.int64)

    return result
