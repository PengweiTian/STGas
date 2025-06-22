import contextlib
import copy
import io
import itertools
import json
import logging
import os
import warnings

import numpy as np
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

logger = logging.getLogger("STGas")


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class DetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, "coco_api")
        self.class_names = dataset.class_names
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids

        self.metric_names = ["AP10_95", "AP_50", "AP_75", "AP_s", "AP_m", "AP_l"]

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            temp_results = []
            temp_score = []
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score,
                    )
                    json_results.append(detection)
                    temp_results.append(detection)
                    temp_score.append(score)
            # print(f"image_id: {image_id}, len={len(temp_results)}, max={max(temp_score)}, min={min(temp_score)}")
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self.results2json(results)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, "results{}.json".format(rank))
        json.dump(results_json, open(json_path, "w"))
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox"
        )

        # # 默认IoU阈值 [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
        iou_thrs = coco_eval.params.iouThrs
        iou_thrs = np.concatenate((np.array([0.1, 0.3]), iou_thrs))  # 12个

        coco_eval.params.iouThrs = iou_thrs

        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        logger.info("\n" + redirect_string.getvalue())

        # 绘制表格
        headers = ["class", "AP10", "AP30", "AP50", "AP75", "AP95", "AP10:95"]
        columns = 7
        precisions = coco_eval.eval["precision"]
        # precisions.shape (IoU阈值格式)*101(maxDets指定在评估过程中每张图像上要考虑的最大检测框数量)*(类别数)*4*3
        per_class_ap = {"AP10": [], "AP30": [], "AP50": [], "AP75": [], "AP95": [], "AP10:95": []}
        per_class_ap_iou = [0.1, 0.3, 0.5, 0.75, 0.95, -1]
        iou2ap = {0.1: "AP10", 0.3: "AP30", 0.5: "AP50", 0.75: "AP75", 0.95: "AP95", -1: "AP10:95"}

        def get_precision_k(k):
            precision_k = precisions[k, :, idx, 0, -1] if k != -1 else precisions[:, :, idx, 0, -1]
            precision_k = precision_k[precision_k > -1]
            ap = np.mean(precision_k) if precision_k.size else float("nan")
            return float(ap * 100)

        for idx, _ in enumerate(self.class_names):
            for iou in per_class_ap_iou:
                iou_idx = coco_eval.params.iouThrs.tolist().index(iou) if iou != -1 else -1
                temp_ap = get_precision_k(iou_idx)
                per_class_ap[iou2ap[iou]].append(temp_ap)

        num_cols = min(columns, len(self.class_names) * len(headers))
        flatten_results = []
        for idx, name in enumerate(self.class_names):
            temp_result = [name]
            for _, ap in per_class_ap.items():
                temp_result.append(ap[idx])
            flatten_results += temp_result

        row_pair = itertools.zip_longest(
            *[flatten_results[i::num_cols] for i in range(num_cols)]
        )
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".1f",
            headers=table_headers,
            numalign="left",
        )
        logger.info("\n" + table)

        aps = coco_eval.stats[:6]
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
        return eval_results
