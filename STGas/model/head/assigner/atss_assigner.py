# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from tensorboard.compat.tensorflow_stub.dtypes import float32

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk):
        # 这个topK就是对每个gt box选取的候选框的个数
        self.topk = topk

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(
            self, bboxes, num_level_bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None
    ):
        """
        bboxes: 2100*4
        num_level_bboxes: [1600,400,100]
        gt_bboxes: 1*1
        gt_bboxes_ignore=None,
        gt_labels: 1*1

        Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt

        1.计算所有bbox（所有金字塔级别的bbox）和gt之间的iou
        2.计算所有bbox和gt之间的中心距离
        3.在每个金字塔级别上，对于每个gt，选择k个距离gt中心最近的bbox，因此我们总共选择k*l个bbox作为每个gt的候选框
        4.为这些候选框获取相应的iou，并计算均值和标准差，将（均值+标准差）设置为iou阈值
        5.选择iou大于或等于iou阈值的候选框作为正样本
        6.在gt中限制正样本的中心位置


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        # 2100*4
        bboxes = bboxes[:, :4]
        # num_gt=1,num_bboxes=2100
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)  # num_gt目标个数 num_bboxes2100

        # compute iou between all bbox and gt
        # 1.计算所有bbox（所有金字塔级别的bbox）和gt之间的iou
        overlaps = bbox_overlaps(bboxes, gt_bboxes)  # 2100*1

        # assign 0 by default
        # assigned_gt_inds: (1,)==>为0
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        # compute center distance between all bbox and gt
        # 2.计算所有bbox和gt之间的中心距离
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)  # 1*2

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        # print(f"max_x:{bboxes[:, 2] - bboxes[:, 0]}, max_y:{bboxes[:, 3] - bboxes[:, 1]}")
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)  # 2100*2

        # bboxes_points[:, None, :] 2100*1*2
        # gt_points[None, :, :] 1*1*2
        distances = (
            (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        )
        # show_overlaps(distances)
        # distances 2100*1

        # Selecting candidates based on the center distance
        # 3.在每个金字塔级别上，对于每个gt，选择k个距离gt中心最近的bbox，因此我们总共选择k*l个bbox作为每个gt的候选框,l为预测图个数
        candidate_idxs = []
        start_idx = 0
        # 40*40，20*20，10*10：num_level_bboxes=[1600,400,100]
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            # 选取当前层次预测图与gt的中心距离
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            # 选择最近的k个距离
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False
            )
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # 4.为这些候选框获取相应的iou，并计算均值和标准差，将（均值+标准差）设置为iou阈值
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        # min_thr = torch.tensor([0.3], dtype=torch.float32).to(overlaps_mean_per_gt.device)
        # overlaps_thr_per_gt = max(overlaps_mean_per_gt + 0.5 * overlaps_std_per_gt, min_thr)
        # print(f"mean={overlaps_mean_per_gt},std={overlaps_std_per_gt}")

        # 5.选择iou大于或等于iou阈值的候选框作为正样本
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        # 损失为0时，is_pos全为True

        # limit the positive sample's center in gt
        # 为了将样本点限制在Gt内部
        for gt_idx in range(num_gt):
            # 对于单目标来说，均为0 即，candidate_idxs[:, 0] += 0
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        # 所有Anchor的中心点
        ep_bboxes_cx = (
            bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        )  # 2100
        ep_bboxes_cy = (
            bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        )  # 2100
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        # 中心点距离左边
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        # 中心点距离上边
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        # 中心点距离右边
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        # 中心点距离下边
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        # pos_count*4*1==>pos_count*1
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        # 损失为0时，is_in_gts均为False，样本中心点均离gt为负值
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)  # 2100个-INF
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]  # 填充正样本的iou
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()  # 2100*1

        # 最大iou
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = (
                argmax_overlaps[max_overlaps != -INF] + 1
        )

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )
