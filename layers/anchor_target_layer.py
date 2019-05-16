import numpy as np
import torch
import sys
sys.path.append("..")
from utils.config import cfg
from utils.utils import bbox_overlap
import math
import pdb

def test():
    gt_boxes = torch.tensor(
        [[14.0000, 19.7000, 27.3500],
        [14.0000, 29.9000, 43.4000]]).cuda()
    im_info = 160
    all_anchors = torch.tensor([
        [ 0.,  1.], [ 0.,  2.], [ 0.,  3.], [ 0.,  4.], [ 0.,  5.], [ 0.,  6.], [ 0.,  8.], [ 0., 16.], 
        [ 4.,  1.], [ 4.,  2.], [ 4.,  3.], [ 4.,  4.], [ 4.,  5.], [ 4.,  6.], [ 4.,  8.], [ 4., 11.], [ 4., 16.],
        [ 8.,  1.], [ 8.,  2.], [ 8.,  3.], [ 8.,  4.], [ 8.,  5.], [ 8.,  6.], [ 8.,  8.], [ 8., 11.], [ 8., 16.],
        [12.,  1.], [12.,  2.], [12.,  3.], [12.,  4.], [12.,  5.], [12.,  6.], [12.,  8.], [12., 11.], [12., 16.],
        [16.,  1.], [16.,  2.], [16.,  3.], [16.,  4.], [16.,  5.], [16.,  6.], [16.,  8.], [16., 11.], [16., 16.],
        [20.,  1.], [20.,  2.], [20.,  3.], [20.,  4.], [20.,  6.], [20.,  8.], [20., 11.], [20., 16.],
        [24.,  1.], [24.,  2.], [24.,  3.], [24.,  4.], [24.,  5.], [24.,  6.], [24.,  8.], [24., 11.], [24., 16.],
        [28.,  1.], [28.,  2.], [28.,  3.], [28.,  4.], [28.,  5.], [28.,  6.], [28.,  8.], [28., 11.], [28., 16.],
        [32.,  1.], [32.,  2.], [32.,  3.], [32.,  4.], [32.,  5.], [32.,  6.], [32.,  8.], [32., 11.], [32., 16.],
        [36.,  1.], [36.,  2.], [36.,  3.], [36.,  4.], [36.,  5.], [36.,  6.], [36.,  8.], [36., 11.], [36., 16.],
        [40.,  1.], [40.,  2.], [40.,  3.], [40.,  4.], [40.,  5.], [40.,  6.], [40.,  8.], [40., 11.], [40., 16.],
        [44.,  1.], [44.,  2.], [44.,  3.], [44.,  4.], [44.,  5.], [44.,  6.], [44.,  8.], [44., 11.], [44., 16.],
        [48.,  1.], [48.,  2.], [48.,  3.], [48.,  4.], [48.,  5.], [48.,  6.], [48.,  8.], [48., 11.], [48., 16.],
        [52.,  1.], [52.,  2.], [52.,  3.], [52.,  4.], [52.,  5.], [52.,  6.], [52.,  8.], [52., 11.], [52., 16.],
        [56.,  1.], [56.,  2.], [56.,  3.], [56.,  4.], [56.,  5.], [56.,  6.], [56.,  8.], [56., 11.], [56., 16.],
        [60.,  1.], [60.,  2.], [60.,  3.], [60.,  4.], [60.,  5.], [60.,  6.], [60.,  8.], [60., 11.], [60., 16.]]).cuda()
    pdb.set_trace()
    anchor_target_layer(gt_boxes, im_info, all_anchors)

def anchor_target_layer(gt_boxes, im_info, all_anchors):
    #生成标注并筛选anchor
    #cls_score [1, 2K, L/16]  all_anchors [P*K, 2]  gt_boxes [N1, 2]
    allow_border = 0
    total_anchor = all_anchors.shape[0]
    anchors = torch.zeros_like(all_anchors)
    anchors[:, 0] = torch.floor(all_anchors[:, 0] - all_anchors[:, 1] / 2)
    anchors[:, 1] = torch.ceil(all_anchors[:, 0] + all_anchors[:, 1] / 2)
    indx = torch.nonzero((anchors[:, 0] >= -allow_border) & (anchors[:, 1] <= im_info + allow_border) & (anchors[:, 0] < anchors[:, 1])).reshape(-1)
    label = torch.ones(len(indx), ).cuda() * -1  #[N, 1]
    anchors = anchors[indx, :]
    overlap = bbox_overlap(anchors, gt_boxes)
    argmax_overlap = torch.argmax(overlap, 1)
    gt_argmax_overlap = torch.argmax(overlap, 0)
    best_indx = gt_argmax_overlap
    #proposal large than positve_threshold
    positive_indx = torch.nonzero(torch.max(overlap, 1)[0] > cfg.Train.positive_threshold).reshape(-1)
    label[positive_indx] = 1
    #proposal small than negative_threshold
    negative_indx = torch.nonzero(torch.max(overlap, 1)[0] < cfg.Train.negative_threshold).reshape(-1)
    label[negative_indx] = 0
    #proposal with max overlap
    label[best_indx] = 1
    num_fg = int(cfg.Train.rpn_batch_size * cfg.Train.rpn_fg_fraction)
    fg_indx = torch.nonzero(label == 1).reshape(-1)
    if num_fg < len(fg_indx):
        rnd_positive_indx = torch.randperm(len(fg_indx))[:len(fg_indx) - num_fg]
        label[fg_indx[rnd_positive_indx]] = -1
    num_bg = int(cfg.Train.rpn_batch_size - num_fg)
    bg_indx = torch.nonzero(label == 0).reshape(-1)
    if num_bg < len(bg_indx):
        rnd_negative_indx = torch.randperm(len(bg_indx))[:len(bg_indx) - num_bg]
        label[bg_indx[rnd_negative_indx]] = -1
    bbox_offset = compute_target(anchors, gt_boxes[argmax_overlap, :])  #[N, 2]
    label = label.reshape(-1, 1)
    label = unmap(label, total_anchor, indx, -1)
    bbox_offset = unmap(bbox_offset, total_anchor, indx, 0)
    
    #负正样本比例不超过 3:1
    positive_num = (label == 1).sum()
    negative_num = (label == 0).sum()
    if positive_num * 3 < negative_num:
        idx = torch.nonzero(label.reshape(-1) == 0)
        label[idx[torch.randperm(len(idx))[:negative_num - 3 * positive_num]]] = -1
    return label, bbox_offset

def unmap(data, count, index, fill = 0):
    #将N维映射会原来的全部anchor中
    res = torch.ones(count, data.shape[-1]).cuda()*fill
    res[index, :] = data

    return res

def compute_target(rois, gt_boxes):
    assert rois.shape[0] == gt_boxes.shape[0], '输入维度不同'
    new_rois = torch.zeros_like(rois)
    new_gt = torch.zeros_like(gt_boxes)
    bbox_offset = torch.zeros_like(rois)
    new_rois[:, 0] = (rois[:, 0] + rois[:, 1]) // 2
    new_rois[:, 1] = rois[:, 1] - rois[:, 0]
    new_gt[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 1]) // 2
    new_gt[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 0]
    bbox_offset[:, 0] = (new_gt[:, 0] - new_rois[:, 0]) / new_rois[:, 1]
    bbox_offset[:, 1] =  torch.log(new_gt[:, 1] / new_rois[:, 1])
    
    return bbox_offset