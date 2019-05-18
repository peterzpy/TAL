import numpy as np
import math
import torch
import pdb
import sys
sys.path.append("..")
from utils.config import cfg
from utils.utils import bbox_overlap

def test():
    im_info = 160
    gt_boxes = torch.tensor([[19.,  0., 64.]]).cuda()
    proposal_bbox = torch.tensor([
        [-5.7831e-05,  1.0113e+00], [-1.8957e-02,  1.9861e+00], [ 2.7177e-01,  2.8964e+00], [ 2.6759e-01,  3.9716e+00], [-1.3653e-01,  5.2097e+00], [ 1.0213e-01,  6.3611e+00],
        [ 5.6039e-01,  7.6222e+00], [ 5.6084e-01,  1.1174e+01], [ 8.3741e-01,  1.5592e+01], [ 3.9848e+00,  1.0435e+00], [ 4.0757e+00,  1.8529e+00], [ 3.7242e+00,  2.8335e+00], 
        [ 4.1367e+00,  3.7668e+00], [ 3.6936e+00,  4.6969e+00], [ 4.3918e+00,  5.5896e+00], [ 3.8806e+00,  7.6149e+00], [ 4.1164e+00,  1.1492e+01], [ 3.9516e+00,  1.7029e+01], 
        [ 7.9714e+00,  9.6806e-01], [ 8.0820e+00,  1.8853e+00], [ 7.8517e+00,  3.0060e+00], [ 8.2491e+00,  4.3500e+00], [ 8.1391e+00,  5.1494e+00], [ 8.1190e+00,  6.2049e+00], 
        [ 8.6227e+00,  7.6029e+00], [ 7.4321e+00,  1.1501e+01], [ 9.2860e+00,  1.4120e+01], [ 1.1966e+01,  1.0173e+00], [ 1.1982e+01,  2.0391e+00], [ 1.2110e+01,  3.2316e+00],
        [ 1.1744e+01,  3.9008e+00], [ 1.1827e+01,  4.8629e+00], [ 1.1998e+01,  5.6788e+00], [ 1.2330e+01,  7.2793e+00], [ 1.2313e+01,  1.2231e+01], [ 1.2535e+01,  1.5216e+01], 
        [ 1.6046e+01,  9.7803e-01], [ 1.5958e+01,  2.0274e+00], [ 1.6284e+01,  2.8597e+00], [ 1.6119e+01,  3.8923e+00], [ 1.5563e+01,  5.0031e+00], [ 1.6239e+01,  6.6190e+00],
        [ 1.6499e+01,  7.3508e+00], [ 1.6148e+01,  1.1797e+01], [ 1.5985e+01,  1.6930e+01], [ 2.0035e+01,  1.0292e+00], [ 2.0044e+01,  1.9747e+00], [ 1.9797e+01,  2.8900e+00], 
        [ 2.0274e+01,  3.5779e+00], [ 1.9881e+01,  4.5341e+00], [ 2.0018e+01,  5.6488e+00], [ 1.9559e+01,  8.1731e+00], [ 2.0474e+01,  1.1461e+01], [ 1.9018e+01,  1.6247e+01], 
        [ 2.4039e+01,  9.7798e-01], [ 2.3955e+01,  1.8940e+00], [ 2.3815e+01,  3.1018e+00], [ 2.4180e+01,  4.1024e+00], [ 2.4132e+01,  5.2519e+00], [ 2.4227e+01,  6.3439e+00], 
        [ 2.4537e+01,  7.5376e+00], [ 2.2729e+01,  1.2372e+01], [ 2.6419e+01,  1.4334e+01], [ 2.8041e+01,  9.3973e-01], [ 2.8056e+01,  1.9754e+00], [ 2.7863e+01,  3.0939e+00], 
        [ 2.7860e+01,  3.8634e+00], [ 2.7956e+01,  5.0906e+00], [ 2.8352e+01,  6.0327e+00], [ 2.8887e+01,  7.0210e+00], [ 2.8025e+01,  1.2008e+01], [ 2.7628e+01,  1.5344e+01], 
        [ 3.1985e+01,  9.7347e-01], [ 3.2015e+01,  1.9804e+00], [ 3.2373e+01,  2.9114e+00], [ 3.2121e+01,  3.8984e+00], [ 3.1708e+01,  4.9032e+00], [ 3.1978e+01,  6.3687e+00], 
        [ 3.2320e+01,  7.4406e+00], [ 3.1948e+01,  1.2195e+01], [ 3.2022e+01,  1.6988e+01], [ 3.6027e+01,  9.9259e-01], [ 3.6051e+01,  1.9555e+00], [ 3.5816e+01,  2.7691e+00],
        [ 3.6111e+01,  3.6524e+00], [ 3.5726e+01,  4.6385e+00], [ 3.6415e+01,  5.3730e+00], [ 3.5618e+01,  8.2060e+00], [ 3.6134e+01,  1.1812e+01], [ 3.5492e+01,  1.6339e+01], 
        [ 3.9999e+01,  1.0154e+00], [ 3.9979e+01,  1.9347e+00], [ 3.9786e+01,  2.9463e+00], [ 4.0214e+01,  4.2418e+00], [ 4.0033e+01,  5.3421e+00], [ 4.0569e+01,  6.2339e+00], 
        [ 4.0510e+01,  7.5624e+00], [ 3.9481e+01,  1.1713e+01], [ 4.0619e+01,  1.3957e+01], [ 4.4002e+01,  1.0077e+00], [ 4.3924e+01,  1.8965e+00], [ 4.3941e+01,  3.0673e+00],
        [ 4.3874e+01,  3.7916e+00], [ 4.4024e+01,  4.9363e+00], [ 4.4233e+01,  5.9265e+00], [ 4.4343e+01,  7.4981e+00], [ 4.4068e+01,  1.2027e+01], [ 4.3702e+01,  1.5223e+01], 
        [ 4.8008e+01,  1.0000e+00], [ 4.7950e+01,  2.0303e+00], [ 4.8248e+01,  2.8915e+00], [ 4.8036e+01,  3.9469e+00], [ 4.7809e+01,  5.3867e+00], [ 4.8093e+01,  6.5194e+00],
        [ 4.8127e+01,  7.4367e+00], [ 4.8196e+01,  1.1341e+01], [ 4.7691e+01,  1.5981e+01], [ 5.2002e+01,  1.0416e+00], [ 5.2047e+01,  1.9342e+00], [ 5.1951e+01,  2.9531e+00], 
        [ 5.2133e+01,  3.7796e+00], [ 5.1946e+01,  4.4709e+00], [ 5.2381e+01,  5.5743e+00], [ 5.1684e+01,  8.1863e+00], [ 5.1773e+01,  1.1669e+01], [ 5.1207e+01,  1.6796e+01], 
        [ 5.5998e+01,  9.7327e-01], [ 5.5971e+01,  1.8290e+00], [ 5.5840e+01,  3.0070e+00], [ 5.6182e+01,  3.9850e+00], [ 5.6077e+01,  5.3113e+00], [ 5.6183e+01,  6.2543e+00], 
        [ 5.6692e+01,  7.5728e+00], [ 5.4824e+01,  1.2618e+01], [ 5.8737e+01,  1.4282e+01], [ 6.0019e+01,  1.0201e+00], [ 6.0077e+01,  2.0333e+00], [ 5.9774e+01,  3.1963e+00], 
        [ 5.9914e+01,  3.8171e+00], [ 5.9657e+01,  5.1209e+00], [ 6.0184e+01,  5.9435e+00], [ 6.0463e+01,  7.5302e+00], [ 6.0093e+01,  1.1476e+01], [ 6.0572e+01,  1.5328e+01]]).cuda()
    pdb.set_trace()
    object_target_layer(gt_boxes, im_info, proposal_bbox)
    
def object_target_layer(gt_boxes, im_info, proposal_bbox):
    total_bbox = proposal_bbox.shape[0]
    new_proposal = torch.empty_like(proposal_bbox)
    new_proposal[:, 0] = proposal_bbox[:, 0] - proposal_bbox[:, 1] / 2
    new_proposal[:, 1] = proposal_bbox[:, 0] + proposal_bbox[:, 1] / 2
    allow_border = 0
    indx = torch.nonzero((new_proposal[:, 0] >= allow_border) & (new_proposal[:, 1] <= im_info + allow_border) & (new_proposal[:, 0] < new_proposal[:, 1])).reshape(-1)
    label = torch.zeros(len(indx), 1).cuda()
    new_proposal = new_proposal[indx, :]
    overlap = bbox_overlap(new_proposal, gt_boxes[:, 1:])
    best_index = torch.argmax(overlap, 0)
    max_overlap, argmax_overlap = torch.max(overlap, 1)
    #这里不对正负样本的比例做要求
    idx = torch.nonzero(max_overlap > cfg.Train.fg_threshold).reshape(-1)
    label[idx, 0] = gt_boxes[argmax_overlap[idx], 0]
    label[best_index, 0] = gt_boxes[:, 0]
    bbox_offset = compute_target(new_proposal[idx, :], gt_boxes[argmax_overlap[idx], 1:])
    label = unmap(label, total_bbox, indx, 0)
    bbox_offset = unmap(bbox_offset, total_bbox, indx[idx], 0)
    
    #负正样本比例不超过 3:1
    positive_num = (label != 0).sum()
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
    new_rois[:, 0] = (rois[:, 0] + rois[:, 1]) / 2
    new_rois[:, 1] = rois[:, 1] - rois[:, 0] + 1
    new_gt[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 1]) / 2
    new_gt[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 0] + 1
    bbox_offset[:, 0] = (new_gt[:, 0] - new_rois[:, 0]) / new_rois[:, 1]
    bbox_offset[:, 1] =  torch.log(new_gt[:, 1] / new_rois[:, 1])
    
    return bbox_offset