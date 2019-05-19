import sys
sys.path.append("..")
import torch
from utils.utils import bbox_overlap

def nms_multi_target(bbox, gt_bbox, score, threshold):
    '''
    @params
        bbox [N, num_fg, 2]
        gt_bbox [K, 3]
        score [N, num_fg]
    @output
        nms_target [N, num_fg]
    '''
    num_boxes = bbox.shape[0]
    num_fg = bbox.shape[1]
    output_list = []
    for cls_idx in range(num_fg):
        gt_bbox_idx = torch.nonzero(gt_bbox[:, 0] == cls_idx + 1).reshape(-1)
        num_valid_gt_bbox = len(gt_bbox_idx)
        label = torch.zeros(num_boxes, 1).cuda()
        if num_valid_gt_bbox == 0:
            output_list.append(label)
        else:
            #[N, 2]
            bbox_temp = bbox[:, cls_idx, :]
            #[N, 1]
            score_temp = score[:, cls_idx : cls_idx + 1]
            #[V, 2]
            gt_bbox_temp = gt_bbox[gt_bbox_idx, 1:]
            overlaps = bbox_overlap(bbox_temp, gt_bbox_temp[:, :2])
            #[N, V]
            overlap_mask = overlaps > threshold[0]
            score_temp = score_temp.repeat(1, num_valid_gt_bbox)
            score_temp *= overlap_mask.float()
            eye_matrix = torch.eye(num_valid_gt_bbox).cuda()
            max_overlap_idx = torch.argmax(score_temp, 1).reshape(-1)
            #[N, K]
            score_temp = score_temp * eye_matrix[max_overlap_idx]
            score_max, score_idx = torch.max(score_temp, 0)
            max_idx = torch.nonzero(score_max).reshape(-1)
            label[score_idx[max_idx]] = 1
            output_list.append(label)
    nms_target = torch.cat(output_list, -1)
    
    return nms_target




