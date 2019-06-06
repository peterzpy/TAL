#!/usr/bin/python3
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import re
import json
import pdb
import nets.RC3D_resnet as RC3D_resnet
from utils.config import cfg
import numpy as np
import json
import utils.utils as utils

CLASSES = ("BackGround", "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch",
            "GolfSwing", "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault", "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking")
num_classes = len(CLASSES)
name_to_id = dict(list(zip(CLASSES, range(num_classes))))
id_to_name = dict(enumerate(CLASSES))

def arg_parse():
    parser = argparse.ArgumentParser(description = "ResNet")
    parser.add_argument("--feature_path", dest = 'feature_path', type = str, default = '/home/share2/zhangpengyi/data/test_feature/')
    parser.add_argument("--preprocessed", dest = 'preprocessed', type = str, default = 'True')
    parser.add_argument("--image_path", dest = 'image_path', type = str, default = '/home/share2/zhangpengyi/data/ActionTestImage/')
    parser.add_argument("--annotation_path", dest = 'annotation_path', type = str, default = '/home/share2/zhangpengyi/data/ActionTestLabel/')
    parser.add_argument("--checkpoint_path", dest = 'checkpoint_path', type = str, default = '/home/share2/zhangpengyi/data/ActionCheckpoint/')
    parser.add_argument("--json_path", dest = 'json_path', type = str, default = '../Annotation_new/')
    parser.add_argument("--video_path", dest = 'video_path', type = str, default = '/home/share2/zhangpengyi/data/ActionVideo/')
    parser.add_argument("--video_annotation_path", dest = 'video_annotation_path', type = str, default = '/home/share2/zhangpengyi/data/ActionTestVideoAnnotation/thumos14_val.json')
    parser.add_argument("--tiou", dest = 'tiou', type = float, default = 0.5)
    args = parser.parse_args()
    return args

def generate_det(args):
    ckpt_path = args.checkpoint_path
    try:
        names = os.listdir(ckpt_path)
        for name in names:
            out = re.findall("ResNet_.*", name)
            if out != []:
                ckpt_path = out[0]
                break
        ckpt_path = os.path.join(args.checkpoint_path, ckpt_path)
    except Exception:
        print("There is no checkpoint in ", args.checkpoint)
        exit
    model = RC3D_resnet.RC3D(num_classes, cfg.Test.Image_shape, args.feature_path)
    model = model.cuda()
    model.zero_grad()
    model.load(ckpt_path)
    test_batch = utils.new_Batch_Generator(name_to_id, num_classes, args.image_path, args.annotation_path, 'test')
    fp = []
    det = []
    for i in range(1, num_classes):
        f = open(os.path.join(args.json_path, "detection_{}.json".format(str(i))), 'w')
        fp.append(f)
        det.append({})
        det[i-1]['object'] = []
    try:
        while True:
            with torch.no_grad():
                data, gt = next(test_batch)
                _, _, object_cls_score, object_offset = model.forward(data)
                #bbox 是按照score降序排列的
                bbox = utils.nms(model.proposal_bbox, object_cls_score, object_offset, model.num_classes, model.im_info)
                if bbox is None:
                    continue
                #pdb.set_trace()
                for _cls, score, proposal in zip(bbox['cls'], bbox['score'], bbox['bbox']):
                    if proposal[:, 0] == proposal[:, 1]:
                        continue
                    temp_dict = {}
                    temp_dict['file_name'] = data
                    temp_dict['start'] = float(proposal[:, 0])
                    temp_dict['end'] = float(proposal[:, 1])
                    temp_dict['score'] = float(score)
                    det[int(_cls[0]) - 1]['object'].append(temp_dict)
                torch.cuda.empty_cache()
    except StopIteration:
        for i in range(num_classes-1):
            json.dump(det[i], fp[i])
            fp[i].close()
    print("generate_gt Done!")

def eval_ap(rec, prec):
    rec.insert(0, 0.)
    rec.append(1.)
    prec.insert(0, 0.)
    prec.append(0.)
    mrec = np.array(rec)
    mpre = np.array(prec)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    ris = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0
    for ri in ris:
        ap += (mrec[ri + 1] - mrec[ri]) * mpre[ri + 1]
    return ap, mrec, mpre

def eval_mAP(args):
    AP = 0
    for cls_idx in range(1, num_classes):
        gt_json = open(os.path.join(args.json_path, "GT_{}.json".format(str(cls_idx))))
        det_json = open(os.path.join(args.json_path, "detection_{}.json".format(str(cls_idx))))
        gt = json.load(gt_json)
        det = json.load(det_json)
        fp = []
        tp = []
        score = []
        for idx in range(len(det['object'])):
            score.append(det['object'][idx]['score'])
        score = np.array(score)
        sort_idx = np.argsort(score).reshape(-1)
        sort_idx = sort_idx[::-1]
        temp_det = det['object']
        temp_gt = gt['object']
        for idx in range(len(temp_det)):
            ovm = -1
            result = -1
            for gt_idx in range(len(temp_gt)):
                if temp_gt[gt_idx]['file_name'] != temp_det[sort_idx[idx]]['file_name']+'.txt':
                    continue
                intersection = max(min(temp_det[sort_idx[idx]]['end'], temp_gt[gt_idx]['end']) - max(temp_det[sort_idx[idx]]['start'], temp_gt[gt_idx]['start']) + 1, 0)
                overlap = intersection / (temp_det[sort_idx[idx]]['end'] - temp_det[sort_idx[idx]]['start'] + temp_gt[gt_idx]['end'] - temp_gt[gt_idx]['start'] + 2 - intersection)
                if overlap > ovm:
                    ovm = overlap
                    result = gt_idx
            if ovm >= args.tiou:
                if temp_gt[result]['use'] == False:
                    temp_gt[result]['use'] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                fp.append(1)
                tp.append(0)
        total = 0
        for idx, val in enumerate(fp):
            fp[idx] += total
            total += val
        total = 0
        for idx, val in enumerate(tp):
            tp[idx] += total
            total += val
        #pdb.set_trace()
        rec = tp.copy()
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt['num']
        prec = tp.copy()
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        ap, _, _ = eval_ap(rec, prec)
        AP += ap
        print(cls_idx, id_to_name[cls_idx], ap, "Done")
    mAP = AP/(num_classes-1)
    print(mAP)
    return mAP

if __name__ == '__main__':
    args = arg_parse()
    print(args)
    if args.preprocessed == 'False':
        utils.new_preprocess(args.video_path, args.image_path, args.video_annotation_path, args.annotation_path)
    utils.generate_gt(args.annotation_path, args.json_path, num_classes-1)
    generate_det(args)
    eval_mAP(args)
