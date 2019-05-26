#!/usr/bin/python3
import sys
sys.path.append("..")
import torch
import numpy as np
import argparse
import os
import time
import re
import nets.RC3D_simplified as RC3D_simplified
from utils.AverageMeter import AverageMeter
from utils.config import cfg
import utils.utils as utils

CLASSES = ("BackGround", "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch",
            "GolfSwing", "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault", "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking")
num_classes = len(CLASSES)
name_to_id = dict(list(zip(CLASSES, range(num_classes))))
id_to_name = dict(enumerate(CLASSES))

def arg_parse():
    parser = argparse.ArgumentParser(description = "BasicNet")
    parser.add_argument("--feature_path", dest = 'feature_path', type = str, default = '/home/share2/zhangpengyi/data/train_feature/')
    parser.add_argument("--image_path", dest = 'image_path', type = str, default = '/home/share2/zhangpengyi/data/ActionImage/')
    parser.add_argument("--annotation_path", dest = 'annotation_path', type = str, default = '/home/share2/zhangpengyi/data/ActionLabel/')
    parser.add_argument("--checkpoint_path", dest = 'checkpoint_path', type = str, default = '/home/share2/zhangpengyi/data/ActionCheckpoint/')
    args = parser.parse_args()
    return args

def test(args):
    runtime = AverageMeter()
    ckpt_path = args.checkpoint_path
    try:
        names = os.listdir(ckpt_path)
        for name in names:
            out = re.findall("BaseNet_.*", name)
            if out != []:
                ckpt_path = out[0]
                break
        ckpt_path = os.path.join(args.checkpoint_path, ckpt_path)
    except Exception:
        print("There is no checkpoint in ", args.checkpoint_path)
        exit
    model = RC3D_simplified.RC3D(num_classes, cfg.Test.Image_shape, args.feature_path)
    model = model.cuda()
    model.zero_grad()
    model.load(ckpt_path)
    test_batch = utils.new_Batch_Generator(name_to_id, num_classes, args.image_path, args.annotation_path)
    tic = time.time()
    data, gt = next(test_batch)
    with torch.no_grad():
        print(gt)
        _, _, object_cls_score, obejct_offset = model.forward(data)
        bbox = utils.nms(model.proposal_bbox, object_cls_score, obejct_offset, model.num_classes, model.im_info)
        toc = time.time()
        torch.cuda.empty_cache()
        runtime.update(toc-tic)
        print('Time {runtime.val:.3f} ({runtime.avg:.3f})\t'.format(runtime=runtime))
        for _cls, score, proposal in zip(bbox['cls'], bbox['score'], bbox['bbox']):
            print("class:{:}({:})\t   score:{:.6f}\t   start:{:.2f}\t  end:{:.2f}\t".format(id_to_name[int(_cls[0])], _cls[0], score[0], proposal[0, 0], proposal[0, 1]))
   
            
if __name__ == "__main__":
    args = arg_parse()
    print(args)
    test(args)