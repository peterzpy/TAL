import sys
sys.path.append("..")
import torch
import numpy as np
import argparse
import os
import time
from nets import RC3D_simplified
from utils.config import cfg
from layers.MyOptim import MyOptim
from utils.AverageMeter import AverageMeter
import utils.utils as utils
import re
import pdb

CLASSES = ("BackGround", "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch",
            "GolfSwing", "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault", "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking")
num_classes = len(CLASSES)
name_to_id = dict(list(zip(CLASSES, range(num_classes))))
id_to_name = dict(enumerate(CLASSES))

def arg_parse():
    parser = argparse.ArgumentParser(description = "BasicNet")
    parser.add_argument("--pretrained", dest = "pretrained", type = str, default = 'False')
    parser.add_argument("--preprocess", dest = "preprocess", type = str, default = 'True')
    parser.add_argument("--image_path", dest = 'image_path', type = str, default = '/home/share2/zhangpengyi/data/ActionImage/')
    parser.add_argument("--video_path", dest = "video_path", type = str, default = '/home/share2/zhangpengyi/data/ActionVideo/')
    parser.add_argument("--video_annotation_path", dest = "video_annotation_path", type = str, default = '/home/share2/zhangpengyi/data/ActionVideoAnnotation/thumos14_val.json')
    parser.add_argument("--annotation_path", dest = 'annotation_path', type = str, default = '/home/share2/zhangpengyi/data/ActionLabel/')
    parser.add_argument("--checkpoint_path", dest = 'checkpoint_path', type = str, default = '/home/share2/zhangpengyi/data/ActionCheckpoint/')
    parser.add_argument("--iters", dest = 'iters', type = int, default = 70000)
    parser.add_argument("--frozon", dest = 'frozon', type = int, default = 10000)
    parser.add_argument("--display_per_iters", dest = "display_per_iters", type = int, default = 20)
    parser.add_argument("--snapshot_per_iters", dest = "snapshot_per_iters", type = int, default = 1000)
    parser.add_argument("--clear_per_iters", dest = "clear_per_iters", type = int, default = 1000)
    args = parser.parse_args()
    return args

def train(args):
    ckpt_path = args.checkpoint_path
    cost = AverageMeter()
    cost1 = AverageMeter()
    cost2 = AverageMeter()
    cost3 = AverageMeter()
    cost4 = AverageMeter()
    runtime = AverageMeter()
    if args.preprocess == 'False':
        utils.preprocess(args.video_path, args.image_path, args.video_annotation_path, args.annotation_path)
    try:
        names = os.listdir(ckpt_path)
        for name in names:
            out = re.findall("BaseNet_.*", name)
            if out != []:
                ckpt_path = out[0]
                break
        step = int(re.findall(r".*_(.*).ckpt", ckpt_path)[0])
        ckpt_path = os.path.join(args.checkpoint_path, ckpt_path)
    except Exception:
        step = 0
        ckpt_path = os.path.join(ckpt_path, "BaseNet.ckpt")
    model = RC3D_simplified.RC3D(num_classes, cfg.Train.Image_shape)
    #TODO cpu测试
    model = model.cuda()
    model.zero_grad()
    if step or args.pretrained == 'True':
        model.load(ckpt_path)
    train_batch = utils.Batch_Generator(name_to_id, num_classes, args.image_path, args.annotation_path)
    optimizer = MyOptim(model.parameters())
    while step < args.iters:
        optimizer.zero_grad()
        tic = time.time()
        train_data, gt_boxes = next(train_batch)
        if gt_boxes.shape[0] == 0:
            continue
        #TODO cpu测试
        data = torch.tensor(train_data, device = 'cuda', dtype = torch.float32)
        gt_boxes = torch.tensor(gt_boxes, device = 'cuda', dtype = torch.float32)
        #data = torch.tensor(train_data, dtype = torch.float32)
        #pdb.set_trace()
        cls_score, proposal_offset, object_cls_score, obejct_offset = model.forward(data)
        loss, loss1, loss2, loss3, loss4 = model.get_loss(cls_score, proposal_offset, object_cls_score, obejct_offset, gt_boxes)
        cost.update(loss)
        cost1.update(loss1)
        cost2.update(loss2)
        cost3.update(loss3)
        cost4.update(loss4)
        toc = time.time()
        runtime.update(toc-tic)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        step += 1
        if step % args.display_per_iters == 0 and step:
            print('iter: [{0}]\t' 'Loss {loss.avg:.4f}\t'
                  'Time {runtime.val:.3f} ({runtime.avg:.3f})\n'
                  'RPN:\nCls_Loss {loss1.avg:.4f}\t Bbox_Loss {loss2.avg:.4f}\nProposal:\nCls_Loss {loss3.avg:.4f}\t Bbox_Loss {loss4.avg:.4f}\n'.format(step, runtime=runtime, loss=cost, loss1=cost1, loss2=cost2, loss3=cost3, loss4=cost4))
        if step % args.snapshot_per_iters == 0 and step:
            try:
                os.remove(ckpt_path)
            except Exception:
                pass
            ckpt_path = os.path.join(args.checkpoint_path, "BaseNet_{:05d}.ckpt".format(step))
            model.save(ckpt_path)
        if step % args.clear_per_iters == 0:
            cost.reset()
            cost1.reset()
            cost2.reset()
            cost3.reset()
            cost4.reset()
            runtime.reset()

if __name__ == "__main__":
    args = arg_parse()
    print(args)
    train(args)