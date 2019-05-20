import numpy as np
import torch
import torch.nn as nn
from utils.config import cfg
import matplotlib.pyplot as plt
import json
import random
import pdb
import time
import cv2
import os

def subscript_index(arr, idx):
    '''
    @params
        arr 任意维度
        idx 逐下标的索引
    @outputs
        result 针对下标对应的返回值
    
    examples:
        >>> x = torch.randn(2, 2, 2, 3)
            x = [[[[ 0.6956, -0.8183, -1.4719],
                   [ 0.2080,  0.6601,  0.5879]],

                   [[ 1.1300,  0.6284, -0.7488],
                   [-1.7741, -0.3267,  1.6636]]],

                   [[[ 0.0633,  0.1632,  0.2690],
                   [ 1.8097,  0.4601,  0.4023]],

                   [[-0.8845, -0.0035,  0.3874],
                   [-0.9179,  0.7707,  2.2981]]]]

        >>> y = torch.randint(3, (5, 3))
            y = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 0, 0],
                 [1, 0, 0]]
        >>> result = subscript_index(x, y)
            result = [[-0.9179,  0.7707,  2.2981],
                      [-0.9179,  0.7707,  2.2981],
                      [-0.9179,  0.7707,  2.2981],
                      [ 0.0633,  0.1632,  0.2690],
                      [ 0.0633,  0.1632,  0.2690]]
    '''

    assert torch.is_tensor(arr), "arr must be tensor"
    assert torch.is_tensor(idx), "idx must be tensor"
    num = idx.shape[0]
    size = idx.shape[1]
    assert size <= len(arr.shape), "idx don't match the arr"
    result = arr[idx.chunk(num, -1)]
    result = result.reshape((num, ) + arr.shape[size:])

    return result
    
def proposal_nms(anchors, cls_prob, proposal_offset):
    index = torch.ones((cls_prob.size()[0], ), dtype = torch.long).cuda()
    refined_proposal = torch.empty_like(anchors) #[C, L]
    refined_proposal[:, 0] = proposal_offset[:, 0] * anchors[:, 1] + anchors[:, 0]
    refined_proposal[:, 1] = anchors[:, 1] * torch.exp(proposal_offset[:, 1])
    new_proposal = torch.empty_like(anchors)
    new_proposal[:, 0] = refined_proposal[:, 0] - refined_proposal[:, 1] / 2
    new_proposal[:, 1] = refined_proposal[:, 0] + refined_proposal[:, 1] / 2
    _, sort_index = torch.sort(cls_prob[:, 0], dim = 0, descending = True)
    if cls_prob.size()[0] > cfg.Train.rpn_pre_nms:
        sort_index = sort_index[:cfg.Train.rpn_pre_nms]
    index[cfg.Train.rpn_pre_nms:] = 0
    sorted_proposal = new_proposal[sort_index, :]
    overlaps = bbox_overlap(sorted_proposal[0:1, :], sorted_proposal)
    index[1 + torch.nonzero(overlaps[0, 1:]  >= cfg.Train.rpn_nms).squeeze()] = 0
    rpn_reserve_index = torch.nonzero(index == 1).view(-1)
    if len(rpn_reserve_index) > cfg.Train.rpn_post_nms:
        index[rpn_reserve_index[cfg.Train.rpn_post_nms:]] = 0
    index = torch.nonzero(index == 1).reshape(-1)

    return index, refined_proposal[index]

def nms(proposal_bbox, object_cls_score, object_offset, num_classes, im_info):
    #proposal_bbox [N, 2]
    #object_cls_score [N, C+1]
    #object_offset [N, 2 * C]
    pdb.set_trace()
    object_offset = object_offset.reshape(object_offset.shape[0], -1, 2)
    cls_prob, cls_prob_idx = torch.max(nn.Softmax(dim = -1)(object_cls_score), -1)
    allow_border = 0
    object_bbox = dict()
    object_bbox['cls'] = []
    object_bbox['score'] = []
    object_bbox['bbox'] = []
    for i in range(1, num_classes):
        temp_prob = cls_prob[cls_prob_idx == i]
        if temp_prob.shape[0] == 0:
            continue
        refined_proposal = torch.empty_like(proposal_bbox[cls_prob_idx == i])
        new_proposal = torch.empty_like(proposal_bbox[cls_prob_idx == i])
        refined_proposal[:, 0] = proposal_bbox[cls_prob_idx == i, 1] * object_offset[cls_prob_idx == i, i - 1, 0] + proposal_bbox[cls_prob_idx == i, 0]
        refined_proposal[:, 1] = proposal_bbox[cls_prob_idx == i, 1] * torch.exp(object_offset[cls_prob_idx == i, i - 1,  1])
        new_proposal[:, 0] = refined_proposal[:, 0] - refined_proposal[:, 1]/2
        new_proposal[:, 1] = refined_proposal[:, 0] + refined_proposal[:, 0]/2
        indx = torch.nonzero((new_proposal[:, 0] >= allow_border) & (new_proposal[:, 1] <= im_info + allow_border) & (new_proposal[:, 0] < new_proposal[:, 1])).reshape(-1)
        prob, temp_idx = torch.sort(temp_prob[indx], descending = True)
        new_proposal = new_proposal[indx, :][temp_idx, :]
        overlaps = bbox_overlap(new_proposal, new_proposal)
        label = torch.ones(len(temp_idx))
        for j in range(0, len(temp_idx)):
            if label[j] == 0:
                continue
            if prob[j] < cfg.Test.score_threshold:
                continue
            object_bbox['cls'].append(i)
            object_bbox['score'].append(prob[j])
            object_bbox['bbox'].append(new_proposal[j, :])
            for k in range(j + 1, len(temp_idx)):
                if overlaps[j, k] >= cfg.Train.nms_threshold:
                    label[k] = 0
        if object_bbox['cls'] == []:
            pdb.set_trace()
        return object_bbox

def bbox_overlap(boxes1, boxes2):
    #input [N1, 2]  [N2, 2]
    #output [N1, N2]
    output1, output2 = torch.meshgrid(boxes1[:, 0], boxes2[:, 0])
    output3, output4 = torch.meshgrid(boxes1[:, 1], boxes2[:, 1])
    intersection = torch.max(torch.min(output4, output3) - torch.max(output2, output1) + torch.ones_like(output1), torch.zeros(boxes1.shape[0], boxes2.shape[0]).cuda())
    overlap = intersection / (output3 - output1 + output4 - output2 + 2 * torch.ones_like(output1) - intersection)
    
    return overlap

'''def preprocess(video_path, image_path, video_annotation_path, annotation_path):
    print("Begin to process the video!")
    video_names = os.listdir(video_path)
    f = open(video_annotation_path)
    frame = cfg.Process.Frame
    annotation = json.load(f) 
    for video_name in video_names:
        temp_name = os.path.join(image_path, video_name.split('.')[0])
        os.mkdir(temp_name)
        last_frame = 0
        with open(os.path.join(annotation_path, video_name.split('.')[0]+".txt"), "w") as annotation_label:
            for label in annotation[video_name.split('.')[0]]['actions']:
                annotation_label.write("{} {} {}\n".format(str(label[0]), str(label[1]*frame/cfg.Process.dilation), str(label[2]*frame/cfg.Process.dilation)))
                last_frame = label[2]*frame/cfg.Process.dilation
        try:
            cap = cv2.VideoCapture(os.path.join(video_path, video_name))
            counter = 0
            while(cap.isOpened()):
                _, frame = cap.read()
                if counter % cfg.Process.dilation - last_frame > 1:
                    break
                if counter % cfg.Process.dilation == 0:
                    frame = cv2.resize(frame, cfg.Train.Image_shape)
                    cv2.imwrite(os.path.join(temp_name, "{:04d}.jpg".format(counter%20+1)), frame)
                counter += 1
        except Exception:
            print("Video process error at ", os.path.join(video_path, video_name))
    f.close()
    print("Done!")'''

def preprocess(video_path, image_path, video_annotation_path, annotation_path):
    #重叠程度为0.5的滑窗式分割，除了极小或极大的物体外都会有较好的效果
    print("Begin to process the video!")
    video_names = os.listdir(video_path)
    f = open(video_annotation_path)
    annotation = json.load(f)
    for video_name in video_names:
        temp_name = os.path.join(image_path, video_name.split('.')[0])
        try:
            annotation_action = annotation[video_name.split('.')[0]]['actions']
        except Exception:
            continue
        index = 0
        index_max = len(annotation_action) - 1
        #annotation_label.write("{} {} {}\n".format(str(label[0]), str(label[1]*frame/cfg.Process.dilation), str(label[2]*frame/cfg.Process.dilation)))
        try:
            cap = cv2.VideoCapture(os.path.join(video_path, video_name))
            num = int(cap.get(7))
            counter = 0
            total = -1
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if counter % (cfg.Process.dilation * cfg.Process.length) == 0:
                    #pdb.set_trace()
                    if index > index_max:
                        break
                    if num - counter < cfg.Process.dilation * cfg.Process.length:
                        break
                    total += 1
                    cur_name = temp_name + "_" + "{:06d}".format(total)
                    if (annotation_action[index][1] + annotation_action[index][2])/2*cfg.Process.Frame < (counter + cfg.Process.dilation * cfg.Process.length):
                        os.mkdir(cur_name)
                        fw = open(os.path.join(annotation_path, video_name.split('.')[0] + "_{:06d}".format(total) + ".txt"), 'w')
                        while(index <= index_max):
                            if (annotation_action[index][1] + annotation_action[index][2])/2 * cfg.Process.Frame  <= (counter + cfg.Process.dilation * cfg.Process.length):
                                if annotation_action[index][1] * cfg.Process.Frame < counter and annotation_action[index][2] * cfg.Process.Frame > (counter + cfg.Process.dilation * cfg.Process.length):
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], 0, cfg.Process.length))
                                elif annotation_action[index][1] * cfg.Process.Frame < counter:
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], 0, (annotation_action[index][2] * cfg.Process.Frame - counter) / cfg.Process.dilation))
                                elif annotation_action[index][2] * cfg.Process.Frame > (counter + cfg.Process.dilation * cfg.Process.length):
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], (annotation_action[index][1] * cfg.Process.Frame - counter) / cfg.Process.dilation, cfg.Process.length))
                                else:
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], (annotation_action[index][1] * cfg.Process.Frame - counter) / cfg.Process.dilation, (annotation_action[index][2] * cfg.Process.Frame - counter) / cfg.Process.dilation))
                                index += 1
                            else:
                                break
                        fw.close()
                        print(os.path.join(annotation_path, video_name.split('.')[0] + "_{:06d}".format(total)) + " Done!")
                if counter % cfg.Process.dilation == 0:
                    #pdb.set_trace()
                    frame = cv2.resize(frame, tuple(cfg.Train.Image_shape))
                    cv2.imwrite(os.path.join(cur_name, "{:04d}.jpg".format(int((counter / cfg.Process.dilation) % cfg.Process.length + 1))), frame)
                counter += 1
            base = cfg.Process.length * cfg.Process.dilation // 2
            counter = 0
            index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, base)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if counter % (cfg.Process.dilation * cfg.Process.length) == 0:
                    #pdb.set_trace()
                    if index > index_max:
                        break
                    if num - counter - base < cfg.Process.dilation * cfg.Process.length:
                        break
                    total += 1
                    cur_name = temp_name + "_" + "{:06d}".format(total)
                    #可能会有在最前面半段的标注，需要跳过
                    while index <= index_max and (annotation_action[index][1] + annotation_action[index][2])/2*cfg.Process.Frame < base + counter:
                        index += 1
                    if index > index_max:
                        break
                    if (annotation_action[index][1] + annotation_action[index][2])/2*cfg.Process.Frame < (base + counter + cfg.Process.dilation * cfg.Process.length):
                        os.mkdir(cur_name)
                        fw = open(os.path.join(annotation_path, video_name.split('.')[0] + "_{:06d}".format(total) + ".txt"), 'w')
                        while(index <= index_max):
                            if (annotation_action[index][1] + annotation_action[index][2])/2 * cfg.Process.Frame  <= (base + counter + cfg.Process.dilation * cfg.Process.length):
                                if annotation_action[index][1] * cfg.Process.Frame < base + counter and annotation_action[index][2] * cfg.Process.Frame > (base + counter + cfg.Process.dilation * cfg.Process.length):
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], 0, cfg.Process.length))
                                elif annotation_action[index][1] * cfg.Process.Frame < base + counter:
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], 0, (annotation_action[index][2] * cfg.Process.Frame - counter - base) / cfg.Process.dilation))
                                elif annotation_action[index][2] * cfg.Process.Frame > (base + counter + cfg.Process.dilation * cfg.Process.length):
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], (annotation_action[index][1] * cfg.Process.Frame - counter - base) / cfg.Process.dilation, cfg.Process.length))
                                else:
                                    fw.write("{:d} {:f} {:f}\n".format(annotation_action[index][0], (annotation_action[index][1] * cfg.Process.Frame - counter - base) / cfg.Process.dilation, (annotation_action[index][2] * cfg.Process.Frame - counter - base) / cfg.Process.dilation))
                                index += 1
                            else:
                                break
                        fw.close()
                        print(os.path.join(annotation_path, video_name.split('.')[0] + "_{:06d}".format(total)) + " Done!")
                if counter % cfg.Process.dilation == 0:
                    frame = cv2.resize(frame, tuple(cfg.Train.Image_shape))
                    cv2.imwrite(os.path.join(cur_name, "{:04d}.jpg".format(int((counter / cfg.Process.dilation) % cfg.Process.length + 1))), frame)
                counter += 1
        except Exception:
            print("Error while processing ", os.path.join(video_path, video_name))
    f.close()
    print("Done!")


def Batch_Generator(name_to_id, num_classes, image_path = "/home/share2/zhangpengyi/data/ActionImage/", annotation_path = "/home/share2/zhangpengyi/data/ActionLabel/", mode = 'train'):
    image_names = os.listdir(image_path)
    while True:
        random.shuffle(image_names)
        for image_name in image_names:
            image_list = os.listdir(os.path.join(image_path, image_name))
            frame_num = len(image_list)
            data = np.empty((1, 3, frame_num, ) + tuple(cfg.Train.Image_shape))
            for indx, image in enumerate(image_list):
                im = plt.imread(os.path.join(image_path, image_name, image))
                data[0, :, indx, :, :] = im.reshape(3, im.shape[0], im.shape[1])
            if mode == 'test':
                yield data
            else:
                with open(os.path.join(annotation_path, image_name+".txt")) as f:
                    lines = f.readlines()
                    label = np.empty((len(lines), 3))
                    for i, line in enumerate(lines):
                        line = line.split(" ")
                        label[i, 0] = int(line[0])
                        label[i, 1] = float(line[1])
                        label[i, 2] = float(line[2].strip())
                yield data, label