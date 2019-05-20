import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
from layers.generate_anchor import generate_anchors
from layers.anchor_target_layer import anchor_target_layer
from layers.object_target_layer import object_target_layer
from utils.config import cfg
from utils.utils import proposal_nms
import time
import math
import pdb
import os

class Conv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(Conv1d, self).__init__()
        if padding == 'SAME':
            pad = kernel_size + (kernel_size - 1)*(dilation - 1) - 1
            if pad % 2 == 0:
                self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = pad//2, dilation = dilation)
            else:
                self.conv = nn.Sequential(nn.ConstantPad1d((pad//2+1, pad//2), 0), nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation = dilation))
        elif padding == 'VALID':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation = dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation)
    def __call__(self, inputs):
        return self.conv(inputs)

class MaxPool1d(nn.Module):
    
    def __init__(self, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(MaxPool1d, self).__init__()
        if padding == 'SAME':
            pad = kernel_size + (kernel_size - 1)*(dilation - 1) - 1
            if pad % 2 == 0:
                self.pool = nn.MaxPool1d(kernel_size, stride, padding = pad//2, dilation = dilation)
            else:
                self.pool = nn.Sequential(nn.ConstantPad1d((pad//2+1, pad//2), 0), nn.MaxPool1d(kernel_size, stride, dilation = dilation))
        elif padding == 'VALID':
            self.pool = nn.MaxPool1d(kernel_size, stride, dilation = dilation)
        else:
            self.pool = nn.MaxPool1d(kernel_size, stride, padding = padding, dilation = dilation)
    def __call__(self, inputs):
        return self.pool(inputs)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, inputs):
        inputs_size = inputs.size()
        inputs = inputs.transpose(1, 2)
        return inputs.reshape(inputs_size[0], inputs_size[2], -1)

'''
class SegmentProposal(nn.Module):

    def __init__(self, receptive_size):
        super(SegmentProposal, self).__init__()
        self.pool = MaxPool1d(math.ceil(receptive_size/3), 1, padding = 'SAME')
        self.conv1 = Conv1d(256, 256, 3, 1, padding = 'SAME', dilation = math.ceil(receptive_size/3))
        self.conv2 = Conv1d(256, 256, 3, 1, padding = 'SAME', dilation = math.ceil(2*receptive_size/3))
        self.conv_cls = Conv1d(256, 2, 1)
        self.conv_segment = Conv1d(256, 2, 1)
        self.relu = nn.ReLU(inplace = True)

    def __call__(self, inputs):
        x = self.pool(inputs)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        cls_score = self.conv_cls(x)
        segment_pred = self.conv_segment(x)
        return cls_score, segment_pred
'''

class SegmentProposal(nn.Module):

    def __init__(self):
        super(SegmentProposal, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = Conv1d(256, 256, 3, 1, padding = 'SAME')
        self.conv_cls = Conv1d(256, int(2*9/cfg.Train.rpn_stride), 1, 1)
        self.conv_segment = Conv1d(256, int(2*9/cfg.Train.rpn_stride), 1, 1)
    
    def __call__(self, inputs):
        x = self.relu(self.conv1(inputs))
        cls_score = self.conv_cls(x)
        segment_pred = self.conv_segment(x)
        return cls_score, segment_pred

class SoiPooling(nn.Module):

    def __init__(self, spp_size = 7):
        super(SoiPooling, self).__init__()
        self.spp_size = spp_size

    def __call__(self, inputs):
        length = inputs.size()[-1]
        kernel_size = math.ceil(length / self.spp_size)
        stride = kernel_size
        padding = math.floor((1+self.spp_size)*kernel_size-1-length)
        #TODO cpu 测试
        #spp_pool = nn.Sequential(nn.ConstantPad1d((padding//2, padding - padding//2), 0), nn.Conv1d(inputs.size()[1], inputs.size()[1], kernel_size, stride))
        spp_pool = nn.Sequential(nn.ConstantPad1d((padding//2, padding - padding//2), 0), nn.MaxPool1d(kernel_size, stride)).cuda()
        spp_feature = spp_pool(inputs)
        
        return spp_feature

class RC3D(nn.Module):

    def __init__(self, num_classes, image_shape):
        super(RC3D, self).__init__()
        self.num_classes = num_classes
        self.anchor_size = [1, 2, 3, 4, 5, 6, 8, 11, 16]
        self.num_anchors = len(self.anchor_size)
        (H, W) = image_shape
        assert H % 16 == 0, "H must be times of 16"
        assert W % 16 == 0, "W must be times of 16"
        self.layer = [64, '_M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'fc', 'fc']
        self.backbone = self.build_backbone()
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv1d(1024, 256, 1)
        self.conv2 = nn.Conv1d(1024, 256, 1)
        self.soipooling = SoiPooling()
        self.fc1 = nn.Linear(256*7, 256)
        self.cls = nn.Linear(256, self.num_classes)
        self.bbox_offset = nn.Linear(256, 2)
        #self.anchors = []
        #for _ in range(self.num_anchors):
        #    self.anchors.append(SegmentProposal())
        self.segment_proposal = SegmentProposal()

    def build_backbone(self):
        layers = []
        in_channels = 3
        stride = 2
        for l in self.layer:
            if l == '_M':
                layers += [nn.MaxPool3d((1, 2,  2), (1, 2, 2))]
            elif l == 'M':
                layers += [nn.MaxPool3d((2, 2, 2), (2, 2, 2))]
                stride *= 2
            elif l == 'fc':
                if (in_channels != 1024):
                    layers += [Flatten()]
                    layers += [nn.Linear(int(in_channels * cfg.Train.Image_shape[0] * cfg.Train.Image_shape[1] / stride / stride), 1024)]
                else:
                    layers += [nn.Linear(in_channels, 1024)]
                in_channels = 1024
            else:
                layers += [nn.Conv3d(in_channels, l, (3, 3, 3), 1, padding = 1)]
                layers += [nn.ReLU(inplace = True)]
                in_channels = l
        return nn.Sequential(*layers)

    def _cls_prob(self, inputs):
        inputs_reshaped = self._reshape(inputs)
        result = nn.Softmax(-1)(inputs_reshaped)

        return result
    
    def _reshape(self, inputs):
        inputs_reshape = inputs.transpose(2, 1)
        inputs_size = inputs_reshape.size()
        inputs_reshape = inputs_reshape.view(inputs_size[0], inputs_size[1], inputs_size[2] // 2, 2)

        return inputs_reshape

    def forward(self, inputs):#gt_boxes [N,2] 要从0开始
        self.im_info = inputs.size()[-1]
        feature = self.backbone(inputs)  #[N, 1024, L/16]
        feature = feature.transpose(1, 2)
        x = self.conv1(feature)
        #self.rpn_proposal = []
        #for i in range(self.num_anchors):
        #    self.anchors[i](x)
        cls_score, proposal_offset = self.segment_proposal(x)
        #TODO cpu测试
        self.anchors = torch.tensor(generate_anchors(x.size()[-1], 16, cfg.Train.rpn_stride, self.anchor_size), dtype = torch.float32, device='cuda')
        #self.anchors = torch.tensor(generate_anchors(x.size()[-1], 16, self.anchor_size)).float()
        cls_prob = self._cls_prob(cls_score).reshape(-1, 2)
        proposal_offset_reshaped = self._reshape(proposal_offset).reshape(-1, 2)
        proposal_idx, proposal_bbox = proposal_nms(self.anchors, cls_prob, proposal_offset_reshaped)
        proposal_offset = proposal_offset_reshaped[proposal_idx]
        #proposal_prob = cls_prob[proposal_idx]
        new_proposal = torch.empty_like(proposal_bbox, dtype = torch.int32)
        new_proposal = torch.empty_like(proposal_bbox, dtype = torch.int32)
        new_proposal[:, 0] = torch.min(torch.max(torch.floor((proposal_bbox[:, 0] - proposal_bbox[:, 1]) / 16), torch.zeros_like(proposal_bbox[:, 0])), torch.ones_like(proposal_bbox[:, 1]) * (feature.size()[-1] - 1)).long()
        new_proposal[:, 1] = torch.max(torch.min(torch.ceil((proposal_bbox[:, 0] + proposal_bbox[:, 1]) / 16), torch.ones_like(proposal_bbox[:, 1]) * (feature.size()[-1] - 1)), torch.zeros_like(proposal_bbox[:, 0])).long()
        self.proposal_bbox = proposal_bbox
        for i in range(new_proposal.size()[0]):
            #if new_proposal[i, 0] > new_proposal[i, 1]:
                #pdb.set_trace()
            if i == 0:
                spp_feature = nn.AdaptiveMaxPool1d((7))(feature[:, :, new_proposal[i, 0] : new_proposal[i, 1] + 1])
                #spp_feature = self.soipooling(feature[:, :, new_proposal[i, 0] : new_proposal[i, 1]])
            else:
                spp_feature = torch.cat((spp_feature, nn.AdaptiveMaxPool1d((7))(feature[:, :, new_proposal[i, 0] : new_proposal[i, 1] + 1])), 0)
                #spp_feature = torch.cat((spp_feature, self.soipooling(feature[:, :, new_proposal[i, 0] : new_proposal[i, 1]])), 0)
        x = self.relu(self.conv2(spp_feature))
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        object_cls_score = self.cls(x)
        object_offset = self.bbox_offset(x)
        cls_score = self._reshape(cls_score).reshape(-1, 2)
        cls_score = cls_score[proposal_idx]
        self.anchors_new = self.anchors[proposal_idx]
        #pdb.set_trace()
        return cls_score, proposal_offset, object_cls_score, object_offset

    def get_loss(self, cls_score, proposal_offset, object_cls_score, object_offset, gt_boxes):#gt_boxes [N, 3] (idx, start, end) idx中类别也是从1开始
        #pdb.set_trace()
        rpn_label, rpn_bbox_offset = anchor_target_layer(gt_boxes[:, 1:], self.im_info, self.anchors_new)
        object_label, object_bbox_offset = object_target_layer(gt_boxes, self.im_info, self.proposal_bbox)
        rpn_label = rpn_label.view(-1, )
        object_label = object_label.view(-1, )
        #为分类问题增加权重
        #rpn_cls_loss
        e_index = torch.nonzero(rpn_label != -1).reshape(-1)
        e_index1 = torch.nonzero(rpn_label == 1).reshape(-1)
        e_index2 = torch.nonzero(object_label != -1).reshape(-1)
        e_index3 = torch.nonzero(object_label > 0).reshape(-1)
        cls_score = cls_score[e_index]
        rpn_label = rpn_label[e_index]
        positive_num = (rpn_label == 1).sum()
        negative_num = (rpn_label == 0).sum()
        cls_rpn_weight = torch.tensor([(positive_num + negative_num)/positive_num, (positive_num + negative_num)/negative_num]).float()
        creterion = nn.CrossEntropyLoss(weight = cls_rpn_weight.cuda())
        loss1 = creterion(cls_score, rpn_label.long())
        #rpn_bbox_loss
        proposal_offset = proposal_offset[e_index1]
        rpn_bbox_offset = rpn_bbox_offset[e_index1]
        loss2 = nn.SmoothL1Loss(reduction = 'mean')(proposal_offset, rpn_bbox_offset)
        #object_cls_loss
        cls_object_weight = torch.empty(self.num_classes).float()
        positive_num = (object_label > 0).sum()
        negative_num = (object_label == 0).sum()
        cls_object_weight[0] = (positive_num + negative_num)/negative_num
        cls_object_weight[1:] = (positive_num + negative_num)/positive_num
        creterion = nn.CrossEntropyLoss(weight = cls_object_weight.cuda())
        loss3 = creterion(object_cls_score[e_index2], object_label[e_index2].long())
        #object_bbox_loss
        object_offset = object_offset.reshape(-1, 2)
        e_index4 = e_index3 * (self.num_classes - 1) + object_label[e_index3].long() - 1
        loss4 = nn.SmoothL1Loss(reduction = 'mean')(object_offset[e_index4], object_bbox_offset[e_index3])
        if math.isnan(loss2.data) or math.isnan(loss4.data):
            print(loss1.data, loss2.data, loss3.data, loss4.data)
            pdb.set_trace()
        return loss1 + cfg.Train.regularization * loss2 + loss3 + cfg.Train.regularization * loss4, loss1, loss2, loss3, loss4

    def load(self, path):
        try:
            pretrained_dict = torch.load(path)
            print("Begin to load model {} ...".format(path))
            model_dict = self.state_dict()
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("Done!")
            del pretrained_dict
            del model_dict
        except Exception:
            print("Error! There is no model in ", os.path.join(os.path(), path))

    def save(self, path):
        print("Begin to load {} ...".format(path))
        f = open(path, 'wb')
        torch.save(self.state_dict(), f)
        f.close()
        print("Done!")