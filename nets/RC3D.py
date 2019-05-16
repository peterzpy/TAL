import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
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
        return inputs.view(inputs_size[0], inputs_size[2], -1)

class segment_proposal(nn.Module):

    def __init__(self, receptive_size):
        super(segment_proposal, self).__init__()
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

class RC3D(nn.Module):

    def __init__(self, num_classes, image_shape):
        super(RC3D, self).__init__()
        self.num_classes = num_classes
        self.anchor = [1, 2, 3, 4, 5, 6, 8, 11, 16]
        self.num_anchors = len(self.anchor)
        H, W = image_shape
        assert H % 16 == 0, "H must be times of 16"
        assert W % 16 == 0, "W must be times of 16"
        self.layer = [64, '_M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'fc', 'fc']
        self.backbone = self.build_backbone()
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv1d(1024, 256, 1)
        self.anchor = []
        for a in self.anchor:
            self.anchor.append(segment_proposal(a))
        #self.rpn_conv = nn.Conv3d(512, 512, (3, 3, 3), 1, padding = 1)
        #self.rpn_pool = nn.MaxPool3d((1, H/16, W/16))
        #self.rpn_cls_score = nn.Conv3d(512, 2*self.num_anchors, (1, 1, 1))
        #self.rpn_bbox_pred = nn.Conv3d(512, 2*self.num_anchors, (1, 1, 1))

    def build_backbone(self):
        layers = []
        in_channels = 3
        for l in self.layer1:
            if l == '_M':
                layers += nn.MaxPool3d((1, 2,  2), (1, 2, 2))
            elif l == 'M':
                layers += nn.MaxPool3d((2, 2, 2), (2, 2, 2))
            elif l == 'fc':
                if (in_channels != 1024):
                    layers += Flatten()
                layers += nn.Linear(in_channels, 1024)
                in_channels = 1024
            else:
                layers += nn.Conv3d(in_channels, l, (3, 3, 3), 1, padding = 1)
                layers += nn.ReLU(inplace = True)
                in_channels = l
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = x.transpose(0, 2, 1)
        x = self.conv1(x)