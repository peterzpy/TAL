import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, inputs):
        inputs_size = inputs.size()
        inputs = inputs.transpose(1, 2)
        return inputs.reshape(inputs_size[0], inputs_size[2], -1)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class I3Res50(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400):
        self.inplanes = 64
        super(I3Res50, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.flatten = Flatten()
        self.fc_last = nn.Linear(5*5*2048, 1024)
        #self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight,  mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i]))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc_last(x)
        
        '''
        x = self.avgpool(feature)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        '''
        return x

    def forward_multi(self, x):
        clip_preds = []
        for clip_idx in range(x.shape[1]): # B, 10, 3, 3, 32, 224, 224
            spatial_crops = []
            for crop_idx in range(x.shape[2]): 
                clip = x[:, clip_idx, crop_idx]
                clip = self.forward_single(clip)
                spatial_crops.append(clip)
            spatial_crops = torch.stack(spatial_crops, 1).mean(1) # (B, 400)
            clip_preds.append(spatial_crops)
        clip_preds = torch.stack(clip_preds, 1).mean(1) # (B, 400)
        return clip_preds

    def forward(self, inputs):

        # 5D tensor == single clip
        assert len(inputs.shape) == 5, "Input dim must be 5!"
        feature = self.forward_single(inputs)

        # 7D tensor == 3 crops/10 clips
        #elif len(inputs.shape).dim() == 7:
        #    pred = self.forward_multi(inputs)
        return feature