import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pdb
import utils.utils as utils
from utils.config import cfg
import h5py
import nets.resnet as resnet
import matplotlib.pyplot as plt

def extract_feature(image_path, feature_path, num_classes, path):
    image_names = os.listdir(image_path)
    model = resnet.resnet101(num_classes=num_classes, shortcut_type='A', sample_size=cfg.Train.Image_shape[0], sample_duration=cfg.Process.new_cluster)
    model = model.cuda()
    model.zero_grad()
    model.load(path)
    with torch.no_grad():
        for image_name in image_names:
            image_list = os.listdir(os.path.join(image_path, image_name))
            for i in range(0, len(image_list), cfg.Process.new_cluster):
                if (len(image_list) - i) < cfg.Process.new_cluster:
                    data = np.empty((1, 3, len(image_list) - i, ) + tuple(cfg.Train.Image_shape))
                    for indx, j in enumerate(range(i, len(image_list))):
                        im = plt.imread(os.path.join(image_path, image_name, image_list[j]))
                        data[0, :, indx, :, :] = im.transpose(2, 0, 1)
                    model = resnet.resnet50(num_classes=num_classes, shortcut_type='A', sample_size=cfg.Train.Image_shape[0], sample_duration=len(image_list) - i)
                    model = model.cuda()
                    model.zero_grad()
                else:
                    data = np.empty((1, 3, cfg.Process.new_cluster, ) + tuple(cfg.Train.Image_shape))
                    for indx, j in enumerate(range(i, i + cfg.Process.new_cluster)):
                        im = plt.imread(os.path.join(image_path, image_name, image_list[j]))
                        data[0, :, indx, :, :] = im.transpose(2, 0, 1)
                x = model.forward(torch.tensor(data).cuda().float())
                x = x.reshape(x.shape[:3])
                if i == 0:
                    feature = x
                else:
                    feature = torch.cat((feature, x), -1)
            f = h5py.File(os.path.join(feature_path, image_name+".h5"), 'w')
            dset = f.create_dataset("feature", data = feature.cpu().numpy())
            f.close()
            print(image_name, "ok")