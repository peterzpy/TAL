import numpy as np

def generate_anchors(feature_size, feat_stride = 16, stride = 0.25, anchor_size = [1, 2, 3, 4, 5, 6, 8, 11, 16]):
    anchors = np.zeros((int(feature_size * len(anchor_size) * 1 / stride), 2))
    anchors[:, 0] = np.tile(np.arange(0, feature_size, stride) * feat_stride, (len(anchor_size), 1)).T.reshape(-1)
    anchors[:, 1] = np.tile(anchor_size, (int(feature_size * 1 / stride), 1)).reshape(-1)
    return anchors
