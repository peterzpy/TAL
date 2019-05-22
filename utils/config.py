from easydict import EasyDict as edict
import numpy as np
import os

_C = edict()
cfg = _C

_C.Process = edict()
_C.Train = edict()
_C.Test = edict()
_C.Network = edict()

_C.Process.dilation = 10
_C.Process.Frame = 30
_C.Process.length = 64


_C.Train.anchor_size = [1, 2, 3, 4, 5, 6, 8, 11, 16]
_C.Train.learning_rate = 0.0001
_C.Train.positive_threshold = 0.7
_C.Train.negative_threshold = 0.3
_C.Train.rpn_stride = 0.25

#Relation Module
#------------------------------------
_C.Train.embedding_feat_dim = 64
_C.Network.nms_threshold = [0.4]
_C.Train.nms_regularization = 1.2
_C.Train.nms_eps = 1e-8
#------------------------------------
_C.Train.nms_threshold = 0.4
_C.Train.rpn_batch_size = 256
_C.Train.batch_size = 64
_C.Train.fg_fraction = 0.25
_C.Train.rpn_fg_fraction = 0.5
_C.Train.fg_threshold = 0.5
_C.Train.rpn_nms = 0.7
_C.Train.rpn_pre_nms = 12000
_C.Train.rpn_post_nms = 300
_C.Train.regularization = 1.2
_C.Train.cls_regularization = 1.2
_C.Train.Image_shape = (160, 160)
_C.Train.cls_gamma = 1

_C.Test.frmae = 112
_C.Test.Image_shape = (160, 160)
_C.Test.score_threshold = 0.5
