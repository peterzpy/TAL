import torch
import sys
sys.path.append("..")
from utils.config import cfg

def MyOptim(params):
    #TODO 修改优化函数
    #optimizer = torch.optim.SGD(params, cfg.Train.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = torch.optim.Adam(params, cfg.Train.learning_rate)
    return optimizer