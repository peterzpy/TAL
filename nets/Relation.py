import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.config import cfg

def extract_position_matrix(bbox, nongt_dim):
    '''
    @params
        bbox [N, 2]
    @outputs
        position_matrix [N, nongt_dim, 2] 
    '''
    center = bbox[:, 0].float()
    length = bbox[:, 1].float()
    delta_center1, delta_center2 = torch.meshgrid(center, center)
    delta_center = (delta_center1 - delta_center2) / length
    delta_center = torch.log(torch.max(delta_center, torch.ones_like(delta_center) * 1e-3))
    delta_length1, delta_length2 = torch.meshgrid(length, length)
    delta_length = delta_length1 / delta_length2
    delta_length = torch.log(torch.max(delta_length, torch.ones_like(delta_length) * 1e-3))
    concat_list = [delta_center, delta_length]
    for idx, delta in enumerate(concat_list):
        delta = delta[:, :nongt_dim]
        concat_list[idx] = delta.reshape(delta.shape + (1, ))
    position_matrix = torch.cat(concat_list, -1)

    return position_matrix

def extract_position_embedding(position_mat, feat_dim, wave_length = 1000):
    '''
    @params
        position_mat [N, nongt_dim, 2]
    @outputs
        embedding [N, nongt_dim, feat_dim]
    '''
    feat_arange = torch.arange(feat_dim / 4).cuda()
    dim_mat = torch.pow(wave_length, feat_arange * 4. / feat_dim)
    dim_mat = dim_mat.reshape(1, 1, 1, -1)
    position_mat = (position_mat * 100.).reshape(position_mat.shape + (1, ))
    div_mat = position_mat / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    size = embedding.shape
    embedding = embedding.reshape(size[0], size[1], feat_dim)

    return embedding

class Relation(nn.Module):
    
    def __init__(self, feat_dim = 256, fc_dim = 16, group = 16, dim = (1024, 1024, 256)):
        '''
        和论文中的实现好像不太一样
        @params
            roi_feat [N, 256]
            position_embedding [N, nongt_dim, cfg.Train.embedding_feat_dim]
            fc_dim == group
            feat_dim = dim[2]
        @outputs
            output: [N, dim[2]]
        '''
        super(Relation, self).__init__()
        assert feat_dim == dim[2], "feat_dim must equal to dim[2]"
        assert fc_dim == group, "fc_dim must equal to group"
        assert dim[0] == dim[1], "dim[0] must equal to dim[1]"
        self.dim = dim
        self.group = group
        self.feat_dim = feat_dim
        self.fc_dim = fc_dim
        self.fc1 = nn.Linear(cfg.Train.embedding_feat_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.dim[0])
        self.fc3 = nn.Linear(self.feat_dim, self.dim[1])
        self.conv = nn.Conv2d(self.fc_dim * self.feat_dim, self.dim[2], 1, 1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, roi_feat, position_embedding, nongt_dim):
        dim_group = (self.dim[0] / self.group, self.dim[1] / self.group, self.dim[2] / self.group)
        nongt_roi_feat = roi_feat[:nongt_dim, :]
        position_feat1 = self.relu(self.fc1(position_embedding))
        #[N, group, nongt_dim] 论文中的 W_G
        aff_weight = position_feat1.transpose(1, 2)
        q_data = self.fc2(roi_feat)
        q_data_batch = q_data.reshape(-1, self.group, int(dim_group[0]))
        #[group, N, dim_group[0]]
        q_data_batch = q_data_batch.transpose(0, 1)
        k_data = self.fc3(nongt_roi_feat)
        k_data_batch = k_data.reshape(-1, self.group, int(dim_group[1]))
        #[group, nongt_dim, dim_group[1]]
        k_data_batch = k_data_batch.transpose(0, 1)
        v_data = nongt_roi_feat
        aff = torch.matmul(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1. / math.sqrt(float(dim_group[1]))) * aff
        #[N, group, nongt_dim] 论文中的 W_A
        aff_scale = aff_scale.transpose(0, 1)
        weighted_aff = torch.log(torch.max(aff_weight, torch.ones_like(aff_weight) * 1e-6)) + aff_scale
        aff_softmax = nn.Softmax(-1)(weighted_aff)
        #[N, group, feat_dim]
        output_t = torch.matmul(aff_softmax, v_data)
        output_t = output_t.reshape(output_t.shape[0], -1, 1, 1)
        linear_out = self.conv(output_t)
        #[N, dim[2]]
        output = linear_out.reshape(-1, self.dim[2])

        return output