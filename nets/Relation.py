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

def extract_multi_position_matrix(bbox):
    '''
    提取多类位置矩阵
    @params
        bbox [N, num_fg, 2]
    @output
        position_matrix [num_fg, N, N, 2]
    '''
    bbox = bbox.transpose(0, 1)
    center = bbox[:, :, 0:1].float()
    length = bbox[:, :, 1:2].float()
    center_new = center.transpose(1, 2)
    length_new = length.transpose(1, 2)
    delta_center = center - center_new
    delta_center = torch.log(torch.max(delta_center, torch.ones_like(delta_center) * 1e-3))
    delta_length = length / length_new
    delta_length = torch.log(torch.max(delta_length, torch.ones_like(delta_length) * 1e-3))
    concat_list = [delta_center, delta_length]
    for idx, delta in enumerate(concat_list):
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

def extract_pairwise_multi_position_embedding(position_mat, feat_dim, wave_length = 1000):
    '''
    多类位置嵌入
    @params
        position_mat [num_fg, N, N, 2]
    @output
        embedding [num_fg, N, N, feat_dim]
    '''
    feat_arange = torch.arange(feat_dim / 4).cuda()
    dim_mat = torch.pow(wave_length, feat_arange * 4. / feat_dim)
    dim_mat = dim_mat.reshape(1, 1, 1, 1, -1)
    position_mat = (position_mat * 100.).reshape(position_mat.shape + (1, ))
    div_mat = position_mat / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    size = embedding.shape
    embedding = embedding.reshape(size[0], size[1], size[2], feat_dim)

    return embedding

def extract_rank_embedding(rank_dim, feat_dim = 256, wave_length = 1000):
    '''
    @params
        rank_dim 最多选取多少个roi，我设定的是num_rois(全部选取)
        feat_dim 256
    @return
        embedding [rank_dim, feat_dim]
    '''
    rank_range = torch.arange(rank_dim).cuda()
    feat_range = torch.arange(feat_dim / 2).cuda()
    dim_mat = torch.pow(wave_length, 2. / feat_dim * feat_range)
    dim_mat = dim_mat.reshape(1, -1)
    rank_mat = rank_range.reshape(-1, 1).float()
    div_mat = rank_mat / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

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

class NMSRelation(nn.Module):
    
    def __init__(self, fc_dim = (64, 16), group = 16, feat_dim = 128, dim = (1024, 1024, 128)):
        '''
        我的 num_rois 使用的是全部的roi区域
        @params
            roi_feat [N, num_fg, feat_dim]
            position_mat [num_fg, N, N, 2]
        @output
            output [N, num_fg, feat_dim]
            aff_softmax [num_fg * group, N, N]
        '''
        super(NMSRelation, self).__init__()
        assert dim[0] == dim[1], "dim[0] must equal to dim[1]"
        assert fc_dim[1] == group, "fc_dim[1] must equal to group"
        self.fc_dim = fc_dim
        self.group = group
        self.feat_dim = feat_dim
        self.dim = dim
        self.fc1 = nn.Linear(fc_dim[0], fc_dim[1])
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(feat_dim, dim[0])
        self.fc3 = nn.Linear(feat_dim, dim[1])
        self.conv = nn.Conv2d(fc_dim[1] * feat_dim, dim[2], 1, 1)

    def forward(self, roi_feat, position_mat, num_rois):
        dim_group = (self.dim[0] / self.group, self.dim[1] / self.group, self.dim[2] / self.group)
        #[num_fg, N, feat_dim]
        roi_feat = roi_feat.transpose(0, 1)
        position_embedding = extract_pairwise_multi_position_embedding(position_mat, self.fc_dim[0])
        #[num_fg, N, N, fc_dim[0]]
        position_embedding_reshaped = position_embedding.reshape(-1, self.fc_dim[0])
        postion_feat1 = self.relu(self.fc1(position_embedding_reshaped))
        aff_weight = postion_feat1.reshape(-1, num_rois, num_rois, self.fc_dim[1])
        #[num_fg, fc_dim[1], N, N]
        aff_weight = aff_weight.permute(0, 3, 1, 2)
        #[num_fg, N, dim[0]]
        q_data = self.fc2(roi_feat)
        #[num_fg, N, group, dim[0] / group]
        q_data_batch = q_data.reshape(q_data.shape[0], q_data.shape[1], self.group, int(dim_group[0]))
        #[num_fg, group, N, dim[0] / group]
        q_data_batch = q_data_batch.transpose(1, 2)
        #[num_fg * group, N, dim[0] / group]
        q_data_batch = q_data_batch.reshape(-1, q_data_batch.shape[2], q_data_batch.shape[-1])
        k_data = self.fc3(roi_feat)
        k_data_batch = k_data.reshape(k_data.shape[0], k_data.shape[1], self.group, int(dim_group[1]))
        k_data_batch = k_data_batch.transpose(1, 2)
        #[num_fg * group, N, dim[1] / group]
        k_data_batch = k_data_batch.reshape(-1, k_data_batch.shape[2], k_data_batch.shape[-1])
        v_data = roi_feat
        aff = torch.matmul(q_data_batch, k_data_batch.transpose(1, 2))
        #[num_fg * group, N, N]
        aff_scale = (1. / math.sqrt(float(dim_group[1]))) * aff
        #[num_fg * fc_dim[1], N, N]
        aff_weight_reshaped = aff_weight.reshape(-1, num_rois, num_rois)
        #[num_fg * group, N, N]
        weighted_aff = torch.log(torch.max(aff_weight_reshaped, torch.ones_like(aff_weight_reshaped) * 1e-6)) + aff_scale
        aff_softmax = nn.Softmax(-1)(weighted_aff)
        aff_softmax_reshaped = aff_softmax.reshape(-1, self.group * num_rois, num_rois)
        #[num_fg, fc_dim[1] * N, feat_dim]
        output_t = torch.matmul(aff_softmax_reshaped, v_data)
        #[num_fg, fc_dim[1], N, feat_dim]
        output_t = output_t.reshape(-1, self.fc_dim[1], num_rois, self.feat_dim)
        output_t = output_t.permute(1, 3, 2, 0)
        #[1, fc_dim[1] * feat_dim, N, num_fg]
        output_t = output_t.reshape(1, -1, num_rois, aff_weight.shape[0])
        #[1, dim[2], N, num_fg]
        linear_out = self.conv(output_t)
        output = linear_out.permute(2, 3, 1, 0)
        output = linear_out.reshape(output.shape[0], output.shape[1], output.shape[2])

        return output, aff_softmax


