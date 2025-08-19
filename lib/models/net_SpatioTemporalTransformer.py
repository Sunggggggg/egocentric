import torch
from torch import nn

class AlternativeST(nn.Module):
    def __init__(self, repeat_time=1, s_layer=2, t_layer=2, embed_dim=256, nhead=8):
        super(AlternativeST, self).__init__()
        self.num_layer = repeat_time
        self.s_layer = s_layer
        self.t_layer = t_layer
        self.STB = nn.ModuleList()
        self.TTB = nn.ModuleList()
        for _ in range(repeat_time):
            if self.s_layer != 0:
                spatial_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
                self.STB.append(nn.TransformerEncoder(spatial_layer, num_layers=s_layer))
            if self.t_layer != 0:
                temporal_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
                self.TTB.append(nn.TransformerEncoder(temporal_layer, num_layers=t_layer))

    def forward(self, feat):
        assert len(feat.shape) == 4, 'The input shape dimension should be 4, e.g., (batch, seq_len, joint_num, feat_dim).'
        batch, seq_len, joint_num, feat_dim = feat.shape
        input_token = feat[:, :, -1].clone().detach()
        for i in range(self.num_layer):
            if joint_num == 21+1:
                if self.s_layer != 0:
                    feat = self.STB[i](feat.reshape(batch*seq_len, joint_num, -1)).reshape(batch, seq_len, joint_num, -1)
                    feat[:, :, -1] = input_token
                if self.t_layer != 0:
                    feat = self.TTB[i](feat.reshape(batch, seq_len, joint_num, -1).permute(0, 2, 1, 3).reshape(batch*joint_num, seq_len, -1)).reshape(batch, joint_num, seq_len, -1).permute(0, 2, 1, 3)
                    feat[:, :, -1] = input_token
            else:
                feat = self.STB[i](feat.reshape(batch*seq_len, joint_num, -1)).reshape(batch, seq_len, joint_num, -1)
                feat = self.TTB[i](feat.reshape(batch, seq_len, joint_num, -1).permute(0, 2, 1, 3).reshape(batch*joint_num, seq_len, -1)).reshape(batch, joint_num, seq_len, -1).permute(0, 2, 1, 3)
        return feat

class MotionNet(nn.Module):
    def __init__(self, num_layer=6, s_layer=1, t_layer=1, joint_feat_dim=256, node_num=23, spatial_embedding=True, temporal_embedding=False, nhead=8):
        super(MotionNet, self).__init__()

        self.use_spatial_embedding = spatial_embedding
        self.use_temporal_embedding = temporal_embedding
        self.transformer = AlternativeST(repeat_time=num_layer, s_layer=s_layer, t_layer=t_layer, embed_dim=joint_feat_dim, nhead=nhead) 
        if self.use_temporal_embedding:
            max_seq_len = 200
            self.temp_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, joint_feat_dim))
        if self.use_spatial_embedding:
            max_joint_num = node_num 
            self.joint_position_embed = nn.Parameter(torch.zeros(1, 1, max_joint_num, joint_feat_dim))

    def forward(self, joint_embedding):
        batch, seq_len = joint_embedding.shape[0], joint_embedding.shape[1]
        # add spatial positional embedding
        if self.use_spatial_embedding:
            joint_embedding = joint_embedding + self.joint_position_embed
        # add temporal positional embedding
        if self.use_temporal_embedding:
            joint_embedding = joint_embedding + self.temp_embed[:,:seq_len,:,:]
        ST_feat = self.transformer(joint_embedding)
        
        return ST_feat