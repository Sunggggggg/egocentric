import numpy as np

import torch
import torch.nn as nn
import configs.constant as _C

from .net_HeadNet import HeadNet, HeadNetConfig
from .net_TokenClassifier import SingleJointMixerModule, DecodeToken
from lib.utils.geometry import rot6d_to_rotmat

class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        # --------- T-pose --------- #
        mean_params = np.load(_C.SMPL.SMPL_MEAN_PARAM)
        mean_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        mean_shape = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        self.register_buffer('mean_pose', mean_pose)        # [1, 144]
        self.register_buffer('mean_shape', mean_shape)      # [1, 10]
        
        # --------- Head enc. --------- #
        self.head_transformer_cfg = HeadNetConfig(
            input_pose_dim=9,
            d_latent=256,
            d_feedforward=1024,
            num_encoder_layers=8,
            num_heads=8,
            dropout_p=0.1,
            activation="gelu"
        )
        self.head_net = HeadNet(self.head_transformer_cfg)
        
        # --------- TokenHMR Decoder --------- #
        self.decode_token = DecodeToken(tokenizer_checkpoint_path=_C.TOKENHMR.MODEL_PATH)
        
        # --------- Stage1. Init Pose --------- #
        self.init_root_layer = nn.Linear(256, 6)
        self.init_cls_logits = SingleJointMixerModule(256, 210)
        
        # --------- Stage2. Final Pose --------- #
        self.tokenize_init_pose = nn.Linear(6, 256)
        
    def init_pose(self, head_feat):
        B, T, _ = head_feat.shape
        
        feat = head_feat.reshape(B*T, -1)               # (B*T, dim)
        T_body_pose = self.mean_pose.expand(B*T, -1)    # (B*T, 144)

        delta_root = self.init_root_layer(feat)             # (B*T, 6)

        init_cls_logits = self.init_cls_logits(feat)        # (B*T, 210, 2048)
        init_cls_logits_softmax = init_cls_logits.softmax(-1)
        delta_pose = self.decode_token(init_cls_logits_softmax)     # (B*T, 126)
        
        init_root_6d = delta_root + T_body_pose[:, :6]      # (B*T, 6)
        init_pose_6d = delta_pose + T_body_pose[:, 6:132]   # (B*T, 126)
        
        init_root_6d = init_root_6d.reshape(B, T, 6).unsqueeze(2)   # (B, T, 1, 6)
        init_pose_6d = init_pose_6d.reshape(B, T, 21, 6)            # (B, T, 21, 6)

        init_body_6d = torch.cat((init_root_6d, init_pose_6d), dim=2)   # (B, T, 22, 6)
        return init_body_6d
    
    def final_pose(self, init_body_6d, head_feat):
        B, T = init_body_6d.shape[:2]
        init_body_6d
        
        
        
        init_pose_6d = init_pose_mat[:, :, :, :2, :].reshape(B, T, 22, 6) # (B, T, 22, 6)
        
        tokenized_init_pose = self.tokenize_init_pose(init_pose_6d) # (B, T, 22, 256)
        
        joint_embedding = torch.cat((tokenized_init_pose, head_feat.unsqueeze(2)), dim=2) # (B, T, 23, 256)
        pose_token = self.st_transfomer(joint_embedding) # (B, T, 23, 256)

        
        final_cls_logits = self.final_cls_logits(pose_token[:, :, 1:22, :].reshape(B*T, 21, 256)) # (B*T, 210, 2048)
        delta_pose = self.decode_token(final_cls_logits) # (B*T, 126)
        delta_root = self.final_root_layer(pose_token[:, :, [0], :]) # (B, T, 1, 6)
        
        final_root_6d = delta_root + init_pose_6d[:, :, [0], :] # (B, T, 1, 6)        
        final_pose_6d = delta_pose.reshape(B, T, 21, 6) + init_pose_6d[:, :, 1:, :]



        final_body_6d = torch.cat((final_root_6d, final_pose_6d), dim=2) # (B, T, 22, 6)
        final_body_mat = rot6d_to_rotmat(final_body_6d.reshape(-1, 6)).reshape(B, T, 22, 3, 3) # (B, T, 22, 3, 3)
        
        # final_pose = final_body_mat[:, :, :2, :].reshape(B, T, 21, 6) # (B, T, 22, 6)

        return final_body_mat
    
    def forward(self, head_traj):
        # head_traj: (B, T, 9)
        
        head_feat = self.head_net(head_traj)        # (B, T, 256)

        # --------- Stage1. Init Pose --------- #
        init_pose_mat = self.init_pose(head_feat)   # (B, T, 22, 6)
        
        # --------- Stage2. Final Pose --------- #
        final_pose_mat = self.final_pose(init_pose_mat, head_feat) # (B, T, 22, 6)
        
        return init_pose_mat, final_pose_mat