"""
TokenHMR에서 했던 방식처럼
token -> SA+CA -> tokenout -> decode
"""

import numpy as np
from sympy import false
import torch
import torch.nn as nn
from smplx import SMPLHLayer

import configs.constant as _C
from lib.utils.rotation_utils import rotation_6d_to_matrix

from .temp_tokenhmr import TemporalTokenDecoder
from .net_HeadNet import HeadNet, HeadNetConfig
from .net_TokenClassifier import TokenGen
from .net_Decoder import asdict, TransformerConfig, TransformerDecoder

flatten = lambda x : torch.flatten(x, start_dim=0, end_dim=1)
unflatten = lambda x, B, T : x.reshape((B, T) + x.shape[1:])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # --------- SMPL --------- #
        self.body_model = SMPLHLayer(_C.SMPL.SMPLH_MODEL_PATH, gender='neutral', num_betas=10)
        
        # --------- T-pose --------- #
        mean_params = np.load(_C.SMPL.SMPL_MEAN_PARAM)
        mean_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).reshape(1, 1, -1)
        self.register_buffer('mean_pose', mean_pose)
        
        # --------- TempTokenHMR --------- #
        temptokenhmr_ckpt = 'exp/codebook/light_transformer/best_net.pth'
        self.token_decoder = TemporalTokenDecoder(temptokenhmr_ckpt)
        self.token_decoder.eval()
        for param in self.token_decoder.parameters():
            param.requires_grad = False
        
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
        
        # --------- Decoding module --------- #
        self.decoder_cfg = asdict(TransformerConfig())
        self.transformer_decoder = TransformerDecoder(**self.decoder_cfg)
        
        self.token_gen = TokenGen(256, 160, 2048)
        self.orient_head = nn.Linear(256, 6)
        self.contact_head = nn.Linear(256, 21)
        
    def forward(self, head_traj):
        """ head_traj : [B, T, 9]
        """
        B, T = head_traj.shape[:2]
        BT = B*T
        
        # ----------- Head net. ----------- # 
        head_feat = flatten(self.head_net(head_traj))   # [BT, 256]
        mean_pose = self.mean_pose.expand(B, T, -1)       # [B, T, 144]

        # ----------- Decoding ----------- # 
        token = torch.zeros(BT, 1, 1).to(head_feat.device)

        token_out = self.transformer_decoder(token, context=head_feat[:, None])
        token_out = token_out.squeeze(1)                # [BT, 256]
        
        pred_orient = unflatten(self.orient_head(token_out), B, T)       # [B, T, 6]
        pred_contact = unflatten(self.contact_head(token_out), B, T)     # [B, T, 21]
        cls_logits_softmax = self.token_gen(token_out)                   # [BT, 160, 2048]
        pred_body = self.token_decoder(unflatten(cls_logits_softmax, B, T)) # [B, T, 21*6]
        
        pred_body_pose = torch.cat([pred_orient, pred_body], dim=-1) + mean_pose[..., :132]     # [BT, 132]
        
        # ----------- Post-processing ----------- #  
        pred_body_pose_rotmat = rotation_6d_to_matrix(pred_body_pose.reshape(-1, 22, 6))        # [BT, 22, 3, 3]
        smpl_output = self.body_model(
            global_orient=pred_body_pose_rotmat[:, [0]],
            body_pose=pred_body_pose_rotmat[:, 1:], 
        )
        
        output = {
            'pred_jnts': unflatten(smpl_output.joints, B, T),
            'pred_verts': unflatten(smpl_output.vertices, B, T),
            'pred_pose': pred_body_pose[..., 6:].reshape(B, T, -1),
            'global_orient': pred_body_pose[..., :6].reshape(B, T, 6),
            'contact': pred_contact,
        }
        
        return output