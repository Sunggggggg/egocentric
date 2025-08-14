import numpy as np

import torch
import torch.nn as nn
import configs.constant as _C
from lib.utils.rotation_utils import rotation_6d_to_matrix

from .temp_tokenhmr import TemporalTokenHMR
from .net_HeadNet import HeadNet, HeadNetConfig
from .net_TokenClassifier import TokenGen

flatten = lambda x : torch.flatten(x, start_dim=0, end_dim=1)
unflatten = lambda x, B, T : x.reshape((B, T) + x.shape[1:])

class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        # --------- T-pose --------- #
        mean_params = np.load(_C.SMPL.SMPL_MEAN_PARAM)
        mean_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        mean_shape = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        self.register_buffer('mean_pose', mean_pose)        # [1, 144]
        self.register_buffer('mean_shape', mean_shape)      # [1, 10]
        
        # --------- TempTokenHMR --------- #
        model_state_dict = torch.load('/mnt/SKY/egocentric/exp/codebook/temp_simple_transformer/best_net.pth')
        self.token_hmr = TemporalTokenHMR()
        self.token_hmr.load_state_dict(model_state_dict['net'], strict=True)
        for param in self.token_hmr.parameters():
            param.requires_grad = False
        
        del self.token_hmr.model.encoder
        
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
        self.token_gen = TokenGen(256, 256, 160)
        self.orient_head = nn.Linear(256, 6)
        self.contact_head = nn.Linear(256, 21)
    
    def forward(self, head_traj):
        """ head_traj : [B, T, 9]
        """
        B, T = head_traj.shape[:2]
        
        # ----------- Body pose ----------- # 
        head_feat = self.head_net(head_traj)             # [B, T, 256]
        
        cls_feat = self.token_gen(flatten(head_feat))
        cls_feat = unflatten(cls_feat, B, T)             # [B, T, 160, 2048]
        output = self.token_hmr.decoding(cls_feat)  
        
        # ----------- Additional ----------- # 
        global_orient = self.orient_head(head_feat)      # [B, T, 6]
        contact_label = self.contact_head(head_feat)     # [B, T, J]
        
        output.update({'global_orient': rotation_6d_to_matrix(global_orient), 
                       'contact': contact_label,})
        
        return output
        
        
        
        