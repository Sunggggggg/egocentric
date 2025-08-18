import re
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from einops import rearrange
from smplx import SMPLH, SMPLHLayer

import configs.constant as _C
from .tokenhmr.vanilla_pose_vqvae import VanillaTokenizer, PoseSPDecoderV1, QuantizeEMAReset
from .tokenhmr.config import TokenHMRConfig
from .masked_autoencoder import MAE
from .net_Codebook import Transformer, TransformerConfig
from lib.utils.rotation_conversions import *

class TemporalTokenHMR(nn.Module):
    """
    Fine-tuning temporal-attentive version
    """
    def __init__(self,):
        super().__init__()
        # 
        smpl_type = 'smplh'
        body_model = eval(f'{smpl_type.upper()}Layer')(_C.SMPL.SMPLH_MODEL_PATH, num_betas=10, ext='pkl')
        self.body_model = body_model.eval()

        # --------- TokenHMR --------- #
        tokenhmr_config = TokenHMRConfig()
        self.tokenizer = VanillaTokenizer(tokenhmr_config)
        # add_safe_globals([CfgNode, set])
        model_state_dict = torch.load(_C.TOKENHMR.MODEL_PATH)
        self.tokenizer.load_state_dict(model_state_dict['net'], strict=True)
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        
        # --------- Spatial --------- #
        self.enc_layer = nn.Linear(256//2, 256)
        self.s_transformer = MAE(depth=4, embed_dim=256, mlp_hidden_dim=512,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=160)
        
        # --------- Temporal --------- #
        temp_trans_config = TransformerConfig(
            input_dim=126, 
            output_dim=126,
            d_latent=256,
            d_feedforward=512, 
            # num_encoder_layers=4,
            num_encoder_layers=2,
            num_heads=4,
        )
        self.temp_transformer = Transformer(temp_trans_config)
    
    def post_processing(self, pose_6d):
        B, T = pose_6d.shape[:2]
        
        pred_pose_body_rotmat = rotation_6d_to_matrix(
            pose_6d.reshape(-1, 6)).reshape(B, T, -1, 3, 3)             # [B, T, 21, 3, 3]
        pred_pose_body_aa = matrix_to_axis_angle(pred_pose_body_rotmat) 
        
        pred_body_mesh = self.body_model(body_pose=pred_pose_body_rotmat.view(B*T, -1, 3, 3))
        
        output = {
            'pred_pose_body_aa': pred_pose_body_aa,
            'pred_pose_body_6d': pose_6d,
            'pred_pose_body_rotmat': pred_pose_body_rotmat, 
            'pred_body_vertices': pred_body_mesh.vertices.reshape(B, T, -1, 3),
            'pred_body_joints': pred_body_mesh.joints.reshape(B, T, -1, 3),
        }
        
        return output
    
    def forward(self, motion, global_step=None):
        """
        motion : [B, T, J, 6] (T=16)
        """
        B, T, J = motion.shape[:3]
        
        x = rearrange(motion, 'b t j c -> (b t) j c')
        x_encoder = self.tokenizer.encoder(x, global_step).contiguous() # [BT, J, c]
        x_encoder = rearrange(x_encoder, 'bt c n -> bt n c')            # [BT, 160, 256]
        x_masked, mask = self.s_transformer(x_encoder)                  # [BT, 160, 128], [BT, 160, 1]
        x_encoder = self.enc_layer(x_masked)
        x_encoder = rearrange(x_encoder, 'bt n c -> bt c n')            # [BT, 256, 160]
        
        # --------- Quantization --------- #
        x_encoder = self.tokenizer.quantizer.preprocess(x_encoder)  # [BT*160, 256]
        code_idx = self.tokenizer.quantizer.quantize(x_encoder)     # [BT*160]
        x_d = self.tokenizer.quantizer.dequantize(code_idx)
        x_quantized = x_encoder + (x_d - x_encoder).detach()
        x_quantized = x_quantized.view(B*T, 160, -1).permute(0, 2, 1).contiguous()
        
        # --------- Decoding --------- #
        output = self.tokenizer.decoder(x_quantized)     # [BT, J, 6]
        pred_pose_body_6d = output['pred_pose_body_6d']
        
        # --------- Temporal --------- #
        pred_pose_body_6d = rearrange(pred_pose_body_6d, '(b t) j c -> b t (j c)', b=B) # [B, T, 126]
        pred_pose_body_6d = self.temp_transformer(pred_pose_body_6d)                    # [B, T, 21*6]
        
        # --------- Post processing --------- #
        output = self.post_processing(pred_pose_body_6d)
        output.update({'mask': mask.reshape(B*T, 160), 'code_idx': code_idx.reshape(B*T, 160)})
        
        return output