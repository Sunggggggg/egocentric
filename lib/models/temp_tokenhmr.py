import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from einops import rearrange
from smplx import SMPLH, SMPLHLayer

import configs.constant as _C
from .tokenhmr.vanilla_pose_vqvae import VanillaTokenizer
from .tokenhmr.config import TokenHMRConfig
from .net_Codebook import Transformer, TransformerConfig
from lib.utils.rotation_conversions import *

smpl_type = 'smplh'
body_model = eval(f'{smpl_type.upper()}Layer')(_C.SMPL.SMPLH_MODEL_PATH, num_betas=10, ext='pkl')
body_model = body_model.cuda() if torch.cuda.is_available() else body_model
body_model = body_model.eval()

class TemporalTokenHMR(nn.Module):
    """
    Fine-tuning temporal-attentive version
    """
    def __init__(self,):
        super().__init__()
        # --------- TokenHMR --------- #
        tokenhmr_config = TokenHMRConfig()
        self.model = VanillaTokenizer(tokenhmr_config)
        # add_safe_globals([CfgNode, set])
        model_state_dict = torch.load(_C.TOKENHMR.MODEL_PATH)
        self.model.load_state_dict(model_state_dict['net'], strict=True)
        
        # --------- Spatio --------- #
        spatio_trans_config = TransformerConfig(
            input_dim=256,
            output_dim=256,
            d_latent=256,
            d_feedforward=512, 
            #num_encoder_layers=4,
            num_encoder_layers=2,
            num_heads=4,
        )
        self.spatio_transformer = Transformer(spatio_trans_config)
        
        # --------- Temporal --------- #
        temp_trans_config = TransformerConfig(
            input_dim=126, 
            output_dim=126,
            d_latent=256,
            d_feedforward=512, 
            #num_encoder_layers=4,
            num_encoder_layers=2,
            num_heads=4,
        )
        self.temp_transformer = Transformer(temp_trans_config)
    
    def post_processing(self, pose_6d):
        B, T = pose_6d.shape[:2]
        
        pred_pose_body_rotmat = rotation_6d_to_matrix(
            pose_6d.reshape(-1, 6)).reshape(B, T, -1, 3, 3)             # [B, T, 21, 3, 3]
        pred_pose_body_aa = matrix_to_axis_angle(pred_pose_body_rotmat) 
        
        pred_body_mesh = body_model(body_pose=pred_pose_body_rotmat.view(B*T, -1, 3, 3))
        
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
        
        # --------- Spatio --------- #
        x = rearrange(motion, 'b t j c -> (b t) j c')
        x_encoder = self.model.encoder(x, global_step).contiguous()          # [BT, J, c]
        x_encoder = rearrange(x_encoder, 'bt c n -> bt n c')                 # [BT, 160, 256]
        x_encoder = self.spatio_transformer(x_encoder)
        x_encoder = rearrange(x_encoder, 'bt n c -> bt c n')
        
        # --------- Quantization --------- #
        x_quantized, loss, perplexity = self.model.quantizer(x_encoder)
        
        # --------- Decoding --------- #
        output = self.model.decoder(x_quantized)     # [BT, J, 6]
        pred_pose_body_6d = output['pred_pose_body_6d']
        
        # --------- Temporal --------- #
        pred_pose_body_6d = rearrange(pred_pose_body_6d, '(b t) j c -> b t (j c)', b=B) # [B, T, 126]
        pred_pose_body_6d = self.temp_transformer(pred_pose_body_6d)                    # [B, T, 21*6]
        
        # --------- Post processing --------- #
        output = self.post_processing(pred_pose_body_6d)
        
        return output, loss, perplexity
    
    def inference(self, motion, chunk_size=16):
        """
        motion : [B, T, J, 6]
        """
        seqlen = motion.shape[1]
        
        frame = np.arange(seqlen)
        frame_chunk = parse_chunk(frame, min_len=chunk_size)
        
        pred_pose = []
        pred_verts = []
        pred_jnts = []
        
        for start, end in tqdm(frame_chunk, desc="Interence...") :
            tar_pose = motion[:, start:end]
            output, _, _ = self.forward(tar_pose)
            tar_output = {output[k][:, chunk_size//2] for k, v in output.items()}   # [B, j, d]
            
            pred_pose.append(tar_output['pred_pose_body_aa'])
            pred_verts.append(tar_output['pred_body_vertices'])
            pred_jnts.append(tar_output['pred_body_joints'])
        
        results = {'pred_pose': torch.cat(pred_pose),
                'pred_verts': torch.cat(pred_verts),
                'pred_jnts': torch.cat(pred_jnts)}
        
        return results
    
    def decoding(self, feat, chunk_size=16):
        """
        motion : [B, T, 160, 256]
        """
        B, seqlen = feat.shape[:2]
        
        frame_chunk = parse_chunk(seqlen, min_len=chunk_size)
        
        pred_pose = []
        pred_verts = []
        pred_jnts = []
        
        for start, end in frame_chunk :
            x_encoder = torch.flatten(feat[:, start:end], 0, 1)     # [BT, 160, 256]
            x_encoder = self.spatio_transformer(x_encoder)
            x_encoder = rearrange(x_encoder, 'bt n c -> bt c n')    # [BT, 256, 160]
            
            # --------- Quantization --------- #
            x_quantized, _, _ = self.model.quantizer(x_encoder)     # [BT, 256, 160]
        
            # --------- Decoding --------- #
            output = self.model.decoder(x_quantized)                # [BT, J, 6]
            pred_pose_body_6d = output['pred_pose_body_6d']
            
            # --------- Temporal --------- #
            pred_pose_body_6d = rearrange(pred_pose_body_6d, '(b t) j c -> b t (j c)', b=B) # [B, T, 126]
            pred_pose_body_6d = self.temp_transformer(pred_pose_body_6d)                    # [B, T, 21*6]
            output = self.post_processing(pred_pose_body_6d)
            
            tar_output = {k : output[k][:, (end-start)//2] for k in output.keys()}   # [B, j, d]
            
            pred_pose.append(tar_output['pred_pose_body_aa'])
            pred_verts.append(tar_output['pred_body_vertices'])
            pred_jnts.append(tar_output['pred_body_joints'])
        
        results = {'pred_pose': torch.stack(pred_pose, dim=1),
                'pred_verts': torch.stack(pred_verts, dim=1),
                'pred_jnts': torch.stack(pred_jnts, dim=1),}
        
        return results

def parse_chunk(seqlen, min_len=16):
    split_idx = np.stack([np.arange(seqlen), np.arange(min_len, seqlen+min_len)], axis=1)
    split_idx[split_idx > seqlen] = seqlen
    return split_idx