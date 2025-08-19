import torch
import torch.nn as nn
from einops import rearrange
from .temp_tokenhmr import TemporalTokenDecoder
from .components.pose_transformer import TransformerCrossAttn
from .net_HeadNet import HeadNet, HeadNetConfig
from .net_SpatioTemporalTransformer import MotionNet
from .net_TokenClassifier import Joint2Token

class JointInitModule(nn.Module):
    def __init__(self, num_joints=21, embed_dim=256):
        super().__init__()
        self.joint_quries = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        self.transformer = TransformerCrossAttn(
            dim=embed_dim, 
            depth=3, 
            heads=4, 
            dim_head=int(embed_dim//4),
            mlp_dim=int(embed_dim*2),
            dropout=0.1, 
        )
        
    def forward(self, context):
        """
        context : [B, 2048, 256] (codebook)
        """
        B = context.shape[0]
        
        joint_quries = self.joint_quries.repeat(B, 1, 1)
        joint_quries = joint_quries + self.pos_embedding
        
        update_quries = self.transformer(joint_quries, context=context)
        return update_quries

class Model(nn.Module): 
    def __init__(self, ):
        super().__init__()
        # --------- Tokenizer --------- #
        temptokenhmr_ckpt = 'exp/codebook/light_transformer/best_net.pth'
        self.token_decoder = TemporalTokenDecoder(temptokenhmr_ckpt)
        self.token_decoder.eval()
        for param in self.token_decoder.parameters():
            param.requires_grad = False
        
        self.init_joint_quries = JointInitModule()
        
        # --------- Head enc. --------- #
        self.head_transformer_cfg = HeadNetConfig(
            input_pose_dim=9, d_latent=256, d_feedforward=1024,
            num_encoder_layers=2, num_heads=8, dropout_p=0.1, activation="gelu")
        self.head_net = HeadNet(self.head_transformer_cfg)
        
        # --------- STformer --------- #
        self.st_transfomer = MotionNet(
            num_layer=3, s_layer=1, t_layer=1, joint_feat_dim=256, 
            node_num=22, spatial_embedding=True, temporal_embedding=True, nhead=4
        )
        self.token_gen = Joint2Token(256, 160, 2048)

    def forward_init(self, B):
        codebook = self.token_decoder.quantizer.codebook.unsqueeze(0).repeat(B, 1, 1)
        joint_quries = self.init_joint_quries(codebook)         # [B, J, dim]  
        
        return joint_quries

    def forward(self, head_traj):
        B, T = head_traj.shape[:2]
        
        joint_quries = self.forward_init(B)
        joint_quries = joint_quries[:, None].repeat(1, T, 1, 1) # [B, T, J, dim]
        
        ### Head net. ###
        head_feat = self.head_net(head_traj)            # [B, T, 256]
        joint_quries = torch.cat([joint_quries, head_feat.unsqueeze(2)], dim=2)
        joint_encode = self.st_transfomer(joint_quries) # [B, T, 22, 256]
        
        joint_encode = rearrange(joint_encode, 'b t j d -> (b t) j d')
        cls_logits_softmax = self.token_gen(joint_encode)
        cls_logits_softmax = rearrange(cls_logits_softmax, '(b t) j d -> b t j d', b=B)
        delta_pose = self.token_decoder(cls_logits_softmax)
        print(delta_pose.shape)
        
        return 