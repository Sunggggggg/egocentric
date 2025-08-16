import torch
import torch.nn as nn
from .modules import MixerLayer, FCBlock
from .tokenhmr.vanilla_pose_vqvae import DecodeTokens as VanillaDecodeTokens

class Proxy(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.set_gpu = False
    def tokenize(self, x):
        if not self.set_gpu:
            self.tokenizer = self.tokenizer.to(x.device)
            self.set_gpu = True
        return self.tokenizer(x)
    
class DecodeToken(nn.Module):
    def __init__(self, tokenizer_checkpoint_path=None, tokenizer_type='Vanilla', is_train = True):
        super().__init__()
        self.tokenizer_module = VanillaDecodeTokens(tokenizer_checkpoint_path)
        self.frozen_tokenizer()
        self.tokenize = Proxy(self.tokenizer_module).tokenize

    def forward(self, cls_logits):
        # x: (B*T, token_num, 2048)
        B_mul_T = cls_logits.shape[0] # B*T

        cls_logits_softmax = cls_logits.softmax(-1) # (B*T, token_num, 2048)
        
        smpl_thetas6D = self.tokenize(cls_logits_softmax).reshape(B_mul_T, -1) # (B*T, 21*6)
        smpl_thetas6D = smpl_thetas6D
        
        return smpl_thetas6D

    def frozen_tokenizer(self):
        self.tokenizer_module.eval()
        for name, params in self.tokenizer_module.named_parameters():
            params.requires_grad = False
            
class SingleJointMixerModule(nn.Module):
    def __init__(self, in_channels, token_num):
        super().__init__()

        self.in_channels = in_channels
        
        self.hidden_dim = 64
        self.hidden_inter_dim = 256
        self.token_inter_dim = 64
        self.dropout = 0.0    
        self.num_blocks = 4
    
        self.token_num = token_num
        self.token_class_num = 2048
        
        self.mixer_tokenizer = FCBlock(
            self.in_channels,
            self.token_num * self.hidden_dim)
        self.mixer_token = nn.ModuleList(
            [MixerLayer(hidden_dim=self.hidden_dim,
                        hidden_inter_dim=self.hidden_inter_dim,
                        token_dim=self.token_num, 
                        token_inter_dim=self.token_inter_dim, 
                        dropout_ratio=self.dropout) for _ in range(self.num_blocks)])
        self.mixer_norm = FCBlock(
            self.hidden_dim, self.hidden_dim)
        self.class_pred = nn.Linear(self.hidden_dim, self.token_class_num)


    def forward(self, x):
        # feature: [B*T, in_channels]
        B_mul_T = x.shape[0] # B*T
        
        tokens = self.mixer_tokenizer(x) # [B*T, token_num*64]
        tokens = tokens.reshape(B_mul_T, self.token_num, -1) # [B*T, token_num, 64]
        
        for mixer_layer in self.mixer_token:
            tokens = mixer_layer(tokens)
            
        tokens = self.mixer_norm(tokens) # [B*T, token_num, 64]
        tokens = self.class_pred(tokens) # [B*T, token_num, 2048]
        return tokens # [B*T, token_num, 2048]

class TokenGen(nn.Module):
    def __init__(self, 
                 in_channels=256, 
                 token_num=160, 
                 token_class_num=2048
                 ):
        super().__init__()
        hidden_dim = 64
        hidden_inter_dim = 256
        token_inter_dim = 64
        dropout = 0.1
        num_blocks = 4
        
        self.token_num = token_num
        self.mixer_trans = FCBlock(in_channels, token_num * hidden_dim)

        self.mixer_head = nn.ModuleList(
            [MixerLayer(hidden_dim, hidden_inter_dim,
                token_num, token_inter_dim, dropout) for _ in range(num_blocks)])
        self.mixer_norm_layer = FCBlock(hidden_dim, hidden_dim)
        self.class_pred_layer = nn.Linear(hidden_dim, token_class_num)

    def forward(self, x):
        """
        x : [BT, 256]
        """
        batch_size = x.shape[0]
        cls_feat = self.mixer_trans(x)  # [BT, 256] -> [BT, 160*32]
        cls_feat = cls_feat.reshape(batch_size, self.token_num, -1) # [BT, 160, 32]
        
        for mixer_layer in self.mixer_head:
            cls_feat = mixer_layer(cls_feat)
        cls_feat = self.mixer_norm_layer(cls_feat)
        cls_logits = self.class_pred_layer(cls_feat) 
        cls_logits_softmax = cls_logits.softmax(-1)
        
        return cls_logits_softmax