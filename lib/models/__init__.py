import torch
from .ego_tokenhmr import Model

def load_egotokenhmr(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = Model()
    model.load_state_dict(ckpt['net'], strict=True) # TODO : Multi-gpu?

    return model