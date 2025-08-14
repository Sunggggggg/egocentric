import os
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
import imageio

from lib.utils.rotation_conversions import *
import configs.constant as _C
from lib.models.temp_tokenhmr import TemporalTokenHMR
from lib.utils.pose_visualize import single_pose

unflatten = lambda x : x.reshape((_C.TRAIN.batch_size, _C.TRAIN.seqlen) + x.shape[1:])
# ----------- Dataset ----------- #
OUT_DIR = 'exp'
NAME = 'temp_simple_transformer'
save_dir = os.path.join(OUT_DIR, NAME)

# ----------- Model ----------- #
model = TemporalTokenHMR().cuda()
model = model.eval()

# ----------- Resume ----------- #
ckpt_file = os.path.join(save_dir, 'best_net.pth')
if os.path.exists(ckpt_file) :
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['net'], strict=True)
    logger.info(f"Load ckpt from {ckpt_file}")
else :
    logger.info(f"No ckpt in  {ckpt_file}")

# ----------- Data ----------- #
data = dict(np.load('parsed_data/tokenization_data/smplh/val/val_HDM05.npz'))

video_names_unique, group = np.unique(data['name'], return_index=True)
perm = np.argsort(group)
group_perm = group[perm]
indices = np.split(np.arange(0, len(data['name'])), group_perm[1:])
chunk = 16

for idx in range(len(video_names_unique)) :
    indexes = indices[idx]
    seqlen = len(indexes)
    name = data['name'][indexes]
    pose_body = torch.from_numpy(data['pose_body'][indexes])  # [T, 63]
    betas = torch.from_numpy(data['betas'][indexes])
    gender = data['gender'][indexes]
    logger.info(f"Seq name : {name[0]}")
    pose_body_6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose_body.reshape(-1, 21, 3)))
    
    writer = imageio.get_writer(
        'test.mp4', fps=30, mode='I', 
        format='FFMPEG', macro_block_size=1,
    )
    
    split_idx = np.stack([np.arange(seqlen), np.arange(chunk, seqlen+chunk)], axis=1)
    for start, end in split_idx :
        tar_pose = pose_body_6d[start:end].unsqueeze(0)             # [1, T, 21, 6]
        output, _, _ = model(tar_pose)                              # [1, T, 21, 6]
        vertices = output['pred_body_vertices'][0, (end-start)//2]  # [6890, 3]
        rend_img = single_pose(vertices)
        
        writer.append_data(rend_img)
    
    writer.close()    
    exit()
    