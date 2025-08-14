import cv2
import torch
import configs.constant as _C
from pathlib import Path
from tqdm import tqdm
from smplx import SMPLH
import imageio

from lib.dataset.dataclass import TrainingData
from lib.dataset.egocentric_dataloader import get_loader
from lib.utils.rotation_utils import quaternion_to_axis_angle
from lib.vis.renderer import Renderer, get_global_cameras
from lib.utils.pose_visualize import motion_contact_gif

HDF5_PATH = Path(_C.DATA.HDF5_PATH)
FILE_LIST_PATH = Path(_C.DATA.FILE_LIST_PATH)
SUBSEQ_LEN = 128
train_loader, val_loader = get_loader(HDF5_PATH, FILE_LIST_PATH, SUBSEQ_LEN)

smpl_cfg = {
        'model_path': "/mnt/SKY/h_egohmr/TokenHMR/_egohenu/lib/data/body_models/smplh",
        'gender': "neutral",
        'num_betas': 10,
        'batch_size': SUBSEQ_LEN
    }
smplh = SMPLH(**smpl_cfg).cuda()
width, height = 500, 500
focal_length = (width**2+height**2)**0.5
renderer = Renderer(width=width, height=height, focal_length=focal_length, 
         device='cuda', faces=smplh.faces, )
colors = torch.ones((1, 4)).float().cuda(); colors[..., :3] *= 0.9

for batch in train_loader :
    batch : TrainingData = batch
    for k, v in batch.__dict__.items() :
        print(k, v.shape)

    world_kp3d = batch.joints_wrt_world             # [B, T, 21, 3]
    world_transl = batch.T_world_root[..., 4:]      # [B, T, 3]
    world_orient = quaternion_to_axis_angle(batch.T_world_root[..., :4]) # [B, T, 3]
    body_pose = quaternion_to_axis_angle(batch.body_quats)               # [B, T, 21, 3]               
    betas = batch.betas[..., :10]
    contact = batch.contacts        # [B, T, 21]
    
    smpl_output = smplh(
        global_orient=world_orient.view(-1, 3).cuda(),
        body_pose=body_pose.view(-1, 63).cuda(),
        betas=betas.view(-1, 10).cuda(),
        transl=world_transl.view(-1, 3).cuda(),
    )
    verts_glob = smpl_output.vertices   # [32, 6890, 3]
    # motion_contact_gif(world_kp3d[0], contact[0])
    
    global_R, global_T, global_lights = \
        get_global_cameras(verts_glob, 'cuda', distance=7, position=(3.0, 3.0, 10.0))
    seqlen = len(verts_glob)
    
    writer = imageio.get_writer(f'output.mp4', fps=30, mode='I', 
                                format='FFMPEG', macro_block_size=1)
    renderer.set_ground(length=10, center_x=0., center_z=0.)
    for t in tqdm(range(seqlen)):
        cameras = renderer.create_camera(global_R[t], global_T[t])
        faces = renderer.faces.clone().squeeze(0)
        img_glob = renderer.render_with_ground(verts_glob[[t]], faces, colors, cameras, global_lights)
        writer.append_data(img_glob[..., ::-1])
    writer.close()
    exit()