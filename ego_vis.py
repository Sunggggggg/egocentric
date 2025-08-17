import os
from pathlib import  Path
import torch
from smplx import SMPLH
import imageio
from tqdm import tqdm
import numpy as np
import configs.constant as _C
from lib.models import load_egotokenhmr
from lib.utils.train_utils import get_device, make_input
from lib.utils.rotation_utils import quaternion_to_axis_angle
from lib.dataset.egocentric_dataloader import AmassHdf5Dataset
from lib.dataset.dataclass import collate_dataclass, TrainingData
from lib.vis.renderer import Renderer, get_global_cameras
from pytorch3d.renderer import look_at_view_transform

# --------- Hyper-param. --------- #
ckpt_path = 'exp/egohmr/only_headnet/060000_net.pth'
HDF5_PATH = Path(_C.DATA.HDF5_PATH)
FILE_LIST_PATH = Path(_C.DATA.FILE_LIST_PATH)
SUBSEQ_LEN = 128

smpl_cfg = {
        'model_path': _C.SMPL.SMPLH_MODEL_PATH,
        'gender': "neutral",
        'num_betas': 10,
        'batch_size': SUBSEQ_LEN
    }
smplh = SMPLH(**smpl_cfg).cuda()

def load_model():
    model = load_egotokenhmr(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = model.to(device)
    model.eval()
    
    return model

def load_data():
    dataset = AmassHdf5Dataset(
        hdf5_path=HDF5_PATH,
        file_list_path=FILE_LIST_PATH,
        splits=("test",),
        subseq_len=SUBSEQ_LEN,
        cache_files=False,
        slice_strategy="deterministic",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_dataclass,
        pin_memory=False
    )
    return dataloader

if __name__ == "__main__": 
    model = load_model()
    dataloader = load_data()
    
    width, height = 500, 500
    focal_length = (width**2+height**2)**0.5
    renderer = Renderer(width=width, height=height, focal_length=focal_length, device='cuda', faces=smplh.faces, )
    colors = torch.ones((1, 4)).float().cuda(); colors[..., :3] *= 0.9
    
    for batch in dataloader :
        batch : TrainingData = batch
        
        # ----------- GT vert. ----------- #
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
        gt_vertices = smpl_output.vertices
        
        # ----------- Prepared ----------- #
        batch = {k: v for k, v in batch.__dict__.items()}
        batch = get_device(batch)
        
        betas = batch['betas'][..., :10]            # [B, T, 10]
        transl = batch['T_world_root'][..., 4:]     # [B, T, 3]
        
        # ----------- Model forward ----------- #
        head_pose = make_input(batch)               # [B, T, 9]
        B, T = head_pose.shape[:2]
        
        output = model(head_pose, betas)
        pred_vertices = output['pred_verts'] + world_transl.reshape(1, -1, 1, 3).cuda()
        pred_vertices = pred_vertices.reshape(-1, 6890, 3)
        
        writer = imageio.get_writer(
            'test.mp4', mode='I', format='FFMPEG', fps=30, macro_block_size=1
        )
        
        gt_vertices
        renderer.set_ground(length=10, center_x=0., center_z=0.)
        # R, T = look_at_view_transform(
        #     eye=torch.tensor([[2.0, 1.0, 7.0]], dtype=torch.float32),
        #     at=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        #     up=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        #     device='cuda'
        # )
        
        R, T = look_at_view_transform(
            dist=7,
            elev=60,
            azim=30,
            up=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            device='cuda'
        )
        
        for t, (pred, gt) in tqdm(enumerate(zip(pred_vertices, gt_vertices)), desc='rendered results...'):
            cameras = renderer.create_camera(R, T)
            faces = renderer.faces.clone().squeeze(0)
            img_gt = renderer.render_with_ground(gt_vertices[[t]], faces, colors, cameras, None)
            img_pred = renderer.render_with_ground(pred_vertices[[t]], faces, colors, cameras, None)
            img_glob = np.hstack([img_gt, img_pred])
            writer.append_data(img_glob[..., ::-1])
        
        writer.close()
        exit()
    