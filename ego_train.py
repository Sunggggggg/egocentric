import torch
from pathlib import Path
from smplx import SMPLH

import configs.constant as _C
from lib.dataset.dataclass import TrainingData
from lib.dataset.egocentric_dataloader import get_loader
from torch.utils.tensorboard import SummaryWriter
from lib.utils import losses
from lib.utils.train_utils import *
from lib.models.ego_tokenhmr import Model
from lib.utils.rotation_utils import *

# ----------- Hyperparam. ----------- #
OUT_DIR = 'exp'
NAME = 'only_headnet'
save_dir = os.path.join(OUT_DIR, 'egohmr', NAME)
os.makedirs(save_dir, exist_ok=True)
_C.TRAIN.batch_size = 1

def train():
    # ----------- Dataset ----------- #
    HDF5_PATH = Path(_C.DATA.HDF5_PATH)
    FILE_LIST_PATH = Path(_C.DATA.FILE_LIST_PATH)
    SUBSEQ_LEN = 128
    train_loader, val_loader = get_loader(HDF5_PATH, FILE_LIST_PATH, SUBSEQ_LEN)
    
    # ----------- Model ----------- #
    model = Model().cuda()
    print(sum([p.numel() for p in model.parameters()])) # 26,741,514

    flatten = lambda x : torch.flatten(x, start_dim=0, end_dim=1)
    unflatten = lambda x, B, T : x.reshape((B, T) + x.shape[1:])
    
    # ----------- SMPLH ----------- # 
    smpl_cfg = {
        'model_path': _C.SMPL.SMPLH_MODEL_PATH,
        'gender': "neutral",
        'num_betas': 10,
        'batch_size': SUBSEQ_LEN * _C.TRAIN.batch_size, 
    }
    smplh = SMPLH(**smpl_cfg).cuda()
    
    # ----------- Trainer ----------- #
    optimizer, scheduler = get_optimizer(model,)
    err_list = reset_err_list('tr')
    
    loss_config = losses.LossConfig()
    Loss = losses.PoseReConsLoss(loss_config)
    
    writer = SummaryWriter(save_dir)
    best_scores = init_best_scores()
    
    for batch in train_loader :
        batch = {k: v for k, v in batch.__dict__.items()}

        # ----------- Model forward ----------- #
        head_pose = make_input(batch).cuda()       # [B, T, 9]
        output = model(head_pose)
        for k, v in output.items():
            print(k, v.shape)
        exit()
        
        
        
        
        
        smpl_input = {
            'global_orient': flatten(matrix_to_axis_angle(output['global_orient'])).view(-1, 3),
            'body_pose':flatten(output['pred_pose']).view(-1, 63), 
            'betas': flatten(batch.betas)[:, :10].cuda(), 
        }
        
        pred_smpl_output = smplh(**smpl_input, pose2rot=True)
        pred_keypoints_3d = pred_smpl_output.joints[:, :22, :] 
        pred_vertices = pred_smpl_output.vertices 
        output = {
            'pred_pose_body_rotmat': axis_angle_to_matrix(output['pred_pose']),
            'pred_body_vertices': pred_vertices,
            'pred_body_joints': pred_keypoints_3d,
        }

        # ----------- Loss ----------- #
        gt_jnts = batch.joints_wrt_world        # 
        gt_pose = convert_quat_to_6d(batch.body_quats)              # [B, T, 21, 6]
        gt_root = convert_quat_to_6d(batch.T_world_root[..., :4])   # [B, T, 6]
       
        
        
                
        loss_pose = Loss.forward_pose(gt_pose, output)
        loss_jnts = Loss.forward_joints(gt_jnts, output)
        
        loss = loss_config.POSE_LOSS_WT * loss_pose + \
                loss_config.JNT_LOSS_WT * loss_jnts 
        loss *= loss_config.LOSS_WT

        

        print(pred_keypoints_3d.shape, pred_vertices.shape)
        
        
        
        
        
        exit()
    return

if __name__ == "__main__":
    train()