import os
import torch
import torch.optim as optim
from lib.data.transforms._so3 import SO3

def get_device(data, device='cuda:0'):
    for k, v in data.items() :
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device).float()
    return data

def get_optimizer(model, lr=2e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, 
                            betas=(0.9, 0.99), weight_decay=0.00001)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75000, 100000], 
                                                     gamma=0.05)
    
    return optimizer, scheduler

def reset_err_list(type='tr'):
    err_list = {f'{type}/curr_pose_recons': 0.,
                f'{type}/curr_mesh_recons': 0.,
                f'{type}/curr_jnt_recons': 0.,
                f'{type}/curr_perplexity': 0.,
                f'{type}/curr_commit': 0.,
                f'{type}/curr_root': 0.,
                f'{type}/curr_contact': 0.,
                f'{type}/curr_pa_jnt_recons': 0.,}
    if type == 'tr':
        err_list.update({
            f'{type}/curr_loss': 0.
        })
    return err_list

def init_best_scores():
    best_scores = {
        f'val/best_iter': 0,
        f'val/best_val_score': 1e8,
        f'val/best_mesh_recons': 1e8,
        f'val/best_jnt_recons': 1e8,
    }
    return best_scores

def save_model(model, save_dir, iter):
    save_dict = {
        'net': model.state_dict(),
    }
    torch.save(save_dict, os.path.join(save_dir, f'{iter:06d}_net.pth'))
    

def convert_quat_to_6d(quat):
    rot_mat = SO3(wxyz=quat).as_matrix()
    return rot_mat[..., :2, :].reshape(*quat.shape[:-1], 6)

def make_input(batch):
    head_rotation = convert_quat_to_6d(batch['T_world_cpf'][..., :4])  # [B, T, 6]
    head_translation = batch['T_world_cpf'][..., 4:]                       # [B, T, 3]
    head_pose = torch.cat([head_rotation, head_translation], dim=-1)
    return head_pose