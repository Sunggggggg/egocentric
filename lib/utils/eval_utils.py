import os
import numpy as np
import tqdm
import torch
from loguru import logger
import configs.constant as _C
from smplx import SMPLH
from os.path import join
from .train_utils import reset_err_list, get_device, make_input
from lib.utils.rotation_conversions import *

def compute_similarity_transform(S1, S2):
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)

    S1_b_d_n = S1.transpose(0, 2, 1)
    S2_b_d_n = S2.transpose(0, 2, 1)
    
    batch_size, dims = S1_b_d_n.shape[0], S1_b_d_n.shape[1]

    mu1 = S1_b_d_n.mean(axis=2, keepdims=True)
    mu2 = S2_b_d_n.mean(axis=2, keepdims=True)
    X1 = S1_b_d_n - mu1
    X2 = S2_b_d_n - mu2

    var1 = np.sum(X1**2, axis=(1, 2))
    var1 = np.maximum(var1, 1e-9)

    K = X1 @ X2.transpose(0, 2, 1)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.transpose(0, 2, 1)

    Z = np.eye(dims)[None, :, :].repeat(batch_size, axis=0)
    sign_det = np.sign(np.linalg.det(U @ Vh))
    Z[:, -1, -1] = sign_det
    
    R = V @ Z @ U.transpose(0, 2, 1)

    trace_RK = np.einsum('bii->b', R @ K)
    scale = trace_RK / var1

    scale_reshaped = scale[:, None, None]
    t = mu2 - scale_reshaped * (R @ mu1)

    S1_hat_b_d_n = scale_reshaped * (R @ S1_b_d_n) + t
    S1_hat = S1_hat_b_d_n.transpose(0, 2, 1)
    
    return S1_hat

def calculate_pose_reconstruction_error(gt_pose, pred_pose):
    return torch.sqrt(torch.pow(gt_pose-pred_pose, 2).sum(-1)).mean()

def calculate_mesh_reconstruction_error(gt_mesh, pred_mesh):
    return torch.sqrt(torch.pow(gt_mesh-pred_mesh, 2).sum(-1)).mean()

def calculate_jnts_reconstruction_error(gt_jnts, pred_jnts):
    valid_joints = [*range(1,22)] # only body joints
    return torch.sqrt(torch.pow(gt_jnts[:, :,valid_joints]-pred_jnts[:, :,valid_joints], 2).sum(-1)).mean()

@torch.no_grad()
def eval_pose_vqvae(val_loader, net, writer, nb_iter, out_dir, best_scores):
    net.eval()

    err_list = reset_err_list('val')
    for batch_idx, batch in enumerate(tqdm.tqdm(val_loader)):
        batch = get_device(batch)
        
        gt_pose_body_6d = batch['pose_body_6d']     # [1, T, ]
        gt_pose = batch['pose_body_rot']            # 
        gt_mesh = batch['body_vertices']
        gt_jnts = batch['body_joints']
        
        output, loss_commit, perplexity = net(gt_pose_body_6d)
        pred_pose_body_rotmat = output['pred_pose_body_rotmat']
        pred_body_vertices = output['pred_body_vertices']
        pred_body_joints = output['pred_body_joints']
    
        pose_error = calculate_pose_reconstruction_error(gt_pose, pred_pose_body_rotmat)
        mesh_error = calculate_mesh_reconstruction_error(gt_mesh, pred_body_vertices)
        jnt_error = calculate_jnts_reconstruction_error(gt_jnts, pred_body_joints)

        err_list['val/curr_pose_recons'] += pose_error.item()
        err_list['val/curr_mesh_recons'] += mesh_error.item()
        err_list['val/curr_jnt_recons'] += jnt_error.item()
        err_list['val/curr_perplexity'] += perplexity.item()
        err_list['val/curr_commit'] += loss_commit.item()
        
        # if batch_idx % val_disp_iter == 0:
        #     visualize_from_mesh('smplh', batch, output, f'eval_{nb_iter}_{batch_idx}', save_dir)
    
    for key, value in err_list.items():
        err_list[key] /= (batch_idx+1)

    err_list['val/curr_jnt_recons'] *= 1000
    err_list['val/curr_mesh_recons'] *= 1000
    
    curr_score = err_list['val/curr_jnt_recons']
    if curr_score < best_scores['val/best_val_score']:
        best_scores['val/best_val_score'] = curr_score
        save_dict = {
            'net': net.state_dict(),
        }
        best_scores['val/best_iter'] = nb_iter
        best_scores['val/best_jnt_recons']  = err_list['val/curr_jnt_recons']
        best_scores['val/best_mesh_recons'] = err_list['val/curr_mesh_recons']
        torch.save(save_dict, join(out_dir, 'best_net.pth'))
        logger.info(f"Eval. Iter {nb_iter}: !!---> BEST MODEL FOUND <---!! Validation Score - {best_scores['val/best_val_score']:.2f}")
    
    print_str = f'Eval. Iter: {nb_iter} | curr_score: {curr_score:.2f}'
    for key, value in err_list.items():
        print_str += f'\t {key[9:]}: {value:.5f}'
    for key, value in best_scores.items():
        print_str += f'\t {key[4:]}: {value:.5f}'
    logger.info(print_str)
    
    if writer is not None:
        for key, value in best_scores.items():
            writer.add_scalar(f'{key}', best_scores[key], nb_iter)
        for key, value in err_list.items():
            writer.add_scalar(f'{key}', err_list[key], nb_iter)
    
    net.train()
    return best_scores

@torch.no_grad()
def eval_pose_hmr(val_loader, net, device, writer, nb_iter, out_dir, best_scores):
    net.eval()

    err_list = reset_err_list('val')
    for batch_idx, batch in enumerate(tqdm.tqdm(val_loader)):
        batch = {k: v for k, v in batch.__dict__.items()}
        batch = get_device(batch, device)
        
        head_pose = make_input(batch)                           # [B, T, 9]
        output = net(head_pose)
        pred_jnts = output['pred_jnts'][:, :, :21].detach().cpu().numpy()       # [B, T, 21, 3]
        gt_jnts = batch['joints_wrt_world'].detach().cpu().numpy()              # [B, T, J, 3]
        
        pred_jnts = pred_jnts - pred_jnts[..., [0], :]
        gt_jnts = gt_jnts - gt_jnts[..., [0], :]
        
        pred_jnts = pred_jnts.reshape(-1, 21, 3)
        gt_jnts = gt_jnts.reshape(-1, 21, 3)
        
        mpjpe = np.linalg.norm(pred_jnts-gt_jnts, axis=-1).mean()
        S1 = compute_similarity_transform(pred_jnts, gt_jnts)
        pa_mpjpe = np.linalg.norm(S1 - gt_jnts, axis=-1).mean()

        err_list['val/curr_jnt_recons'] += mpjpe
        err_list['val/curr_pa_jnt_recons'] += pa_mpjpe
            
    for key, value in err_list.items():
        err_list[key] /= (batch_idx+1)

    err_list['val/curr_jnt_recons'] *= 1000
    err_list['val/curr_pa_jnt_recons'] *= 1000
    
    curr_score = err_list['val/curr_jnt_recons']
    if curr_score < best_scores['val/best_val_score']:
        best_scores['val/best_val_score'] = curr_score
        save_dict = {
            'net': net.state_dict(),
        }
        best_scores['val/best_iter'] = nb_iter
        best_scores['val/best_jnt_recons']  = err_list['val/curr_jnt_recons']
        best_scores['val/curr_pa_jnt_recons'] = err_list['val/curr_pa_jnt_recons']
        torch.save(save_dict, join(out_dir, 'best_net.pth'))
        logger.info(f"Eval. Iter {nb_iter}: !!---> BEST MODEL FOUND <---!! Validation Score - {best_scores['val/best_val_score']:.2f}")
    
    print_str = f'Eval. Iter: {nb_iter} | curr_score: {curr_score:.2f}'
    for key, value in err_list.items():
        print_str += f'\t {key[9:]}: {value:.5f}'
    for key, value in best_scores.items():
        print_str += f'\t {key[4:]}: {value:.5f}'
    logger.info(print_str)
    
    if writer is not None:
        for key, value in best_scores.items():
            writer.add_scalar(f'{key}', best_scores[key], nb_iter)
        for key, value in err_list.items():
            writer.add_scalar(f'{key}', err_list[key], nb_iter)
    
    net.train()
    return best_scores