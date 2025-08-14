import os
import tqdm
import torch
from loguru import logger
import configs.constant as _C
from smplx import SMPLH
from os.path import join
from .train_utils import reset_err_list, get_device
from lib.utils.rotation_conversions import *

def calculate_pose_reconstruction_error(gt_pose, pred_pose):
    return torch.sqrt(torch.pow(gt_pose-pred_pose, 2).sum(-1)).mean()

def calculate_mesh_reconstruction_error(gt_mesh, pred_mesh):
    return torch.sqrt(torch.pow(gt_mesh-pred_mesh, 2).sum(-1)).mean()

def calculate_jnts_reconstruction_error(gt_jnts, pred_jnts):
    valid_joints = [*range(1,22)] # only body joints
    return torch.sqrt(torch.pow(gt_jnts[:, :,valid_joints]-pred_jnts[:, :,valid_joints], 2).sum(-1)).mean()

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
        err_list[key] /= batch_idx

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