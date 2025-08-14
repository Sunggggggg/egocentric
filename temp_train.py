import os
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import configs.constant as _C
from lib.dataset.codebook_dataloader import get_dataloader
from lib.models.temp_tokenhmr import TemporalTokenHMR
from lib.utils import losses
from lib.utils.train_utils import get_optimizer, reset_err_list, init_best_scores, save_model, get_device
from lib.utils.eval_utils import eval_pose_vqvae
from lib.utils.pose_visualize import visualize_mesh

unflatten = lambda x : x.reshape((_C.TRAIN.batch_size, _C.TRAIN.seqlen) + x.shape[1:])
# ----------- Hyperparam. ----------- #
OUT_DIR = 'exp'
NAME = 'light_transformer'
save_dir = os.path.join(OUT_DIR, 'codebook', NAME)
os.makedirs(save_dir, exist_ok=True)

# ----------- Dataset ----------- #
train_loader_iter = get_dataloader('train')
val_loader = get_dataloader('test')

# ----------- Model ----------- #
model = TemporalTokenHMR().cuda()

# ----------- Train ----------- #
err_list = reset_err_list('tr')
optimizer, scheduler = get_optimizer(model,)
loss_config = losses.LossConfig()
Loss = losses.PoseReConsLoss(loss_config)
writer = SummaryWriter(save_dir)
best_scores = init_best_scores()

# ----------- Resume ----------- #
# ckpt_file = os.path.join(save_dir, 'best_net.pth')
# if os.path.exists(ckpt_file) :
#     ckpt = torch.load(ckpt_file)
#     model.load_state_dict(ckpt['net'], strict=True)
#     logger.info(f"Load ckpt from : {ckpt_file}")

TOTAL_ITER = 200000
model = model.train()
for nb_iter in tqdm(range(1, TOTAL_ITER + 1)) :
    batch = next(train_loader_iter)
    batch = get_device(batch)
    
    # ----------- Ground truth ----------- #
    gt_pose_body_6d = batch['pose_body_6d']  # [B, T, 21, 6]
    gt_pose_body_aa = batch['pose_body_aa']  # [B, T, 21, 3]
    gt_pose = batch['pose_body_rot']
    gt_mesh = batch['body_vertices']
    gt_jnts = batch['body_joints']
    
    # ----------- Model forward ----------- #
    output, loss_commit, perplexity = model(gt_pose_body_6d, nb_iter)
    
    loss_pose = Loss.forward_pose(gt_pose, output)
    loss_mesh = Loss.forward_mesh(gt_mesh, output)
    loss_jnts = Loss.forward_joints(gt_jnts, output)
    
    loss = loss_config.POSE_LOSS_WT * loss_pose + \
            loss_config.MESH_LOSS_WT * loss_mesh + \
            loss_config.JNT_LOSS_WT * loss_jnts + \
            loss_config.COMMIT_LOSS_WT * loss_commit
    loss *= loss_config.LOSS_WT

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    err_list['tr/curr_pose_recons'] += loss_config.POSE_LOSS_WT * loss_pose.item()
    err_list['tr/curr_mesh_recons'] += loss_config.MESH_LOSS_WT * loss_mesh.item()
    err_list['tr/curr_jnt_recons'] += loss_config.JNT_LOSS_WT * loss_jnts.item()
    err_list['tr/curr_perplexity'] += perplexity.item()
    err_list['tr/curr_commit'] += loss_config.COMMIT_LOSS_WT * loss_commit.item()
    err_list['tr/curr_loss'] += loss_config.LOSS_WT * loss.item()
    
    if nb_iter % _C.TRAIN.PRINT_ITER ==  0 :
        for key, value in err_list.items():
            err_list[key] /= _C.TRAIN.PRINT_ITER
        
        if writer is not None:
            for key, value in err_list.items():
                writer.add_scalar(f'{key}', err_list[key], nb_iter)

        print_str = f'Train. Iter {nb_iter}: lr: {scheduler.get_last_lr()[0]:.5f}'
        for key, value in err_list.items():
            print_str += f'\t{key[7:]}: {value:.5f}'
        logger.info(print_str)
        
        err_list = reset_err_list('tr')

    if nb_iter % _C.TRAIN.VAL_ITER  == 0:
        best_scores = eval_pose_vqvae(val_loader, model, writer, nb_iter, save_dir, best_scores)
    
    if nb_iter % _C.TRAIN.SAVE_ITER == 0 :
        rend_img = visualize_mesh(batch, output, save_dir, nb_iter)
        writer.add_image(f'rendered_{nb_iter:06}.jpg', rend_img.transpose(2, 0, 1), nb_iter)
        save_model(model, save_dir, nb_iter)