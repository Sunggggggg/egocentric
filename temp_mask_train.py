import os
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
from smplx import SMPLH
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


import configs.constant as _C
from lib.dataset.codebook_dataloader import get_dataloader
from lib.models.temp_mae_tokenhmr import TemporalTokenHMR
from lib.utils import losses
from lib.utils.train_utils import get_optimizer, reset_err_list, init_best_scores, save_model, get_device
from lib.utils.eval_utils import eval_pose_vqvae
from lib.utils.pose_visualize import visualize_mesh
from lib.vis.renderer import Renderer

unflatten = lambda x : x.reshape((_C.TRAIN.batch_size, _C.TRAIN.seqlen) + x.shape[1:])
# ----------- Hyperparam. ----------- #
OUT_DIR = 'exp'
NAME = 'mae_transformer'
save_dir = os.path.join(OUT_DIR, 'codebook', NAME)
os.makedirs(save_dir, exist_ok=True)

# ----------- Dataset ----------- #
train_loader_iter = get_dataloader('train')
val_loader = get_dataloader('test')

# ----------- Model ----------- #
from lib.models.temp_mae_tokenhmr import TemporalTokenHMR
model = TemporalTokenHMR().cuda()

# ----------- Train ----------- #
err_list = reset_err_list('tr')
optimizer, scheduler = get_optimizer(model,)
loss_config = losses.LossConfig()
Loss = losses.PoseReConsLoss(loss_config)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")
writer = SummaryWriter(save_dir)
best_scores = init_best_scores()

width, height = 500, 500
focal_len = (width ** 2 + height ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
faces = SMPLH(_C.SMPL.SMPLH_MODEL_PATH, num_betas=10, ext='pkl').faces
renderer = Renderer(width, height, focal_len, device, faces)

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
    B, T = gt_pose_body_6d.shape[:2]
    code_idx_list = []
    with torch.no_grad():
        for t in range(T):
            code_idx = model.tokenizer.encode(gt_pose_body_6d[:, t])
            code_idx_list.append(code_idx)
        indices = torch.cat(code_idx_list, dim=0)    # [B, T, 160]
    
    # code_idx = code_idx.reshape(B, -1).detach().cpu().numpy()
    # for b in range(B):
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(code_idx[b], bins=len(code_idx[b]), color='skyblue', edgecolor='black')
    #     plt.xlabel('Value', fontsize=12)
    #     plt.ylabel('Frequency', fontsize=12)
    #     plt.savefig('test.png')
    #     plt.close()
    
    output = model(gt_pose_body_6d, nb_iter)
    mask, predicted_indices = output['mask'], output['code_idx']
    loss = criterion(
        predicted_indices.float().flatten(0)[mask.flatten(0).to(torch.bool)],
        indices.float().flatten(0)[mask.flatten(0).to(torch.bool)],
    )
    print(loss)
    exit()
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
        rend_img = visualize_mesh(batch, output, save_dir, nb_iter, renderer)
        writer.add_image(f'rendered_{nb_iter:06}.jpg', rend_img.transpose(2, 0, 1), nb_iter)
        save_model(model, save_dir, nb_iter)