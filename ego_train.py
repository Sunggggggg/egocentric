import torch
from pathlib import Path
from loguru import logger
from smplx import SMPLH
from tqdm import tqdm

import configs.constant as _C
from lib.dataset.dataclass import TrainingData
from lib.dataset.egocentric_dataloader import get_loader
from torch.utils.tensorboard import SummaryWriter
from lib.utils import losses
from lib.utils.train_utils import *
from lib.models.ego_tokenhmr import Model
from lib.utils.rotation_utils import *
from lib.utils.eval_utils import eval_pose_hmr
from lib.utils.pose_visualize import visualize_mesh

# ----------- Hyperparam. ----------- #
OUT_DIR = 'exp'
NAME = 'only_headnet'
save_dir = os.path.join(OUT_DIR, 'egohmr', NAME)
os.makedirs(save_dir, exist_ok=True)
_C.TRAIN.batch_size = 1

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

writer = SummaryWriter(save_dir)
best_scores = init_best_scores()
loss_config = losses.LossConfig()
bce_loss_function = torch.nn.BCEWithLogitsLoss()

nb_iter = 0
for epoch in range(10000) :
    for batch in tqdm(train_loader) :
        batch = {k: v for k, v in batch.__dict__.items()}
        batch = get_device(batch)

        # ----------- Model forward ----------- #
        head_pose = make_input(batch)               # [B, T, 9]
        B, T = head_pose.shape[:2]
        
        output = model(head_pose)
        pred_jnts = output['pred_jnts'][..., :22, :]    # [B, T, 22, 3]
        pred_pose = output['pred_pose']             # [B, T, 21, 6]
        pred_root = output['global_orient']         # [B, T, 6]
        pred_contact = output['contact']            # [B, T, 21]
        
        pred_jnts = (pred_jnts - pred_jnts[..., [0], :])[..., 1:, :]
        
        # ----------- Loss ----------- #
        gt_jnts = batch['joints_wrt_world']                                 # [B, T, 21, 3]
        gt_pose = convert_quat_to_6d(batch['body_quats']).reshape(B, T, -1) # [B, T, 21*6]
        gt_root = convert_quat_to_6d(batch['T_world_root'][..., :4])        # [B, T, 6]
        gt_contact = batch['contacts']              # [B, T, 21]

        gt_jnts = gt_jnts - gt_jnts[..., [0], :]
        
        
        loss_jnts = F.mse_loss(pred_jnts, gt_jnts)
        loss_pose = F.mse_loss(pred_pose, gt_pose)
        loss_root = F.mse_loss(pred_root, gt_root)
        loss_contact = bce_loss_function(pred_contact, gt_contact)

        
        loss = loss_config.POSE_LOSS_WT * loss_pose + \
                loss_config.JNT_LOSS_WT * loss_jnts + \
                loss_config.JNT_LOSS_WT * loss_root + \
                loss_config.CONTACT_LOSS_WT * loss_contact
        loss *= loss_config.LOSS_WT
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        err_list['tr/curr_pose_recons'] += loss_config.POSE_LOSS_WT * loss_pose.item()
        err_list['tr/curr_jnt_recons'] += loss_config.JNT_LOSS_WT * loss_jnts.item()
        err_list['tr/curr_root'] += loss_config.JNT_LOSS_WT * loss_root.item()
        err_list['tr/curr_contact'] += loss_config.CONTACT_LOSS_WT * loss_contact.item()
        err_list['tr/curr_loss'] += loss_config.LOSS_WT * loss.item()
        
        # ----------- Logging ----------- #
        nb_iter += 1
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
            best_scores = eval_pose_hmr(val_loader, model, writer, nb_iter, save_dir, best_scores)
        
        if nb_iter % _C.TRAIN.SAVE_ITER == 0 :
            with torch.no_grad():
                world_orient = quaternion_to_axis_angle(batch['T_world_root'][..., :4]) # [B, T, 3]
                body_pose = quaternion_to_axis_angle(batch['body_quats'])               # [B, T, 21, 3]        
                betas = batch['betas'][..., :10]
                smpl_output = smplh(
                    global_orient=world_orient.view(-1, 3),
                    body_pose=body_pose.view(-1, 63),
                    betas=betas.view(-1, 10),
                )
                
                batch['body_vertices'] = smpl_output.vertices.reshape(B, T, -1, 3)
                rend_img = visualize_mesh(batch, output, save_dir, nb_iter)
            writer.add_image(f'rendered_{nb_iter:06}.jpg', rend_img.transpose(2, 0, 1), nb_iter)
            save_model(model, save_dir, nb_iter)