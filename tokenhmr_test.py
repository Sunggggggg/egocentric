import configs.constant as _C
import torch
import tqdm
from lib.models.tokenhmr.config import TokenHMRConfig
from lib.models.tokenhmr.vanilla_pose_vqvae import VanillaTokenizer
from lib.dataset.codebook_dataloader import get_dataloader
from lib.utils.train_utils import get_device

def calculate_jnts_reconstruction_error(gt_jnts, pred_jnts):
    valid_joints = [*range(1,22)] # only body joints
    return torch.sqrt(torch.pow(gt_jnts[:,valid_joints]-pred_jnts[:,valid_joints], 2).sum(-1)).mean()

val_loader = get_dataloader('test')
tokenhmr_config = TokenHMRConfig()
model = VanillaTokenizer(tokenhmr_config).cuda()
model_state_dict = torch.load(_C.TOKENHMR.MODEL_PATH)
model.load_state_dict(model_state_dict['net'], strict=True)

mpjpe_list = []
for batch_idx, batch in enumerate(tqdm.tqdm(val_loader)):
    batch = get_device(batch)
    pose_body_6d = batch['pose_body_6d'][0]     # [T, 21, 6]
    body_joints = batch['body_joints'][0]
    x_decoder, loss, perplexity = model(pose_body_6d)
    pred_body_joints = x_decoder['pred_body_joints']
    mpjpe = calculate_jnts_reconstruction_error(body_joints, pred_body_joints)
    mpjpe = mpjpe.detach().cpu()
    mpjpe_list.append(mpjpe)

print(sum(mpjpe_list)/len(mpjpe_list) * 1000)