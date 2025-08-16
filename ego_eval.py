import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import cv2
import numpy as np
import torch
from tqdm import tqdm
import dataclasses
import csv
import imageio
from smplx import SMPLH

import configs.constant as _C
from lib.models import load_egotokenhmr
from lib.dataset.egocentric_dataloader import AmassHdf5Dataset
from lib.dataset.dataclass import collate_dataclass, TrainingData
from lib.utils.rotation_utils import quaternion_to_axis_angle
from lib.utils.eval_utils import compute_similarity_transform


def move_dataclass_to_device(data_object: Any, device: torch.device) -> Any:
    if isinstance(data_object, torch.Tensor):
        return data_object.to(device)
    elif isinstance(data_object, (list, tuple)):
        return type(data_object)([move_dataclass_to_device(item, device) for item in data_object])
    elif isinstance(data_object, dict):
        return {key: move_dataclass_to_device(value, device) for key, value in data_object.items()}
    elif dataclasses.is_dataclass(data_object):
        kwargs = {}
        for field in dataclasses.fields(data_object):
            field_value = getattr(data_object, field.name)
            kwargs[field.name] = move_dataclass_to_device(field_value, device)
        return type(data_object)(**kwargs)
    else:
        return data_object


class Evaluator:
    def __init__(self, metrics: list, dataset: str, smpl_model: SMPLH):
        self.dataset_name = dataset
        self.smpl_model = smpl_model.eval()
        self.total_mpjpe = 0.0
        self.total_pampjpe = 0.0
        
        self.num_samples = 0
        self.results = {metric: [] for metric in metrics}
        print(f"Initialized Evaluator for {self.dataset_name}")

    def __call__(self, output: Dict[str, torch.Tensor], gt: TrainingData):
        
        if output['final_keypoints_3d'] is None or gt.joints_wrt_world is None:
            print("Skipping metric calculation due to missing joint data.")
            return
        
        B, T = gt.betas.shape[:2]
        eval_world = False
        if eval_world:
            # Pred -------------------------------------------------------------------------------------------------------------------------------
            pred_keypoints_3d = output['final_keypoints_3d']
            
            
            pred_verts_flat = output['final_vertices']
            right_eye_pred = (pred_verts_flat[:, 6260, :] + pred_verts_flat[:, 6262, :]) / 2.0
            left_eye_pred = (pred_verts_flat[:, 2800, :] + pred_verts_flat[:, 2802, :]) / 2.0
            cpf_pos_pred_local = (right_eye_pred + left_eye_pred) / 2.0

            cpf_pos_pred_local = cpf_pos_pred_local.reshape(B*T, 3)
            
            gt_cpf_translation_world = gt.T_world_cpf[..., 4:7].reshape(B*T, 3)
            translation_vector = gt_cpf_translation_world - cpf_pos_pred_local
            
            translation_vector_expanded = translation_vector.unsqueeze(1)
            
            pred_joints_3d_aligned = pred_keypoints_3d[:, 1:, :] + translation_vector_expanded
            # ------------------------------------------------------------------------------------------------------------------------------------
            
            # GT ---------------------------------------------------------------------------------------------------------------------------------
            gt_joints_3d_aligned = gt.joints_wrt_world.reshape(B*T, 21, 3)
            # ------------------------------------------------------------------------------------------------------------------------------------
        else:
            pred_joints_3d_aligned = (output['final_keypoints_3d'] - output['final_keypoints_3d'][:, [0], :])[:, 1:, :]
            gt_joints_3d_aligned = gt.joints_wrt_world.reshape(B*T, 21, 3) - gt.T_world_root[..., 4:].reshape(B*T, 3).unsqueeze(1)
        
        
        pred_joints_3d_np = pred_joints_3d_aligned.detach().cpu().numpy()  # [B*T, J, 3]
        gt_joints_for_eval_np = gt_joints_3d_aligned.detach().cpu().numpy() # [B*T, J, 3]

        # MPJPE ---------------------------------------------------------------------------------------------------------------------------------
        mpjpe_batch = np.linalg.norm(pred_joints_3d_np - gt_joints_for_eval_np, axis=-1).mean(axis=-1) * 1000
        self.total_mpjpe += mpjpe_batch.sum()
        # ---------------------------------------------------------------------------------------------------------------------------------------
        
        # PA-MPJPE ------------------------------------------------------------------------------------------------------------------------------
        pa_pred_joints_3d_np = compute_similarity_transform(pred_joints_3d_np, gt_joints_for_eval_np)
        pampjpe_batch = np.linalg.norm(pa_pred_joints_3d_np - gt_joints_for_eval_np, axis=-1).mean(axis=-1) * 1000
        # ---------------------------------------------------------------------------------------------------------------------------------------
        
        self.total_pampjpe += pampjpe_batch.sum()
        self.num_samples += B * T

    def record_mpjpe_to_csv(self, header, batch_data, output, idx, ) -> None:
        file_path = Path('./results/result.csv')
        file_exists = os.path.isfile(file_path)
        if header is None:
            header = ['index', 'MPJPE', 'PA-MPJPE']

        #########################################################################################################################
        total_mpjpe, total_pampjpe = 0., 0.
        B, T = batch_data.betas.shape[:2]
        
        pred_joints_3d_aligned = (output['final_keypoints_3d'].reshape(B, T, 22, 3)[0, ...] - output['final_keypoints_3d'].reshape(B, T, 22, 3)[0, :, [0], :])[:, 1:, :]
        gt_joints_3d_aligned = batch_data.joints_wrt_world[0, ...].squeeze() - batch_data.T_world_root[0, :, 4:].unsqueeze(1)
        
        pred_joints_3d_np = pred_joints_3d_aligned.detach().cpu().numpy()  # [B*T, J, 3]
        gt_joints_for_eval_np = gt_joints_3d_aligned.detach().cpu().numpy() # [B*T, J, 3]

        
        mpjpe_batch = np.linalg.norm(pred_joints_3d_np - gt_joints_for_eval_np, axis=-1).mean(axis=-1) * 1000
        total_mpjpe += mpjpe_batch.sum()
        
        pa_pred_joints_3d_np = compute_similarity_transform(pred_joints_3d_np, gt_joints_for_eval_np)
        pampjpe_batch = np.linalg.norm(pa_pred_joints_3d_np - gt_joints_for_eval_np, axis=-1).mean(axis=-1) * 1000

        total_pampjpe += pampjpe_batch.sum()
        #########################################################################################################################

        mpjpe_value = total_mpjpe / T
        pampjpe_value = total_pampjpe / T
        


        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)

            if isinstance(idx, int):
                idx = [idx]
            if isinstance(mpjpe_value, float):
                mpjpe_value = [mpjpe_value]
            if isinstance(pampjpe_value, float):
                pampjpe_value = [pampjpe_value]

            for idx_val, mpjpe_val, pampjpe_val in zip(idx, mpjpe_value, pampjpe_value):
                writer.writerow([f"{idx_val:05d}", mpjpe_val, pampjpe_val])

        print(f"Values recorded to: {file_path}")
    
    def log(self):
        print(f"\n--- {self.dataset_name} Evaluation Results ---")
        if self.num_samples == 0:
            print("No samples processed yet.")
        else:
            avg_mpjpe = self.total_mpjpe / self.num_samples
            avg_pampjpe = self.total_pampjpe / self.num_samples
            print(f"Total Samples: {self.num_samples}")
            print(f"Average MPJPE: {avg_mpjpe:.2f} mm")
            print(f"Average PAMPJPE: {avg_pampjpe:.2f} mm")
        print("-----------------------------------")

    def get_metrics_dict(self) -> Dict[str, float]:
        if self.num_samples == 0:
            return {'MPJPE': float('nan'), 'PAMPJPE': float('nan')}
        
        avg_mpjpe = self.total_mpjpe / self.num_samples
        avg_pampjpe = self.total_pampjpe / self.num_samples
        
        metrics = {
            'MPJPE': avg_mpjpe,
            'PAMPJPE': avg_pampjpe,
        }
        return metrics


def render_predictions(args, dataset_name, batch_data: TrainingData, output, smpl_model, mesh_ren, mesh_renderer_henu, idx):

    vis_mode = 'gt_init_final'  # 'gt_pred', 'gt_init_final'
    vis_traj = True            # True: Pose Only, False: Pose & Trajectory
    
    store_mp4 = False
    store_gif = True
    store_png = False
    store_obj = False

    result_dir = './results'

    num_img = 2
    if vis_mode == 'gt_init_final':
        num_img = 3
    
    mesh_renderer = mesh_ren
    if vis_traj:
        mesh_renderer = mesh_renderer_henu
    
    B, T = batch_data.betas.shape[:2]
    
    # GT ---------------------------------------------------------------------------------------------------------------------------------
    gt_global_orient_aa = quaternion_to_axis_angle(batch_data.T_world_root[..., :4].reshape(B * T, 4)).squeeze(1)
    gt_body_pose_aa = quaternion_to_axis_angle(batch_data.body_quats.reshape(B * T, -1, 4)).reshape(B * T, -1)
    gt_betas = batch_data.betas.reshape(B*T, -1)[:, :10]
    gt_smpl_input_params = {
        'global_orient': gt_global_orient_aa,
        'body_pose': gt_body_pose_aa,
        'betas': gt_betas,
    }
    gt_smpl_output = smpl_model(**{k: v for k,v in gt_smpl_input_params.items()}, pose2rot=True)
    gt_vertices = gt_smpl_output.vertices.reshape(B, T, -1, 3)
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # Pred -------------------------------------------------------------------------------------------------------------------------------
    pred_verts_flat = output['final_vertices']
    pred_vertices = pred_verts_flat.reshape(B, T, -1, 3)
    
    if vis_mode == 'gt_init_final':
        init_verts_flat = output['init_vertices']
        init_vertices = init_verts_flat.reshape(B, T, -1, 3)
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    
    if vis_traj:
        # GT ---------------------------------------------------------------------------------------------------------------------------------
        gt_pelvis_translation = batch_data.T_world_root[..., 4:7].reshape(B, T, 3)
        gt_pelvis_translation_expanded = gt_pelvis_translation.unsqueeze(2)
        gt_vertices = gt_vertices + gt_pelvis_translation_expanded
        # ------------------------------------------------------------------------------------------------------------------------------------

        # Pred -------------------------------------------------------------------------------------------------------------------------------
        right_eye_pred = (pred_verts_flat[:, 6260, :] + pred_verts_flat[:, 6262, :]) / 2.0
        left_eye_pred = (pred_verts_flat[:, 2800, :] + pred_verts_flat[:, 2802, :]) / 2.0
        cpf_pos_pred_local = (right_eye_pred + left_eye_pred) / 2.0

        cpf_pos_pred_local = cpf_pos_pred_local.reshape(B, T, 3)
        
        gt_cpf_translation_world = batch_data.T_world_cpf[..., 4:7].reshape(B, T, 3)
        translation_vector = gt_cpf_translation_world - cpf_pos_pred_local
        
        translation_vector_expanded = translation_vector.unsqueeze(2)
        pred_vertices = pred_vertices + translation_vector_expanded
        
        if vis_mode == 'gt_init_final':
            right_eye_init = (init_verts_flat[:, 6260, :] + init_verts_flat[:, 6262, :]) / 2.0
            left_eye_init = (init_verts_flat[:, 2800, :] + init_verts_flat[:, 2802, :]) / 2.0
            cpf_pos_init_local = (right_eye_init + left_eye_init) / 2.0

            cpf_pos_init_local = cpf_pos_init_local.reshape(B, T, 3)

            translation_vector = gt_cpf_translation_world - cpf_pos_init_local
            
            translation_vector_expanded = translation_vector.unsqueeze(2)
            init_vertices = init_vertices + translation_vector_expanded
        # ------------------------------------------------------------------------------------------------------------------------------------


    temp_pred_img = mesh_renderer(pred_vertices[0, 0].float().unsqueeze(0))
    height, width = temp_pred_img.shape[2], temp_pred_img.shape[3] * num_img
    

    if store_gif or store_mp4:
        fps = 30
        if store_mp4:
            video_path = os.path.join(result_dir, f'render_{dataset_name}_{idx:05d}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if store_gif:
            gif_path = os.path.join(result_dir, f'render_{dataset_name}_{idx:05d}.gif')
            frames_for_gif = []
        
        
        for t in range(T):
            gt_vertices_t = gt_vertices[0, t].unsqueeze(0)
            gt_rendered_image_t = mesh_renderer(gt_vertices_t.float())
            
            pred_vertices_t = pred_vertices[0, t].unsqueeze(0)
            rendered_image_t = mesh_renderer(pred_vertices_t.float())
            combined_frame = torch.cat([gt_rendered_image_t[0], rendered_image_t[0]], dim=2)
            
            if vis_mode == 'gt_init_final':
                init_vertices_t = init_vertices[0, t].unsqueeze(0)
                init_rendered_image_t = mesh_renderer(init_vertices_t.float())
                combined_frame = torch.cat([gt_rendered_image_t[0], init_rendered_image_t[0], rendered_image_t[0]], dim=2)


            frame_np = combined_frame.cpu().numpy().transpose(1, 2, 0) * 255
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            if store_mp4:
                video_writer.write(frame_bgr)
            if store_gif:
                frames_for_gif.append(frame_np)
                
            if t == T // 2:
                if store_png:
                    save_path_img = os.path.join(result_dir, f'render_{dataset_name}_{idx:05d}.png')
                    cv2.imwrite(save_path_img, frame_bgr)
                if store_obj:
                    gt_save_path_obj = os.path.join(result_dir, f'render_{dataset_name}_{idx:05d}_GT.obj')
                    gt_vertices_np = gt_vertices_t[0].cpu().numpy()
                    faces_np = smpl_model.faces.astype(np.int32)
                    save_obj(gt_save_path_obj, gt_vertices_np, faces_np)
                    
                    pred_save_path_obj = os.path.join(result_dir, f'render_{dataset_name}_{idx:05d}_PRED.obj')
                    pred_vertices_np = pred_vertices_t[0].cpu().numpy()
                    faces_np = smpl_model.faces.astype(np.int32)
                    save_obj(pred_save_path_obj, pred_vertices_np, faces_np)

                    if vis_mode == 'gt_init_final':
                        init_save_path_obj = os.path.join(result_dir, f'render_{dataset_name}_{idx:05d}_INIT.obj')
                        init_vertices_np = init_vertices_t[0].cpu().numpy()
                        faces_np = smpl_model.faces.astype(np.int32)
                        save_obj(init_save_path_obj, init_vertices_np, faces_np)
        if store_mp4:
            video_writer.release()
            print(f"Saved video to {video_path}")
        if store_gif:
            imageio.mimsave(gif_path, frames_for_gif, fps=fps)
            print(f"Saved GIF to {gif_path}")
    return None


def save_obj(filepath: str, vertices: np.ndarray, faces: np.ndarray):
    """
    주어진 vertices와 faces를 .obj 파일로 저장합니다.
    """
    with open(filepath, 'w') as f:
        # Vertices 쓰기
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        # Faces 쓰기 (OBJ는 1-based index 사용)
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')
            

def run_eval(model: torch.nn.Module, model_cfg: Any, device: torch.device, args: argparse.Namespace):
    smpl = SMPLH(_C.SMPL.SMPLH_MODEL_PATH).to(device=device)
    
    HDF5_PATH = Path(_C.DATA.HDF5_PATH)
    FILE_LIST_PATH = Path(_C.DATA.FILE_LIST_PATH)
    SUBSEQ_LEN = 128
    
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        collate_fn=collate_dataclass,
        pin_memory=False
    )

    metrics = ['MPJPE', 'PAMPJPE']

    evaluator = Evaluator(
        metrics=metrics,
        dataset=args.current_dataset,
        smpl_model=smpl
    )

    pbar = tqdm(dataloader, desc=f"Evaluating {args.current_dataset}")
    total_iters_done = 0

    with torch.no_grad():
        for i, batch_data in enumerate(pbar):
            batch_data = move_dataclass_to_device(batch_data, device)

            try:
                out = model(batch_data)
            except RuntimeError as e:
                print(f"RuntimeError during model forward pass for batch {i}: {e}")
                raise e

            evaluator(out, batch_data)
            total_iters_done = i + 1

            if total_iters_done:
                evaluator.log()
                evaluator.record_mpjpe_to_csv(None, batch_data, out, total_iters_done)
                render_predictions(args, args.current_dataset, batch_data, out, smpl, mesh_renderer, mesh_renderer_henu, total_iters_done)

    evaluator.log()
    error = None
    metrics_dict = evaluator.get_metrics_dict()
    save_eval_result(args.results_file, metrics_dict, args.checkpoint, args.current_dataset, error=error, iters_done=total_iters_done, exp_name=args.exp_name)

def save_eval_result(csv_path: str, metric_dict: Dict[str, float], checkpoint_path: str, dataset_name: str, error: Optional[str] = None, iters_done: Optional[int] = None, exp_name: Optional[str] = None):
    timestamp = pd.Timestamp.now()
    exists = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    metric_names = list(metric_dict.keys())
    metric_values = [float(f'{value:.2f}') for value in metric_dict.values()]
    N = len(metric_names)

    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            error=[error] * N,
            iters_done=[iters_done] * N,
        ),
        index=list(range(N)),
    )
    df.to_csv(csv_path, mode="a", header=not exists, index=False)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    # Meta data
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--results_file', type=str, default='eval_regression.csv')
    parser.add_argument('--checkpoint', type=str, default='exp/egohmr/only_headnet/030000_net.pth')
    parser.add_argument('--dataset', type=str, default='AMASS_TEST') 
    
    # Dataloader
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False)

    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--log_freq', type=int, default=10)
    
    
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    exp_name = 'eval' if args.exp_name is None else args.exp_name
    results_dir = f'./results/release/{exp_name}'
    render_dir = f'{results_dir}/render'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    args.render_dir = render_dir
    args.results_file = os.path.join(results_dir, args.results_file)

    model = load_egotokenhmr(checkpoint_path=args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded on {device}!')
    print('Using checkpoint:', args.checkpoint)
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)

    for dataset_name_str in args.dataset.split(','):
        current_dataset_name = dataset_name_str.strip()
        print(f"\n--- Evaluating on {current_dataset_name} ---")
        args.current_dataset = current_dataset_name
        run_eval(model, device, args)

if __name__ == '__main__':
    main()