import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from smplx import SMPLH

import configs.constant as _C
from lib.utils import rotation_conversions

class MixedTrainDataset(Dataset):
    def __init__(self, ds_list, partition, split):

        self.ds_list = ds_list
        partition = [float(part) for part in partition]
        self.partition = np.array(partition).cumsum()
        
        self.datasets = [PoseDataset(ds, split) for ds in ds_list]
        self.length = max([len(ds) for ds in self.datasets])

    def __getitem__(self, index):
            p = np.random.rand()
            for i in range(len(self.ds_list)):
                if p <= self.partition[i]:
                    return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length

class ValDataset(Dataset):
    def __init__(self, dataset_list, split, seqlen=_C.TRAIN.seqlen):
        super().__init__()
        self.seqlen = seqlen
        self.all_windows = []
        self.pose_body_data = []
        self.betas_data = []
        
        for dt in dataset_list:
            data_dict = np.load(os.path.join(_C.DATA.AMASS_ROOT, split, f'{split}_{dt}.npz'))
            pose_body = data_dict['pose_body']
            betas = data_dict['betas']
            names = data_dict['name']
            
            current_idx = len(self.pose_body_data)
            video_indices = self._group_data_by_video(names)
            for video_name in video_indices:
                start_idx, end_idx = video_indices[video_name]
                video_length = end_idx - start_idx
                
                for i in range(start_idx, end_idx - self.seqlen + 1, self.seqlen):
                    self.all_windows.append(current_idx + i)
                    
            self.pose_body_data.append(pose_body)
            self.betas_data.append(betas)
            
            print(f"Processing {dt} for {split} / total window {len(self.all_windows)} (so far)")

        self.pose_body = np.concatenate(self.pose_body_data, axis=0)
        self.betas = np.concatenate(self.betas_data, axis=0)
        
        print(f"Finished loading validation data. Total windows: {len(self.all_windows)}")

        self.smpl_model = SMPLH(_C.SMPL.SMPLH_MODEL_PATH, num_betas=10, 
                                ext='pkl', batch_size=seqlen)
    
    def _group_data_by_video(self, names):
        video_indices = {}
        if len(names) == 0:
            return video_indices

        current_name = names[0]
        start_idx = 0
        
        for i, name in enumerate(names):
            if name != current_name:
                video_indices[current_name] = (start_idx, i)
                current_name = name
                start_idx = i

        video_indices[current_name] = (start_idx, len(names))
        return video_indices

    def __len__(self):
        return len(self.all_windows)

    def __getitem__(self, idx):
        item = {}
        window_start = self.all_windows[idx]
        window_end = window_start + self.seqlen
        
        pose_body_aa = self.pose_body[window_start:window_end]
        pose_body_aa = torch.from_numpy(pose_body_aa).float().reshape(-1, 63)
        item['pose_body_aa'] = pose_body_aa.clone()
        
        body_model = self.smpl_model(body_pose=pose_body_aa)
        item['body_vertices'] = body_model.vertices.detach().float()    # [T, 6890, 3]
        item['body_joints'] = body_model.joints.detach().float()        # [T, 17, 3]
        
        pose_body_rot = rotation_conversions.axis_angle_to_matrix(pose_body_aa.view(-1, 21, 3))
        pose_body_6d = rotation_conversions.matrix_to_rotation_6d(pose_body_rot)
        item['pose_body_rot'] = pose_body_rot.clone()
        item['pose_body_6d'] = pose_body_6d.clone()
        
        return item


class PoseDataset(Dataset):
    def __init__(self, dt, split, seqlen=_C.TRAIN.seqlen, step=8):
        data_dict = np.load(os.path.join(_C.DATA.AMASS_ROOT, split, f'{split}_{dt}.npz'))
        self.pose_body = data_dict['pose_body']
        self.betas = data_dict['betas']
        self.names = data_dict['name']
        self.seqlen = seqlen
        
        self.all_windows_start_indices = []
        video_indices = self._group_data_by_video(self.names)
        
        for video_name in video_indices:
            start_idx, end_idx = video_indices[video_name]
            video_length = end_idx - start_idx
    
            if video_length < self.seqlen:
                continue
   
            for i in range(start_idx, end_idx - self.seqlen + 1, step):
                self.all_windows_start_indices.append(i)
        
        print(f"Processing {dt} for {split} / total window {len(self.all_windows_start_indices)}, total seqlen : {len(self.names)}")
        
        self.smpl_model = SMPLH(_C.SMPL.SMPLH_MODEL_PATH, num_betas=10, 
                                ext='pkl', batch_size=seqlen)
        
    def _group_data_by_video(self, names):
        video_indices = {}
        if len(names) == 0:
            return video_indices
        current_name = names[0]
        start_idx = 0
        
        for i, name in enumerate(names):
            if name != current_name:
                video_indices[current_name] = (start_idx, i)
                current_name = name
                start_idx = i

        video_indices[current_name] = (start_idx, len(names))
        return video_indices
    
    def __len__(self):
        return len(self.all_windows_start_indices)

    def __getitem__(self, idx):
        item = {}
        
        window_start = self.all_windows_start_indices[idx]
        window_end = window_start + self.seqlen
        
        pose_body_aa = self.pose_body[window_start:window_end]
        pose_body_aa = torch.from_numpy(pose_body_aa).float().reshape(-1, 63)
        item['pose_body_aa'] = pose_body_aa.clone()
        
        body_model = self.smpl_model(body_pose=pose_body_aa)
        item['body_vertices'] = body_model.vertices.detach().float()    # [T, 6890, 3]
        item['body_joints'] = body_model.joints.detach().float()        # [T, 17, 3]
        
        pose_body_rot = rotation_conversions.axis_angle_to_matrix(pose_body_aa.view(-1, 21, 3))
        pose_body_6d = rotation_conversions.matrix_to_rotation_6d(pose_body_rot)
        item['pose_body_rot'] = pose_body_rot.clone()
        item['pose_body_6d'] = pose_body_6d.clone()
        
        return item


def get_dataloader(split):
    if split == 'train' :
        ds_list = _C.DATA.TRAINLIST.split('_')
        partition = [1] if len(ds_list) == 1 else _C.DATA.TRAIN_PART.split('_')
    elif split == 'test' :
        ds_list = _C.DATA.TESTLIST.split('_')
        partition = [1/len(ds_list)]*len(ds_list)
    
    if split == 'train':
        dataset = MixedTrainDataset(ds_list, partition, split)
        dataloader = DataLoader(dataset, batch_size=_C.TRAIN.batch_size, shuffle=True, num_workers=4, drop_last=True)
    else :
        dataset = ValDataset(ds_list, split)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    if split == 'train':
        return cycle(dataloader)
    else:
        return dataloader
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
