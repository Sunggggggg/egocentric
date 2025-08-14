from __future__ import annotations

import torch
from torch import Tensor

from .tensor_dataclass import TensorDataclass

import dataclasses
class TrainingData(TensorDataclass):

    T_world_root: Tensor
    contacts: Tensor
    betas: Tensor
    body_quats: Tensor
    T_cpf_tm1_cpf_t: Tensor
    T_world_cpf: Tensor
    height_from_floor: Tensor
    joints_wrt_cpf: Tensor
    joints_wrt_world: Tensor
    mask: Tensor
    hand_quats: Tensor

def collate_dataclass(batch: list) -> TrainingData:

    if not batch:
        raise ValueError("batch is empty")
        
    elem = batch[0]
    if not dataclasses.is_dataclass(elem):
        raise TypeError("collate_dataclass can only be used with dataclasses")
    
    collated_data = {}
    for f in dataclasses.fields(elem):
        
        key = f.name
        first_item_val = getattr(elem, key)

        if first_item_val is None:
            if all(getattr(b, key) is None for b in batch):
                collated_data[key] = None
            else:
                raise TypeError(f"Field '{key}' contains a mix of Tensors and None.")
        elif isinstance(first_item_val, torch.Tensor):
            collated_data[key] = torch.stack([getattr(b, key) for b in batch])
        else:
            collated_data[key] = [getattr(b, key) for b in batch]

    return type(elem)(**collated_data)
