from pathlib import Path
from typing import Union, Any, Literal, cast

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataclass import TrainingData
from lib.dataset.dataclass import collate_dataclass
import configs.constant as _C

AMASS_SPLITS = {
    "train": ["ACCAD", "BMLhandball", "BMLmovi", "BioMotionLab_NTroje", "CMU",
        "DFaust_67", "DanceDB", "EKUT", "Eyes_Japan_Dataset", "KIT",
        "MPI_Limits", "TCD_handMocap", "TotalCapture",],
    
    "val": ["HumanEva", "MPI_HDM05", "MPI_mosh", "SFU"],
    
    "test": ["Transitions_mocap", "SSM_synced"],}

class AmassHdf5Dataset(torch.utils.data.Dataset[TrainingData]):
    def __init__(
        self,
        hdf5_path: Path,
        file_list_path: Path,
        splits: tuple[
            Literal["train", "val", "test", "test_humor", "just_humaneva"], ...
        ],
        subseq_len: int,
        cache_files: bool,
        slice_strategy: Literal[
            "deterministic", "random_uniform_len", "random_variable_len"
        ],
        min_subseq_len: Union[int, None] = None,
        random_variable_len_proportion: float = 0.3,
        random_variable_len_min: int = 16,
    ) -> None:
        datasets = []
        for split in set(splits):
            datasets.extend(AMASS_SPLITS[split])

        self._slice_strategy: Literal[
            "deterministic", "random_uniform_len", "random_variable_len"
        ] = slice_strategy
        self._random_variable_len_proportion = random_variable_len_proportion
        self._random_variable_len_min = random_variable_len_min
        self._hdf5_path = hdf5_path

        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self._groups = [
                p
                for p in file_list_path.read_text().splitlines()
                if p.partition("/")[0] in datasets
                and cast(
                    h5py.Dataset,
                    cast(h5py.Group, hdf5_file[p])["T_world_root"],
                ).shape[0]
                >= (subseq_len if min_subseq_len is None else min_subseq_len)
            ]
            self._subseq_len = subseq_len
            assert len(self._groups) > 0
            assert len(cast(h5py.Group, hdf5_file[self._groups[0]]).keys()) > 0

            self._approximated_length = (
                sum(
                    cast(
                        h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]
                    ).shape[0]
                    for g in self._groups
                )
                // subseq_len
            )

        # self._cache: dict[str, dict[str, Any]] | None = {} if cache_files else None
        self._cache: Union[dict[str, dict[str, Any]], None] = {} if cache_files else None

    def __getitem__(self, index: int) -> TrainingData:
        group_index = index % len(self._groups)
        slice_index = index // len(self._groups)
        del index

        group = self._groups[group_index]

        hdf5_file = None

        if self._cache is not None:
            if group not in self._cache:
                hdf5_file = h5py.File(self._hdf5_path, "r")
                assert hdf5_file is not None
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
            npz_group = self._cache[group]
        else:
            hdf5_file = h5py.File(self._hdf5_path, "r")
            npz_group = hdf5_file[group]
            assert isinstance(npz_group, h5py.Group)

        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self._subseq_len

        mask = torch.ones(self._subseq_len, dtype=torch.bool)
        if self._slice_strategy == "deterministic":
            valid_start_indices = total_t - self._subseq_len
            start_t = (
                (slice_index * self._subseq_len) % valid_start_indices
                if valid_start_indices > 0
                else 0
            )
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_uniform_len":
            start_t = np.random.randint(0, total_t - self._subseq_len + 1)
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_variable_len":
            random_subseq_len = min(
                (
                    np.random.randint(self._random_variable_len_min, self._subseq_len)
                    if np.random.random() < self._random_variable_len_proportion
                    else self._subseq_len
                ),
                total_t,
            )
            start_t = np.random.randint(0, total_t - random_subseq_len + 1)
            end_t = start_t + random_subseq_len
            mask[random_subseq_len:] = False
        else:
            pass

        kwargs: dict[str, Any] = {}
        for k in npz_group.keys():

            v = npz_group[k]
            assert isinstance(k, str)
            assert isinstance(v, (h5py.Dataset, np.ndarray))
            if k == "betas":
                assert v.shape == (1, 16)
                array = v[:]
            else:
                assert v.shape[0] == total_t
                array = v[start_t:end_t]

            if array.shape[0] != self._subseq_len:
                array = np.concatenate([array, np.repeat(array[-1:,], self._subseq_len - array.shape[0], axis=0),], axis=0,)
            kwargs[k] = torch.from_numpy(array)
        kwargs["mask"] = mask
        
        if "hand_quats" not in kwargs:
            kwargs["hand_quats"] = None

        if hdf5_file is not None:
            hdf5_file.close()

        return TrainingData(**kwargs)

    def __len__(self) -> int:
        return self._approximated_length
    

def get_loader(HDF5_PATH, FILE_LIST_PATH, SUBSEQ_LEN, MUL_GPU=False):
    train_dataset = AmassHdf5Dataset(
        hdf5_path=HDF5_PATH,
        file_list_path=FILE_LIST_PATH, 
        splits=("train", ),
        subseq_len=SUBSEQ_LEN,
        cache_files=False,
        slice_strategy='random_uniform_len',
    )
    sampler = DistributedSampler(train_dataset, shuffle=MUL_GPU) if MUL_GPU else RandomSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=_C.TRAIN.batch_size,
        num_workers=4,
        # shuffle=True,
        sampler=sampler,
        collate_fn=collate_dataclass
    )
    
    val_dataset = AmassHdf5Dataset(
            hdf5_path=HDF5_PATH,
            file_list_path=FILE_LIST_PATH,
            splits=("test", ),
            subseq_len=SUBSEQ_LEN,
            cache_files=False,
            slice_strategy="deterministic",
    )
    sampler = DistributedSampler(val_dataset, shuffle=MUL_GPU) if MUL_GPU else SequentialSampler(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=4,
        # shuffle=False,
        sampler=sampler,
        collate_fn=collate_dataclass
    )
    
    return train_loader, val_loader