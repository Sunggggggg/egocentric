class TRAIN:
    batch_size = 48
    seqlen = 16
    PRINT_ITER = 100
    VAL_ITER = 1000
    SAVE_ITER = 10000
    
    SMPL_PRIOR = 'parsed_data/gmm_08.pkl'

class DATA:
    AMASS_ROOT = 'parsed_data/tokenization_data/smplh'
    TRAINLIST = 'CMU_KIT_BMLrub_DanceDB_BMLmovi_EyesJapan_BMLhandball_TotalCapture_EKUT_ACCAD_TCDHands_MPI-Limits_MOYO'
    TRAIN_PART = '0.12_0.11_0.11_0.05_0.05_0.05_0.05_0.05_0.03_0.03_0.03_0.16_0.16'
    TESTLIST = 'Transitions_SSM'
    
    HDF5_PATH = 'parsed_data/data/egotokenHMR_dataset.hdf5'
    FILE_LIST_PATH = 'parsed_data/data/egotokenHMR_dataset_files.txt'

class SMPL:
    SMPL_MEAN_PARAM = 'parsed_data/smpl/smpl_mean_params.npz'
    SMPLH_MODEL_PATH = 'parsed_data/smpl/SMPLH_NEUTRAL.pkl'

class TOKENHMR:
    MODEL_PATH = 'parsed_data/tokenhmr/tokenizer.pth'
    