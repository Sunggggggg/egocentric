class TRAIN:
    batch_size = 48
    seqlen = 16
    PRINT_ITER = 10
    VAL_ITER = 1000
    SAVE_ITER = 10000
    # PRINT_ITER = 1
    # VAL_ITER = 1
    # SAVE_ITER = 1
    
    SMPL_PRIOR = 'parsed_data/gmm_08.pkl'

class DATA:
    AMASS_ROOT = 'parsed_data/tokenization_data/smplh'
    TRAINLIST = 'CMU_KIT_BMLrub_DanceDB_BMLmovi_EyesJapan_BMLhandball_TotalCapture_EKUT_ACCAD_TCDHands_MPI-Limits_MOYO'
    TRAIN_PART = '0.12_0.11_0.11_0.05_0.05_0.05_0.05_0.05_0.03_0.03_0.03_0.16_0.16'
    TESTLIST = 'Transitions_SSM'
    
    HDF5_PATH = '/mnt2/SKY/egoh/_egohenu/lib/data/amass/egotokenHMR_dataset.hdf5'
    FILE_LIST_PATH = '/mnt2/SKY/egoh/_egohenu/lib/data/amass/egotokenHMR_dataset_files.txt'

class SMPL:
    SMPL_MEAN_PARAM = '/mnt2/SKY/egoh/pretrained/smpl_mean_params.npz'
    SMPLH_MODEL_PATH = '/mnt2/SKY/egoh/_egohenu/lib/data/body_models/smplh/SMPLH_NEUTRAL.pkl'

class TOKENHMR:
    MODEL_PATH = 'parsed_data/tokenhmr/tokenizer.pth'
    