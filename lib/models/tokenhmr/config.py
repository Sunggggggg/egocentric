from dataclasses import dataclass

@dataclass(frozen=True)
class TokenHMRConfig:

    CODE_DIM = 256
    NB_CODE = 2048
    DOWN_T = 1
    DEPTH = 2
    WIDTH = 512
    QUANTIZER = 'ema_reset'
    ROT_TYPE = 'rot6d'
    DILATION_RATE = 3
    TOKEN_SIZE_MUL = 4
    TOKEN_SIZE_DIV = 4
    NB_JOINTS = 21