from dataclasses import dataclass
from typing import Tuple

@dataclass
class Cfg:
    SEED: int = 42
    DATA_DIR: str = ''
    IMAGE_SIZE: int = 32
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 2
    LATENT_N: int = 256
    USE_PER_SAMPLE_PERM: bool = True
    KEY_U64: int = 1311768467463790320
    DO_WRONG_KEY_EVAL: bool = True
    WRONG_KEY_U64: int = 1084818905618843912
    SNR_BOB_DB: float = 12.0
    SNR_EVE_DB: float = 20.0
    EPOCHS: int = 30
    LR_MAIN: float = 0.0002
    LR_EVE: float = 0.0002
    LR_INT: float = 0.0002
    AMP: bool = True
    WEIGHT_DECAY: float = 0.0001
    ATTACKER_STEPS_PER_MAIN: int = 2
    LAMBDA_MS_SSIM: float = 0.1
    TARGET_CIPHER_POWER: float = 1.0
    GRAD_CLIP_NORM: float = 1.0
    LAMBDA_DEF: float = 0.2
    LAMBDA_EVE: float = 0.0
    RUN_ATTACKER_BEST_RESPONSE: bool = True
    ATTACKER_EPOCHS: int = 8
    ATTACKER_LR: float = 0.0002
    OUT_DIR: str = './SecureDSC_out'
    EVAL_MAX_BATCHES: int = 50
    SNR_SWEEP: Tuple[float, ...] = (0, 3, 6, 9, 12, 15, 18, 21)
    EVE_SNR_SWEEP: Tuple[float, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24)
    BOB_SNR_FIXED_FOR_EVE_SWEEP: float = 12.0
    JOINT_FIXED_EVE_SNRS: Tuple[float, ...] = (0, 6, 12, 18, 24)
    GAP_DELTAS_DB: Tuple[float, ...] = (0, 6, 12)
    MAKE_HEATMAPS: bool = False
    SAVE_SAMPLE_GRIDS: bool = True
    SAMPLE_GRID_N: int = 16
    PRINT_EVERY: int = 200
    SAVE_PLOTS_PDF: bool = True
    SAVE_PLOTS_PNG: bool = True
    PLOTS_DPI: int = 220
    PLOTS_FIG_W: float = 10.5
    PLOTS_FIG_H: float = 6.5
    RUN_PAPER_SANITY_CHECKS: bool = True
    SANITY_SHUFFLE_TEST: bool = True
    SANITY_UNSCRAMBLED_BASELINE: bool = True
    SANITY_WORSTCASE_LEAKAGE: bool = True
    LEAKAGE_PCTL: float = 95.0
