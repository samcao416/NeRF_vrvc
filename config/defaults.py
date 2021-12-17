from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.COARSE_RAY_SAMPLING = 64
_C.MODEL.FINE_RAY_SAMPLING = 80
_C.MODEL.SAMPLE_METHOD = "NEAR_FAR"
_C.MODEL.BOARDER_WEIGHT = 1e10 
_C.MODEL.SAME_SPACENET = False
_C.MODEL.BACKBONE_DIM = 256 

_C.MODEL.TKERNEL_INC_RAW = True #把input一起输入到positional encoding里

_C.MODEL.USE_DIR = True

_C.MODEL.BLENDING_SCHEME = 'VOLUME RENDERING' # 'VOLUME RENDERING'
_C.MODEL.EMBED_TYPE = 'POSITIONAL' # 'POSITIONAL'

_C.MODEL.LOSS = 'L2' # 'L1', 'L2'
_C.MODEL.SPACENET = 'NORMAL'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""
_C.DATASETS.TYPE = 'syn'
_C.DATASETS.TMP_RAYS = "rays_tmp"
# List of the dataset names for testing, as present in paths_catalog.py

_C.DATASETS.PE_METHOD = 'origin' # origin or mip
_C.DATASETS.USE_MASK = False
_C.DATASETS.FACTOR = 1 # Downsample rate
_C.DATASETS.FIXED_NEAR = -1.0
_C.DATASETS.FIXED_FAR = -1.0
_C.DATASETS.SCALE = 1.0 # Use it to scale the Cartesian coordinate units

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005 #?
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1 #?
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3 #?
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.TEST_PERIOD = 1000
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.BUNCH = 4096 #?
_C.SOLVER.START_ITERS=50
_C.SOLVER.END_ITERS=200
_C.SOLVER.LR_SCALE=0.1
_C.SOLVER.COARSE_STAGE = 10

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16 

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""
_C.TEST.SAMPLE_NUMS = 100000
_C.TEST.STEP_SIZE = 1
_C.TEST.STEP_NUM = 2
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "/sharedata/home/caojm/outputs/nerf"
