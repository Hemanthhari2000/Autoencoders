import torch

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
INPUT_DIM = 28 * 28
HIDDEN_DIM = 200
Z_DIM = 20
NUM_EPOCS = 50
BATCH_SIZE = 32
LR_RATE = 3e-4  # Karpathy Constant
