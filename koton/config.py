import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Node Types
NODE_TYPES = ["process", "file", "socket"]

# Canonical Edge Types (Source, Relation, Destination)
CANONICAL_EDGE_TYPES = [
    ("process", "fork", "process"),
    ("process", "exec", "process"),
    ("process", "read", "file"),
    ("process", "write", "file"),
    ("process", "connect", "socket"),
    # Bidirectional reverse edges for GNN message propagation
    ("process", "rev_fork", "process"),
    ("process", "rev_exec", "process"),
    ("file", "rev_read", "process"),
    ("file", "rev_write", "process"),
    ("socket", "rev_connect", "process"),
]

# Feature Dimensions
PROCESS_FEAT_DIM = 16
FILE_FEAT_DIM = 12
SOCKET_FEAT_DIM = 8

# Model Architecture
GNN_HIDDEN_CHANNELS = 64
GNN_NUM_HEADS = 4
GNN_NUM_LAYERS = 2
DROPOUT_RATE = 0.2

# Training & Anomaly Detection Tuning
TARGET_FPR = 0.001          # Target False Positive Rate: 0.1%
FOCAL_ALPHA = 0.85          # Focal loss weight for rare attack class
FOCAL_GAMMA = 2.0           # Focal loss focusing parameter
LEARNING_RATE = 0.003
BATCH_SIZE = 1              # Graph-level sliding window batch
NUM_EPOCHS = 25

# Streaming Provenance Graph Parameters
SLIDING_WINDOW_SEC = 30.0   # Window duration in seconds
WINDOW_STEP_SEC = 2.0       # Window sliding frequency in seconds
MAX_CACHE_ENTITIES = 5000   # Max cached persistent nodes across windows

# Server Settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
SSE_PING_INTERVAL_SEC = 1.0
