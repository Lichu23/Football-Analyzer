from pathlib import Path

ROOT = Path(__file__).parent

VIDEO_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "output"

# Model — swap to yolov8x.pt for higher accuracy (slower)
MODEL_WEIGHTS = "yolov8l.pt"
DEVICE = "cpu"
CONFIDENCE = 0.5
IOU = 0.5

# ByteTrack settings
TRACK_BUFFER = 90        # frames to keep a lost track alive (~3s at 30fps)
TRACK_THRESHOLD = 0.5    # min confidence to start a new track
MIN_TRACK_FRAMES = 30    # discard any player seen in fewer than this many frames

# Set to 2 or 3 to skip frames and speed up processing on long videos
FRAME_SKIP = 1

# Standard football pitch dimensions in meters
PITCH_LENGTH = 105
PITCH_WIDTH = 68

# Heatmap rendering
HEATMAP_SIGMA = 15   # gaussian blur radius — higher = smoother
HEATMAP_ALPHA = 0.65 # overlay opacity
