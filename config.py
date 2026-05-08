from pathlib import Path

ROOT = Path(__file__).parent

VIDEO_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "output"

# Model — swap to yolov8x.pt for higher accuracy (slower)
MODEL_WEIGHTS = "yolov8m.pt"
DEVICE = "cuda"
BATCH_SIZE = 8   # frames per YOLO inference call — fills GPU pipeline, eliminates saw-tooth
CONFIDENCE = 0.5
IOU = 0.5

# ByteTrack settings
TRACK_BUFFER = 180       # frames to keep a lost track alive (~6s at 30fps)
TRACK_THRESHOLD = 0.5    # min confidence to start a new track
MIN_TRACK_FRAMES = 60    # discard any player seen in fewer than this many frames (applied after merge)
PRE_MERGE_MIN_FRAMES = 5 # minimal noise filter applied before merge — keeps short fragments for re-ID

# Set to 2 or 3 to skip frames and speed up processing on long videos
FRAME_SKIP = 1

# Standard football pitch dimensions in meters
PITCH_LENGTH = 105
PITCH_WIDTH = 68

# Heatmap rendering
HEATMAP_SIGMA = 15   # gaussian blur radius — higher = smoother
HEATMAP_ALPHA = 0.65 # overlay opacity

# Field boundary polygon — normalized (x, y), y=0 is top of frame
# Drops bench/substitute detections before ByteTrack sees them
FIELD_POLYGON = [
    (0.00, 0.47),
    (1.00, 0.43),
    (1.00, 1.00),
    (0.00, 1.00),
]

# Jersey color extraction
JERSEY_TOP_FRAC  = 0.15
JERSEY_BOT_FRAC  = 0.50
JERSEY_SIDE_FRAC = 0.15
MIN_COLOR_SAMPLES = 5
SAMPLE_CROPS_PER_PLAYER = 5   # sample crops saved per player for team preview

# Team visuals
TEAM_COLORS = ["#3b82f6", "#ef4444"]   # team 0 = blue, team 1 = red

# Outlier/referee exclusion — LAB color distance from nearest team centroid.
# A player whose jersey color is further than this from both teams is excluded.
# LAB space: white vs dark = ~60 units apart. Referee teal ≈ 30-40 from nearest team.
# Lower = more strict (excludes more). Higher = more permissive (keeps more).
OUTLIER_COLOR_THRESHOLD_LAB = 20.0

# Spatial-temporal re-ID merge
MERGE_MAX_GAP_FRAMES       = 3600  # max video-frame gap (~2 min at 30fps) — camera can lose player for long
MERGE_MAX_DIST             = 0.30  # max normalized distance for mid-field tracks
MERGE_EDGE_MARGIN          = 0.12  # tracks ending/starting within this margin get a looser distance check
MERGE_COLOR_THRESHOLD_LAB  = 25.0  # max LAB jersey color distance between two tracks to allow a merge

# Phase 2 — Movement metrics
SPRINT_THRESHOLD_KMH = 20.0   # speed above this counts as a sprint
MAX_SPEED_FILTER_KMH = 40.0   # discard speeds above this (tracking jitter)
SPEED_SMOOTH_WINDOW  = 5      # rolling average window for speed smoothing
