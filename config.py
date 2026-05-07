from pathlib import Path

ROOT = Path(__file__).parent

VIDEO_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "output"

# Model — swap to yolov8x.pt for higher accuracy (slower)
MODEL_WEIGHTS = "yolov8s.pt"
DEVICE = "cpu"
CONFIDENCE = 0.5
IOU = 0.5

# ByteTrack settings
TRACK_BUFFER = 180       # frames to keep a lost track alive (~6s at 30fps)
TRACK_THRESHOLD = 0.5    # min confidence to start a new track
MIN_TRACK_FRAMES = 60    # discard any player seen in fewer than this many frames

# Field boundary polygon — normalized (x, y) coords where y=0 is top of frame.
# Detections whose foot point falls outside this polygon are ignored before tracking,
# so substitutes and people on the bench never get a tracking ID.
# Adjust the far-side y values to match the far touchline position in your video.
FIELD_POLYGON = [
    (0.00, 0.47),   # far-left corner
    (1.00, 0.43),   # far-right corner  (slight tilt from camera perspective)
    (1.00, 1.00),   # near-right corner
    (0.00, 1.00),   # near-left corner
]

# Set to 2 or 3 to skip frames and speed up processing on long videos
FRAME_SKIP = 3

# Standard football pitch dimensions in meters
PITCH_LENGTH = 105
PITCH_WIDTH = 68

# Heatmap rendering
HEATMAP_SIGMA = 15   # gaussian blur radius — higher = smoother
HEATMAP_ALPHA = 0.65 # overlay opacity

# Phase 3 — Team analytics
JERSEY_TOP_FRAC  = 0.15   # skip top 15% of bbox (head)
JERSEY_BOT_FRAC  = 0.50   # crop jersey down to 50% of bbox height
JERSEY_SIDE_FRAC = 0.15   # horizontal inset to avoid background bleed
MIN_COLOR_SAMPLES = 5     # min jersey samples for a player to enter clustering

# Jersey number OCR
OCR_EVERY_N_FRAMES = 60   # run OCR once every ~2s (at 30fps) — balance speed vs coverage
JERSEY_NUM_TOP_FRAC  = 0.20  # start of number crop (just below head)
JERSEY_NUM_BOT_FRAC  = 0.65  # end of number crop
JERSEY_NUM_SIDE_FRAC = 0.20  # horizontal inset
OCR_MIN_VOTES = 3            # need at least this many consistent readings to trust a number
OCR_MIN_CONFIDENCE = 0.30    # fraction of OCR attempts that must agree

# Spatial-temporal re-ID merge
# Two tracks of the same team that ended/started close together in time and space
# are merged into one player.
MERGE_MAX_GAP_FRAMES = 180   # max processed-frame gap (~6s at 30fps with skip=3)
MERGE_MAX_DIST       = 0.12  # max normalized distance between last/first positions

POSSESSION_GRID_COLS = 6
POSSESSION_GRID_ROWS = 4

TEAM_COLORS = ["#3b82f6", "#ef4444"]  # team 0 = blue, team 1 = red
