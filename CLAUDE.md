# Football Analyzer — Claude Code Instructions

## Project overview
Local Python tool that analyzes football match videos to generate player heatmaps, movement metrics, and team analytics using computer vision and ML.

## Tech stack
- **Detection**: YOLOv8 via `ultralytics`
- **Tracking**: `supervision` (ByteTrack)
- **Video I/O**: `opencv-python`
- **Heatmap / visualization**: `matplotlib`, `numpy`
- **Runtime**: Python 3.10+

## Project structure (target)
```
football-analyzer/
├── main.py               # entry point — run analysis on a video
├── tracker.py            # player detection + tracking logic
├── heatmap.py            # heatmap generation per player
├── pitch.py              # pitch diagram drawing utilities
├── utils.py              # shared helpers
├── config.py             # thresholds, paths, constants
├── output/               # generated heatmap images go here
└── videos/               # input video files go here
```

## Roadmap phases
See `analytics.md` for the full roadmap. Current target: **Phase 1 — Player heatmap**.

## How to run
```bash
pip install -r requirements.txt
python main.py --video videos/sample.mp4
```

## Key decisions
- Local-first: validate ML logic before building any API or frontend
- Tracking ID is used as player identity (no jersey number detection until Phase 4)
- Pitch diagram is a static overhead drawing; player positions are homography-mapped from the video frame
- GPU optional: CUDA speeds up inference but CPU works for short clips

## Output
- One heatmap image per tracked player saved to `output/`
- Images labeled by tracking ID (e.g., `player_7_heatmap.png`)
