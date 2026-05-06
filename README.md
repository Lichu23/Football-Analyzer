# Football Analyzer

A local computer vision tool that analyzes football match videos to generate player heatmaps, movement metrics, and team analytics.

## Features

- **Player detection & tracking** — YOLOv8 + ByteTrack, one tracking ID per player
- **Player heatmaps** — shows where each player spent the most time, overlaid on a to-scale pitch diagram
- More phases in progress (speed, sprints, team classification, ball tracking)

## Requirements

- Python 3.10+
- AMD/NVIDIA GPU optional (CPU works for short clips)

## Setup

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

Drop your video into the `videos/` folder, then run:

```powershell
python main.py --video videos/yourfile.mp4
```

Heatmap PNGs are saved to `output/`, one per tracked player.

### Download a clip from YouTube

```powershell
# Download 1 minute starting at 11:00
yt-dlp -o "videos/match.webm" --download-sections "*660-720" "YOUR_URL"
```

## Project structure

```
football-analyzer/
├── main.py        # CLI entry point
├── tracker.py     # YOLOv8 + ByteTrack detection and tracking
├── heatmap.py     # heatmap generation per player
├── pitch.py       # football pitch diagram (matplotlib)
├── config.py      # all settings and constants
├── utils.py       # shared helpers
├── videos/        # input video files (gitignored)
└── output/        # generated heatmap PNGs (gitignored)
```

## Configuration

Edit `config.py` to adjust:

| Setting | Default | Description |
|---|---|---|
| `MODEL_WEIGHTS` | `yolov8l.pt` | swap to `yolov8x.pt` for higher accuracy |
| `CONFIDENCE` | `0.5` | detection confidence threshold |
| `FRAME_SKIP` | `1` | set to `2-3` to speed up long videos |
| `MIN_TRACK_FRAMES` | `30` | minimum frames to keep a track |

## Roadmap

- [x] Phase 1 — Player heatmaps
- [ ] Phase 2 — Distance, speed, sprint detection
- [ ] Phase 3 — Team classification by jersey color
- [ ] Phase 4 — Ball tracking, pass network, Voronoi
- [ ] Phase 5 — Multi-video support
- [ ] Phase 6+ — SaaS (FastAPI, Celery, frontend, Stripe)
