# Registration of Changes

## [Phase 1] — Player Heatmap
> Start date: 2026-05-06

### Added
- `CLAUDE.md` — project instructions and structure for Claude Code
- `Registration Changes.md` — this file, tracks all changes per phase
- `analytics.md` — full project roadmap (Phases 1–9)
- `requirements.txt` — Python dependencies (ultralytics, supervision, opencv, matplotlib, scipy)
- `config.py` — all constants: paths, model weights, confidence thresholds, pitch dimensions
- `main.py` — CLI entry point (`python main.py --video videos/sample.mp4`)
- `tracker.py` — YOLOv8 + ByteTrack player detection and tracking, outputs normalized positions per ID
- `heatmap.py` — generates one heatmap PNG per player overlaid on the pitch diagram
- `pitch.py` — draws a to-scale football pitch using matplotlib (105m × 68m)
- `utils.py` — video metadata helper
- `videos/` — folder for input video files
- `output/` — folder where heatmap PNGs are saved
