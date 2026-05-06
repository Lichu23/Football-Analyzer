import argparse
from pathlib import Path

from config import OUTPUT_DIR
from tracker import track_players
from heatmap import generate_all_heatmaps
from utils import video_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Football Analyzer — Phase 1: Player Heatmaps")
    parser.add_argument("--video", required=True, help="Path to input video file")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found at {video_path}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    info = video_info(str(video_path))
    print(f"Video : {video_path.name}")
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    print(f"  {info['frame_count']} frames — {info['duration_sec']:.1f}s")

    print("\nTracking players...")
    positions = track_players(str(video_path))

    output_files = generate_all_heatmaps(positions)

    print(f"\nDone. {len(output_files)} heatmaps saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
