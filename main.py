import argparse
from pathlib import Path

from config import OUTPUT_DIR
from tracker import track_players
from jersey import merge_by_number, merge_spatiotemporal
from heatmap import generate_all_heatmaps, generate_all_team_heatmaps
from team import assign_teams, team_summary
from zones import generate_possession_map
from lines import generate_team_lines
from utils import video_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Football Analyzer")
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

    # ── Stage 1: track ────────────────────────────────────────────────────────
    print("\nTracking players...")
    positions, color_samples, number_votes, frame_ranges = track_players(str(video_path))
    print(f"  Raw unique tracks after filter: {len(positions)}")

    # ── Stage 2: preliminary team assignment (needed before number merge) ────
    print("\nPreliminary team assignment...")
    team_labels = assign_teams(color_samples)
    team_summary(team_labels)

    # ── Stage 3: merge by (team + jersey number) ──────────────────────────────
    print("\nStage 1 merge — team + jersey number...")
    positions, color_samples, player_numbers, frame_ranges = merge_by_number(
        positions, color_samples, number_votes, frame_ranges, team_labels
    )
    print(f"  After number merge: {len(positions)} tracks")

    # ── Stage 4: merge by spatial-temporal proximity ──────────────────────────
    print("\nStage 2 merge — spatial-temporal (same team, same location)...")
    positions, color_samples, frame_ranges = merge_spatiotemporal(
        positions, color_samples, frame_ranges, team_labels
    )
    # Final team assignment on fully merged data — larger color pools = more stable
    team_labels = assign_teams(color_samples)
    print(f"  Final unique players: {len(positions)}")
    team_summary(team_labels)

    # ── Stage 5: generate all outputs ────────────────────────────────────────
    print("\nGenerating individual player heatmaps...")
    individual_files = generate_all_heatmaps(positions)

    team_heatmap_files = generate_all_team_heatmaps(positions, team_labels)

    print("\nGenerating possession zones...")
    zones_file = generate_possession_map(positions, team_labels)

    print("\nGenerating team lines...")
    lines_file = generate_team_lines(positions, team_labels)

    total = len(individual_files) + len(team_heatmap_files) + 2
    print(f"\nDone. {total} outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
