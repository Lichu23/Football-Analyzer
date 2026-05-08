import argparse
from pathlib import Path

from config import OUTPUT_DIR, MIN_TRACK_FRAMES
from tracker import track_players
from merge import merge_spatiotemporal
from team import assign_teams, team_summary
from team_preview import generate_team_preview
from heatmap import generate_heatmap
from metrics import calculate_all_metrics, print_metrics_table, save_metrics_csv
from trajectory import generate_all_trajectories
from utils import video_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Football Analyzer — Phase 2: Movement Metrics")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--team", type=int, choices=[0, 1], default=None,
        help="Analyze only this team (0 or 1). Omit to analyze both.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found at {video_path}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    info = video_info(str(video_path))
    fps = info["fps"]
    print(f"Video : {video_path.name}")
    print(f"  {info['width']}x{info['height']} @ {fps:.1f} fps")
    print(f"  {info['frame_count']} frames — {info['duration_sec']:.1f}s")

    # ── Track ─────────────────────────────────────────────────────────────────
    print("\nTracking players...")
    positions, color_samples, sample_crops, frame_ranges = track_players(str(video_path))

    # ── Spatial-temporal merge ────────────────────────────────────────────────
    print("\nMerging duplicate tracks...")
    positions, color_samples, sample_crops, frame_ranges = merge_spatiotemporal(
        positions, color_samples, sample_crops, frame_ranges
    )

    # ── Post-merge track length filter ────────────────────────────────────────
    before = len(positions)
    positions = {pid: pos for pid, pos in positions.items() if len(pos) >= MIN_TRACK_FRAMES}
    color_samples = {pid: color_samples[pid] for pid in positions}
    sample_crops  = {pid: sample_crops[pid]  for pid in positions}
    frame_ranges  = {pid: frame_ranges[pid]  for pid in positions}
    print(f"  {before} merged tracks → {len(positions)} kept (min {MIN_TRACK_FRAMES} positions).")

    # ── Team assignment ───────────────────────────────────────────────────────
    print("\nAssigning teams...")
    team_labels = assign_teams(color_samples)
    team_summary(team_labels)

    # ── Team preview ──────────────────────────────────────────────────────────
    print("\nGenerating team preview...")
    generate_team_preview(sample_crops, team_labels, color_samples)

    if args.team is None:
        print("\n  → Open output/team_preview.png to identify your team.")
        print("    Re-run with --team 0 or --team 1 to scope output to one team.")

    # ── Movement metrics ──────────────────────────────────────────────────────
    print("\nCalculating movement metrics...")
    all_metrics = calculate_all_metrics(positions, team_labels, fps)

    # ── Per-team outputs ──────────────────────────────────────────────────────
    teams_to_analyze = [args.team] if args.team is not None else [0, 1]

    for team_id in teams_to_analyze:
        team_dir = OUTPUT_DIR / f"team_{team_id}"
        team_dir.mkdir(exist_ok=True)

        print_metrics_table(all_metrics, team_id)
        save_metrics_csv(all_metrics, team_id, team_dir)

        print(f"\nGenerating heatmaps — Team {team_id}...")
        for pid, pos in sorted(positions.items()):
            if team_labels.get(pid) == team_id:
                generate_heatmap(pid, pos, output_dir=team_dir)

        print(f"\nGenerating trajectories — Team {team_id}...")
        generate_all_trajectories(positions, all_metrics, team_labels, team_id, team_dir)

    print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
