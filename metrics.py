import csv
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

from config import (
    PITCH_LENGTH, PITCH_WIDTH, FRAME_SKIP,
    SPRINT_THRESHOLD_KMH, MAX_SPEED_FILTER_KMH, SPEED_SMOOTH_WINDOW,
)


@dataclass
class PlayerMetrics:
    player_id: int
    team_id: int
    distance_m: float
    avg_speed_kmh: float
    max_speed_kmh: float
    sprint_count: int
    sprint_distance_m: float
    speeds_kmh: list[float] = field(default_factory=list)


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    result = []
    for i in range(len(values)):
        sl = values[max(0, i - half): min(len(values), i + half + 1)]
        result.append(sum(sl) / len(sl))
    return result


def calculate_metrics(
    player_id: int,
    team_id: int,
    positions: list[tuple[float, float]],
    fps: float,
) -> PlayerMetrics:
    if len(positions) < 2:
        return PlayerMetrics(player_id, team_id, 0.0, 0.0, 0.0, 0, 0.0)

    time_step = FRAME_SKIP / fps  # seconds between consecutive position samples

    raw_distances: list[float] = []
    raw_speeds: list[float] = []

    for i in range(1, len(positions)):
        dx = (positions[i][0] - positions[i - 1][0]) * PITCH_LENGTH
        dy = (positions[i][1] - positions[i - 1][1]) * PITCH_WIDTH
        dist = sqrt(dx ** 2 + dy ** 2)
        speed_kmh = (dist / time_step) * 3.6

        if speed_kmh > MAX_SPEED_FILTER_KMH:
            speed_kmh = 0.0
            dist = 0.0

        raw_distances.append(dist)
        raw_speeds.append(speed_kmh)

    speeds = _smooth(raw_speeds, SPEED_SMOOTH_WINDOW)
    total_distance = sum(raw_distances)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0

    sprint_count = 0
    sprint_distance = 0.0
    in_sprint = False

    for dist, speed in zip(raw_distances, speeds):
        if speed >= SPRINT_THRESHOLD_KMH:
            if not in_sprint:
                sprint_count += 1
                in_sprint = True
            sprint_distance += dist
        else:
            in_sprint = False

    return PlayerMetrics(
        player_id=player_id,
        team_id=team_id,
        distance_m=round(total_distance, 1),
        avg_speed_kmh=round(avg_speed, 1),
        max_speed_kmh=round(max_speed, 1),
        sprint_count=sprint_count,
        sprint_distance_m=round(sprint_distance, 1),
        speeds_kmh=speeds,
    )


def calculate_all_metrics(
    positions: dict[int, list[tuple[float, float]]],
    team_labels: dict[int, int],
    fps: float,
) -> dict[int, PlayerMetrics]:
    return {
        pid: calculate_metrics(pid, team_labels.get(pid, 0), pos, fps)
        for pid, pos in positions.items()
    }


def print_metrics_table(all_metrics: dict[int, PlayerMetrics], team_id: int) -> None:
    rows = sorted(
        [m for m in all_metrics.values() if m.team_id == team_id],
        key=lambda m: m.player_id,
    )
    if not rows:
        return
    print(f"\n  Team {team_id} — Movement Metrics")
    print(f"  {'ID':>4}  {'Dist (m)':>9}  {'Avg km/h':>9}  {'Max km/h':>9}  {'Sprints':>8}  {'Sprint m':>9}")
    print("  " + "─" * 56)
    for m in rows:
        print(
            f"  {m.player_id:>4}  {m.distance_m:>9.1f}  "
            f"{m.avg_speed_kmh:>9.1f}  {m.max_speed_kmh:>9.1f}  "
            f"{m.sprint_count:>8}  {m.sprint_distance_m:>9.1f}"
        )


def save_metrics_csv(all_metrics: dict[int, PlayerMetrics], team_id: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"
    rows = sorted(
        [m for m in all_metrics.values() if m.team_id == team_id],
        key=lambda m: m.player_id,
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player_id", "distance_m", "avg_speed_kmh", "max_speed_kmh", "sprint_count", "sprint_distance_m"])
        for m in rows:
            writer.writerow([m.player_id, m.distance_m, m.avg_speed_kmh, m.max_speed_kmh, m.sprint_count, m.sprint_distance_m])
    print(f"  Saved {path.name}")
    return path
