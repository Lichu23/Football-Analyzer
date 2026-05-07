import numpy as np
from sklearn.cluster import KMeans

from config import MIN_COLOR_SAMPLES


def assign_teams(
    color_samples: dict[int, list[tuple[float, float, float]]],
) -> dict[int, int]:
    """
    Cluster players into two teams by KMeans on mean jersey BGR color.
    Returns {player_id: team_id} where team_id is 0 or 1.
    """
    eligible_ids = [pid for pid, s in color_samples.items() if len(s) >= MIN_COLOR_SAMPLES]

    if len(eligible_ids) < 2:
        return {pid: 0 for pid in color_samples}

    avg_colors = np.array([
        np.mean(color_samples[pid], axis=0) for pid in eligible_ids
    ])

    n_clusters = min(2, len(eligible_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(avg_colors)
    team_map: dict[int, int] = {pid: int(lbl) for pid, lbl in zip(eligible_ids, labels)}

    # Players with too few color samples: assign to nearest centroid
    sparse_ids = [pid for pid in color_samples if pid not in team_map]
    if sparse_ids:
        sparse_colors = np.array([
            np.mean(color_samples[pid], axis=0) if color_samples[pid]
            else avg_colors.mean(axis=0)
            for pid in sparse_ids
        ])
        for pid, lbl in zip(sparse_ids, kmeans.predict(sparse_colors)):
            team_map[pid] = int(lbl)

    return team_map


def team_summary(team_labels: dict[int, int]) -> None:
    counts: dict[int, int] = {}
    for lbl in team_labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    for tid, count in sorted(counts.items()):
        print(f"  Team {tid}: {count} players")
