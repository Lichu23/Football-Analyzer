import math
from collections import defaultdict

import cv2
import numpy as np

from config import MERGE_MAX_GAP_FRAMES, MERGE_MAX_DIST, MERGE_EDGE_MARGIN, MERGE_COLOR_THRESHOLD_LAB

# Looser distance threshold for tracks that exit/enter near the frame edge
_EDGE_MERGE_MAX_DIST = 0.60


def _near_edge(pos: tuple[float, float]) -> bool:
    x, y = pos
    return (
        x < MERGE_EDGE_MARGIN or x > (1 - MERGE_EDGE_MARGIN) or
        y < MERGE_EDGE_MARGIN or y > (1 - MERGE_EDGE_MARGIN)
    )


def _avg_lab(colors: list) -> np.ndarray | None:
    if not colors:
        return None
    avg_bgr = np.clip(np.mean(colors, axis=0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)


def _jerseys_similar(colors_a: list, colors_b: list) -> bool:
    """True if both tracks have similar enough jersey color to be the same player."""
    lab_a = _avg_lab(colors_a)
    lab_b = _avg_lab(colors_b)
    if lab_a is None or lab_b is None:
        return True  # no color data — allow merge, other checks will guard
    return float(np.linalg.norm(lab_a - lab_b)) <= MERGE_COLOR_THRESHOLD_LAB


def merge_spatiotemporal(
    positions: dict[int, list[tuple[float, float]]],
    color_samples: dict[int, list[tuple[float, float, float]]],
    sample_crops: dict[int, list],
    frame_ranges: dict[int, tuple[int, int]],
) -> tuple[dict, dict, dict, dict]:
    """
    Merge tracks that are temporally non-overlapping, spatially close, and
    have similar jersey color.

    Jersey color similarity replaces the old team-label check — it prevents
    cross-player merges without the chicken-and-egg of preliminary team labels.
    Near-edge tracks get a looser distance threshold (camera may have panned)
    but still require color similarity.
    """
    pids = list(positions.keys())

    parent = {pid: pid for pid in pids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, pid_a in enumerate(pids):
        for pid_b in pids[i + 1:]:
            a_start, a_end = frame_ranges[pid_a]
            b_start, b_end = frame_ranges[pid_b]

            if a_end < b_start:
                gap = b_start - a_end
                end_pos = positions[pid_a][-1]
                start_pos = positions[pid_b][0]
            elif b_end < a_start:
                gap = a_start - b_end
                end_pos = positions[pid_b][-1]
                start_pos = positions[pid_a][0]
            else:
                continue  # temporal overlap → definitely two different players

            if gap > MERGE_MAX_GAP_FRAMES:
                continue

            if not _jerseys_similar(color_samples.get(pid_a, []), color_samples.get(pid_b, [])):
                continue

            dist = math.sqrt(
                (end_pos[0] - start_pos[0]) ** 2 +
                (end_pos[1] - start_pos[1]) ** 2
            )
            near = _near_edge(end_pos) or _near_edge(start_pos)
            threshold = _EDGE_MERGE_MAX_DIST if near else MERGE_MAX_DIST
            if dist <= threshold:
                union(pid_a, pid_b)

    groups: dict[int, list[int]] = defaultdict(list)
    for pid in pids:
        groups[find(pid)].append(pid)

    merged_pos: dict[int, list] = {}
    merged_colors: dict[int, list] = {}
    merged_crops: dict[int, list] = {}
    merged_ranges: dict[int, tuple[int, int]] = {}

    merged_count = 0
    for group_pids in groups.values():
        canonical = min(group_pids)
        merged_pos[canonical] = [p for pid in group_pids for p in positions[pid]]
        merged_colors[canonical] = [c for pid in group_pids for c in color_samples.get(pid, [])]
        merged_crops[canonical] = sample_crops.get(canonical) or next(
            (sample_crops[pid] for pid in group_pids if sample_crops.get(pid)), []
        )
        merged_ranges[canonical] = (
            min(frame_ranges[pid][0] for pid in group_pids),
            max(frame_ranges[pid][1] for pid in group_pids),
        )
        if len(group_pids) > 1:
            merged_count += len(group_pids) - 1

    print(f"  Merged {merged_count} duplicate tracks → {len(merged_pos)} unique players")
    return merged_pos, merged_colors, merged_crops, merged_ranges
