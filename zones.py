import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from pathlib import Path

from pitch import draw_pitch
from config import (
    OUTPUT_DIR, PITCH_LENGTH, PITCH_WIDTH,
    POSSESSION_GRID_COLS, POSSESSION_GRID_ROWS, TEAM_COLORS,
)


def generate_possession_map(
    positions: dict[int, list[tuple[float, float]]],
    team_labels: dict[int, int],
) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # grid[row, col, team] = sample count
    grid = np.zeros((POSSESSION_GRID_ROWS, POSSESSION_GRID_COLS, 2))
    col_step = PITCH_LENGTH / POSSESSION_GRID_COLS
    row_step = PITCH_WIDTH / POSSESSION_GRID_ROWS

    for pid, pos_list in positions.items():
        team = team_labels.get(pid, 0)
        for x_norm, y_norm in pos_list:
            x = x_norm * PITCH_LENGTH
            y = (1 - y_norm) * PITCH_WIDTH
            col = min(int(x / col_step), POSSESSION_GRID_COLS - 1)
            row = min(int(y / row_step), POSSESSION_GRID_ROWS - 1)
            grid[row, col, team] += 1

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)

    for row in range(POSSESSION_GRID_ROWS):
        for col in range(POSSESSION_GRID_COLS):
            t0, t1 = grid[row, col, 0], grid[row, col, 1]
            total = t0 + t1
            if total == 0:
                continue
            dom_team = 0 if t0 >= t1 else 1
            # alpha scales from 0.1 (50/50 split) to 0.7 (total dominance)
            dominance = max(t0, t1) / total
            alpha = 0.1 + 0.6 * (dominance - 0.5) / 0.5
            rect = patches.Rectangle(
                (col * col_step, row * row_step),
                col_step, row_step,
                facecolor=to_rgba(TEAM_COLORS[dom_team], alpha=alpha),
                edgecolor="white",
                linewidth=0.4,
            )
            ax.add_patch(rect)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEAM_COLORS[0], markersize=12, label="Team 0"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEAM_COLORS[1], markersize=12, label="Team 1"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    ax.set_title("Possession Zones", color="white", fontsize=13, pad=10)
    fig.patch.set_facecolor("#1a1a2e")

    output_path = OUTPUT_DIR / "possession_zones.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved {output_path.name}")
    return output_path
