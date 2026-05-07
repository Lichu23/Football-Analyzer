import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pitch import draw_pitch
from config import OUTPUT_DIR, PITCH_LENGTH, PITCH_WIDTH, TEAM_COLORS


def generate_team_lines(
    positions: dict[int, list[tuple[float, float]]],
    team_labels: dict[int, int],
) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)

    team_x: dict[int, list[float]] = {0: [], 1: []}

    for pid, pos_list in positions.items():
        team = team_labels.get(pid, 0)
        for x_norm, _ in pos_list:
            team_x[team].append(x_norm * PITCH_LENGTH)

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)

    for team_id in [0, 1]:
        xs = team_x[team_id]
        if not xs:
            continue
        mean_x = float(np.mean(xs))
        std_x = float(np.std(xs))
        color = TEAM_COLORS[team_id]

        # Average line (constrained to pitch bounds)
        ax.plot(
            [mean_x, mean_x], [0, PITCH_WIDTH],
            color=color, linewidth=2.5,
            label=f"Team {team_id} — avg {mean_x:.1f} m",
        )
        # Spread band (±1 std dev)
        ax.fill_betweenx(
            [0, PITCH_WIDTH],
            max(0, mean_x - std_x),
            min(PITCH_LENGTH, mean_x + std_x),
            alpha=0.15, color=color,
        )

    ax.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    ax.set_title("Team Lines — Average Horizontal Position (±1 std)", color="white", fontsize=13, pad=10)
    fig.patch.set_facecolor("#1a1a2e")

    output_path = OUTPUT_DIR / "team_lines.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved {output_path.name}")
    return output_path
