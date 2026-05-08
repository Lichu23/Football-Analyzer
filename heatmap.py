import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

from pitch import draw_pitch
from config import OUTPUT_DIR, PITCH_LENGTH, PITCH_WIDTH, HEATMAP_SIGMA, HEATMAP_ALPHA


def generate_heatmap(
    player_id: int,
    positions: list[tuple[float, float]],
    output_dir: Path | None = None,
) -> Path:
    save_dir = output_dir or OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax, length=PITCH_LENGTH, width=PITCH_WIDTH)

    if positions:
        xs = [p[0] * PITCH_LENGTH for p in positions]
        # Flip y: video frame has y=0 at top, pitch diagram has y=0 at bottom
        ys = [(1 - p[1]) * PITCH_WIDTH for p in positions]

        heatmap, _, _ = np.histogram2d(
            xs, ys,
            bins=(210, 136),
            range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
        )
        heatmap = gaussian_filter(heatmap.T, sigma=HEATMAP_SIGMA)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Mask near-zero values so they render as transparent instead of black
        masked = np.ma.masked_where(heatmap < 0.05, heatmap)

        ax.imshow(
            masked,
            extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
            origin="lower",
            cmap="hot",
            alpha=HEATMAP_ALPHA,
            vmin=0.05,
            vmax=1,
        )
        ax.set_aspect("equal")  # restore after imshow resets it

    ax.set_title(
        f"Player {player_id}  —  {len(positions)} samples",
        color="white", fontsize=13, pad=10,
    )
    fig.patch.set_facecolor("#1a1a2e")

    output_path = save_dir / f"player_{player_id}_heatmap.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved {output_path.name}")
    return output_path


def generate_all_heatmaps(all_positions: dict[int, list[tuple[float, float]]]) -> list[Path]:
    print(f"\nGenerating heatmaps for {len(all_positions)} players...")
    return [generate_heatmap(pid, pos) for pid, pos in sorted(all_positions.items())]
