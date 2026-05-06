import matplotlib.pyplot as plt
import matplotlib.patches as patches

_PITCH_COLOR = "#3a7d44"
_LINE_COLOR = "white"
_LW = 2


def draw_pitch(ax, length=105, width=68):
    ax.set_facecolor(_PITCH_COLOR)
    ax.set_xlim(-3, length + 3)
    ax.set_ylim(-3, width + 3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer boundary — facecolor must be "none" so the heatmap (drawn as an image
    # behind all patches in matplotlib's layer order) is not hidden by this rectangle
    ax.add_patch(patches.Rectangle((0, 0), length, width, lw=_LW, edgecolor=_LINE_COLOR, facecolor="none"))

    # Halfway line
    ax.plot([length / 2, length / 2], [0, width], color=_LINE_COLOR, lw=_LW)

    # Center circle + spot
    ax.add_patch(plt.Circle((length / 2, width / 2), 9.15, color=_LINE_COLOR, fill=False, lw=_LW))
    ax.plot(length / 2, width / 2, "o", color=_LINE_COLOR, ms=3)

    # Penalty areas (18-yard box)
    for x_start in [0, length - 16.5]:
        ax.add_patch(patches.Rectangle(
            (x_start, (width - 40.32) / 2), 16.5, 40.32,
            lw=_LW, edgecolor=_LINE_COLOR, facecolor="none",
        ))

    # Goal areas (6-yard box)
    for x_start in [0, length - 5.5]:
        ax.add_patch(patches.Rectangle(
            (x_start, (width - 18.32) / 2), 5.5, 18.32,
            lw=_LW, edgecolor=_LINE_COLOR, facecolor="none",
        ))

    # Penalty spots
    ax.plot(11, width / 2, "o", color=_LINE_COLOR, ms=3)
    ax.plot(length - 11, width / 2, "o", color=_LINE_COLOR, ms=3)

    # Penalty arcs (the portion outside each penalty area)
    ax.add_patch(patches.Arc((11, width / 2), 18.3, 18.3, angle=0, theta1=307, theta2=53, color=_LINE_COLOR, lw=_LW))
    ax.add_patch(patches.Arc((length - 11, width / 2), 18.3, 18.3, angle=0, theta1=127, theta2=233, color=_LINE_COLOR, lw=_LW))

    # Goals
    for x, dx in [(0, -2.44), (length, 2.44)]:
        ax.add_patch(patches.Rectangle(
            (x, (width - 7.32) / 2), dx, 7.32,
            lw=_LW, edgecolor=_LINE_COLOR, facecolor="none",
        ))

    # Corner arcs
    for cx, cy, t1, t2 in [(0, 0, 0, 90), (length, 0, 90, 180), (0, width, 270, 360), (length, width, 180, 270)]:
        ax.add_patch(patches.Arc((cx, cy), 2, 2, angle=0, theta1=t1, theta2=t2, color=_LINE_COLOR, lw=_LW))

    return ax
