"""Generate every plot used in the book (docs/*.md).

Run from the repo root:   python3 scripts/make_figures.py
Output: docs/figures/*.svg  (re-run any time; it overwrites)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette (validated reference palette, light mode) ----
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, RED = "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE_LIGHT = "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "svg.fonttype": "path",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), format="svg", bbox_inches="tight")
    if os.environ.get("PREVIEW_DIR"):  # optional PNG previews for checking layout
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".svg", ".png")), dpi=70, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def arrow(ax, start, end, color=BLUE, lw=2.2, label=None, label_offset=(0, 0), ls="-", **kw):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
                        linestyle=ls, shrinkA=0, shrinkB=0, zorder=3, **kw)
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, color=INK, ha="center", va="center", fontsize=10.5)


def axes_through_origin(ax):
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)


# ============================ Chapter 0 ============================
def ch0():
    # parabola
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(-3, 3.001, 0.25)
    ax.plot(x, x ** 2, color=BLUE)
    pts = np.array([-2, -1, 0, 1, 2])
    ax.scatter(pts, pts ** 2, s=55, color=ORANGE, zorder=4, edgecolor=SURF, linewidth=1.5)
    for p in pts:
        ax.annotate(f"({p}, {p*p})", (p, p * p), textcoords="offset points",
                    xytext=(12 if p >= 0 else -12, -4), ha="left" if p >= 0 else "right", fontsize=9, color=INK2)
    ax.annotate("lowest point: x = 0\n(training hunts for spots like this)", (0, 0), xytext=(-1.1, 6.5), ha="center",
                arrowprops=dict(arrowstyle="->", color=INK2), fontsize=9.5, color=INK)
    ax.set_xlabel("input x"); ax.set_ylabel("output f(x) = x²"); ax.set_title("f(x) = x²: a U-shaped curve")
    save(fig, "ch0_parabola.svg")

    # straight line with slope & intercept
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.linspace(-2, 3, 50)
    ax.plot(x, 2 * x + 1, color=BLUE)
    axes_through_origin(ax)
    ax.plot([1, 2], [3, 3], color=ORANGE, lw=2); ax.plot([2, 2], [3, 5], color=ORANGE, lw=2)
    ax.text(1.5, 2.45, "run = 1", ha="center", color=INK); ax.text(2.1, 4, "rise = 2", color=INK)
    ax.scatter([0], [1], s=60, color=AQUA, zorder=4, edgecolor=SURF, linewidth=1.5)
    ax.annotate("intercept b = 1\n(where it crosses the y-axis)", (0, 1), xytext=(-1.9, 4.2),
                arrowprops=dict(arrowstyle="->", color=INK2), fontsize=9.5)
    ax.text(2.2, 1.6, "slope m = rise / run = 2", fontsize=10, color=INK)
    ax.set_title("y = 2x + 1"); ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "ch0_line.svg")

    # exp and log as mirror images
    fig, ax = plt.subplots(figsize=(5.2, 5))
    x = np.linspace(-3, 2.2, 200)
    ax.plot(x, np.exp(x), color=BLUE, label="eˣ  (always positive)")
    x2 = np.linspace(0.03, 9, 300)
    ax.plot(x2, np.log(x2), color=ORANGE, label="ln x  (negative between 0 and 1)")
    t = np.linspace(-3, 9, 2)
    ax.plot(t, t, color=MUTED, ls="--", lw=1.2, label="y = x (the mirror)")
    axes_through_origin(ax)
    ax.scatter([0, 1], [1, 0], color=INK, s=30, zorder=5)
    ax.annotate("e⁰ = 1", (0, 1), xytext=(-2.7, 2.2), arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("ln 1 = 0", (1, 0), xytext=(2.2, -2.3), arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlim(-3, 9); ax.set_ylim(-3.5, 9); ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, fontsize=9)
    ax.set_title("exp and ln undo each other")
    save(fig, "ch0_exp_log.svg")


def ch0_extra():
    # domain and range
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    x = np.linspace(0.02, 6, 300)
    ax = axs[0]; axes_through_origin(ax)
    ax.axvspan(-4, 0, color=GRID, alpha=0.8, lw=0)
    ax.plot(x, np.log(x), color=ORANGE, lw=2.5)
    ax.text(-2, 1.0, "not allowed:\nln of 0 or\nnegatives", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(-4, 6); ax.set_ylim(-3.5, 2.5)
    ax.set_title("ln x  domain: x > 0  range: everything", fontsize=10)
    x = np.linspace(-6, 6, 300)
    ax = axs[1]; axes_through_origin(ax)
    ax.axhspan(0, 1, color=BLUE_LIGHT, alpha=0.7, lw=0)
    ax.plot(x, 1 / (1 + np.exp(-x)), color=AQUA, lw=2.5)
    ax.text(-5.6, 1.15, "range: always between 0 and 1,\nso it can be read as a probability", fontsize=9, color=INK2)
    ax.set_ylim(-0.3, 1.6); ax.set_title("sigmoid  domain: everything  range: (0, 1)", fontsize=10)
    ax = axs[2]; axes_through_origin(ax)
    ax.axhspan(-3, 0, color=GRID, alpha=0.8, lw=0)
    ax.plot(x, np.maximum(0, x), color=BLUE, lw=2.5)
    ax.text(0.5, -1.6, "outputs never go here", fontsize=9, color=INK2)
    ax.set_xlim(-4, 4); ax.set_ylim(-3, 4); ax.set_title("ReLU  domain: everything  range: x ≥ 0", fontsize=10)
    save(fig, "ch0_domain_range.svg")

    # composition is not commutative / inverse: horizontal line test
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.linspace(-2.6, 2.6, 300)
    ax = axs[0]; axes_through_origin(ax)
    ax.plot(x, x ** 2, color=BLUE, lw=2.5); ax.axhline(4, color=ORANGE, ls="--", lw=1.5)
    ax.scatter([-2, 2], [4, 4], color=ORANGE, s=60, zorder=5, edgecolor=SURF)
    ax.text(0, 4.4, "output 4 came from x = −2 or x = 2:\ncannot be undone uniquely", ha="center", fontsize=9)
    ax.set_ylim(-1, 6.5); ax.set_title("x² (all inputs): no inverse", fontsize=10.5)
    ax = axs[1]; axes_through_origin(ax)
    ax.plot(x, np.exp(x), color=AQUA, lw=2.5); ax.axhline(2, color=ORANGE, ls="--", lw=1.5)
    ax.scatter([np.log(2)], [2], color=ORANGE, s=60, zorder=5, edgecolor=SURF)
    ax.text(-2.4, 4.4, "every output came from exactly one\ninput: it can be undone (by ln)", fontsize=9)
    ax.set_ylim(-1, 6.5); ax.set_title("eˣ: has an inverse", fontsize=10.5)
    fig.suptitle("The horizontal line test: if a flat line hits the curve twice, the function cannot be undone",
                 color=INK2, fontsize=10)
    save(fig, "ch0_inverse.svg")


# ============================ Chapter 1 ============================
def ch1():
    # [3,4] arrow with Pythagoras
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(Polygon([[0, 0], [3, 0], [3, 4]], closed=True, color=BLUE_LIGHT, alpha=0.7, lw=0))
    ax.plot([0, 3], [0, 0], color=ORANGE); ax.plot([3, 3], [0, 4], color=ORANGE)
    arrow(ax, (0, 0), (3, 4), color=BLUE, lw=2.6)
    ax.text(1.5, -0.35, "3 east", ha="center"); ax.text(3.15, 2, "4 north", va="center")
    ax.text(1.05, 2.35, "length = √(3² + 4²) = 5", rotation=53, ha="center", va="center", color=INK)
    ax.add_patch(Rectangle((2.7, 0), 0.3, 0.3, fill=False, ec=INK2, lw=1))
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.7, 4.6); ax.set_aspect("equal")
    ax.set_title("The vector [3, 4] is an arrow")
    save(fig, "ch1_vector_34.svg")

    # 3D walk [1,2,2]
    fig = plt.figure(figsize=(5.6, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURF)
    # box edges
    import itertools
    for s, e in itertools.combinations(np.array(list(itertools.product([0, 1], [0, 2], [0, 2]))), 2):
        if np.sum(np.abs(s - e) > 0) == 1:
            ax.plot(*zip(s, e), color=GRID, lw=1)
    ax.plot([0, 1], [0, 0], [0, 0], color=ORANGE, lw=3)
    ax.plot([1, 1], [0, 2], [0, 0], color=ORANGE, lw=3)
    ax.plot([1, 1], [2, 2], [0, 2], color=ORANGE, lw=3, label="the walk: 1 east, 2 north, 2 up")
    ax.plot([0, 1], [0, 2], [0, 0], color=AQUA, lw=2, ls="--", label="floor diagonal: √5")
    ax.plot([0, 1], [0, 2], [0, 2], color=BLUE, lw=3, label="shortcut: length 3")
    ax.scatter([0], [0], [0], color=INK, s=30); ax.scatter([1], [2], [2], color=INK, s=30)
    ax.text(0, 0, -0.35, "start", fontsize=9); ax.text(1, 2, 2.2, "end [1, 2, 2]", fontsize=9)
    ax.set_xlabel("east"); ax.set_ylabel("north"); ax.set_zlabel("up")
    ax.set_box_aspect((1, 2, 2)); ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Length of [1, 2, 2] in 3D")
    save(fig, "ch1_3d_walk.svg")

    # two triangles flattened
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax in axs:
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax = axs[0]
    ax.add_patch(Polygon([[0, 0], [1, 0], [1, 2]], color=BLUE_LIGHT, lw=0))
    ax.plot([0, 1, 1, 0], [0, 0, 2, 0], color=INK2, lw=1.5)
    ax.plot([0, 1], [0, 2], color=AQUA, lw=3)
    ax.text(0.5, -0.2, "1", ha="center"); ax.text(1.08, 1, "2"); ax.text(0.25, 1.15, "√(1²+2²) = √5", color=INK, rotation=63)
    ax.set_xlim(-0.6, 1.8); ax.set_ylim(-0.4, 2.4); ax.set_aspect("equal"); ax.set_title("Triangle 1: along the floor")
    ax = axs[1]
    r5 = np.sqrt(5)
    ax.add_patch(Polygon([[0, 0], [r5, 0], [r5, 2]], color=BLUE_LIGHT, lw=0))
    ax.plot([0, r5, r5, 0], [0, 0, 2, 0], color=INK2, lw=1.5)
    ax.plot([0, r5], [0, 0], color=AQUA, lw=3)
    ax.plot([0, r5], [0, 2], color=BLUE, lw=3)
    ax.text(r5 / 2, -0.2, "√5", ha="center"); ax.text(r5 + 0.08, 1, "2 (up)")
    ax.text(0.7, 1.05, "√(5 + 4) = 3", rotation=41, color=INK)
    ax.set_xlim(-0.3, 3.2); ax.set_ylim(-0.4, 2.4); ax.set_aspect("equal"); ax.set_title("Triangle 2: then climb")
    fig.suptitle("The long side of one triangle becomes a short side of the next", color=INK2, fontsize=10)
    save(fig, "ch1_two_triangles.svg")

    # movies scatter
    movies = {"Scream": (4, 8), "Shaun of the Dead": (8, 6), "Paddington": (9, 0), "The Conjuring": (1, 10)}
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, (f, s) in movies.items():
        ax.scatter(f, s, s=90, color=BLUE, zorder=4, edgecolor=SURF, linewidth=2)
        ax.annotate(name, (f, s), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.plot([4, 8], [8, 6], color=AQUA, ls="--", lw=1.6); ax.text(6.1, 7.4, "4.5 apart (3D)", color=INK2, fontsize=9)
    ax.plot([4, 9], [8, 0], color=ORANGE, ls="--", lw=1.6); ax.text(6.8, 3.6, "9.5 apart (3D)", color=INK2, fontsize=9)
    ax.set_xlim(0, 11); ax.set_ylim(-1, 11.5)
    ax.set_xlabel("how funny (0-10)"); ax.set_ylabel("how scary (0-10)")
    ax.set_title("Movies as points: similar vibes sit close")
    save(fig, "ch1_movies.svg")

    # add & scale
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4))
    ax = axs[0]
    arrow(ax, (0, 0), (3, 1), BLUE, label="a = [3, 1]", label_offset=(0.3, -0.45))
    arrow(ax, (3, 1), (4, 4), ORANGE, label="b = [1, 3]", label_offset=(0.85, 0))
    arrow(ax, (0, 0), (1, 3), ORANGE, ls="--", lw=1.3)
    arrow(ax, (1, 3), (4, 4), BLUE, ls="--", lw=1.3)
    arrow(ax, (0, 0), (4, 4), AQUA, lw=2.8, label="a + b = [4, 4]", label_offset=(0.75, -0.35))
    ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 5); ax.set_aspect("equal"); ax.set_title("Add: walk a, then walk b")
    ax = axs[1]
    arrow(ax, (0, 0), (2, 4), AQUA, lw=6, label="2 × [1, 2] = [2, 4]", label_offset=(0.4, 1.75))
    arrow(ax, (0, 0), (1, 2), BLUE, lw=2.6, label="[1, 2]", label_offset=(-0.95, 0.1))
    arrow(ax, (0, 0), (-0.5, -1), ORANGE, lw=2.6, label="-0.5 × [1, 2]", label_offset=(1.1, -0.1))
    ax.set_xlim(-1.5, 4); ax.set_ylim(-1.5, 4.5); ax.set_aspect("equal"); ax.set_title("Scale: same line, new length")
    for ax in axs:
        axes_through_origin(ax)
    save(fig, "ch1_add_scale.svg")

    # dot product shadow
    a, b = np.array([3, 1]), np.array([2, 3])
    proj = (a @ b) / (a @ a) * a
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    t = np.linspace(-0.3, 1.5, 2)
    ax.plot(t * 3, t * 1, color=GRID, lw=1.2)
    ax.plot([b[0], proj[0]], [b[1], proj[1]], color=MUTED, ls=":", lw=1.6)
    ax.plot([0, proj[0]], [0, proj[1]], color=ORANGE, lw=7, alpha=0.45, solid_capstyle="butt")
    arrow(ax, (0, 0), tuple(a), BLUE, label="a = [3, 1]   (‖a‖ = √10)", label_offset=(0.9, -0.55))
    arrow(ax, (0, 0), tuple(b), AQUA, label="b = [2, 3]", label_offset=(-0.75, 0.2))
    ax.text(proj[0] + 0.2, proj[1] - 0.55, f"shadow of b on a\nlength = 9/√10 ≈ 2.85", color=INK2, fontsize=9)
    ax.text(0.2, 3.5, "a · b = ‖a‖ × shadow = √10 × 9/√10 = 9\n        = 3×2 + 1×3 = 9", fontsize=10, color=INK)
    ax.set_xlim(-0.5, 4.6); ax.set_ylim(-0.6, 4.2); ax.set_aspect("equal")
    ax.set_title("The dot product as a shadow")
    save(fig, "ch1_dot_shadow.svg")

    # three sign cases
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    cases = [((2, 1.6), "positive: shadow points along a", AQUA), ((0, 2), "zero: no shadow (90°)", MUTED),
             ((-1.6, 1.4), "negative: shadow points backwards", ORANGE)]
    for ax, (bv, title, col) in zip(axs, cases):
        a = np.array([2.5, 0]); bv = np.array(bv)
        proj = (a @ bv) / (a @ a) * a
        ax.plot([bv[0], proj[0]], [bv[1], proj[1]], color=MUTED, ls=":", lw=1.4)
        if abs(proj[0]) > 1e-9:
            ax.plot([0, proj[0]], [0, 0], color=col, lw=7, alpha=0.5, solid_capstyle="butt")
        arrow(ax, (0, 0), tuple(a), BLUE); arrow(ax, (0, 0), tuple(bv), INK2)
        ax.text(2.5, -0.35, "a", color=BLUE, ha="center"); ax.text(bv[0], bv[1] + 0.15, "b", ha="center")
        ax.set_xlim(-2, 3); ax.set_ylim(-0.6, 2.4); ax.set_aspect("equal")
        ax.set_title(f"a · b = {a @ bv:+.0f}".replace("+0", "0"), fontsize=11)
        ax.set_xlabel(title, fontsize=9.5, color=INK2)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch1_dot_signs.svg")



def ch1_cosine():
    # ---- Idea 1: arrows of different lengths, same direction, shrink to one spot ----
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]; axes_through_origin(ax)
    for v, lab in [((6, 8), "[6, 8]  length 10"), ((3, 4), "[3, 4]  length 5"), ((1.5, 2), "[1.5, 2]  length 2.5")]:
        arrow(ax, (0, 0), v, BLUE, lw=1.6)
        ax.text(v[0] + 0.25, v[1] - 0.15, lab, fontsize=9.5)
    arrow(ax, (0, 0), (5, 1), AQUA, lw=1.6); ax.text(5.1, 1.25, "[5, 1]  length 5.1", fontsize=9.5)
    ax.set_xlim(-0.5, 9); ax.set_ylim(-0.5, 8.8); ax.set_aspect("equal")
    ax.set_title("Before: arrows of different lengths", fontsize=10.5)
    ax = axs[1]; axes_through_origin(ax)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.2, ls="--")
    arrow(ax, (0, 0), (0.6, 0.8), BLUE, lw=2.6)
    ax.scatter([0.6], [0.8], s=80, color=BLUE, zorder=6, edgecolor=SURF, linewidth=2)
    ax.text(0.68, 0.9, "[0.6, 0.8]\nall three blue arrows\nshrink to this one", fontsize=9.5)
    uw = np.array([5, 1.0]) / np.hypot(5, 1)
    arrow(ax, (0, 0), tuple(uw), AQUA, lw=2.6)
    ax.scatter([uw[0]], [uw[1]], s=80, color=AQUA, zorder=6, edgecolor=SURF, linewidth=2)
    ax.text(uw[0] + 0.07, uw[1] - 0.22, "[0.98, 0.2]\na different direction,\na different spot", fontsize=9.5)
    ax.text(-1.25, -1.3, "dashed: the circle of radius 1. Every direction is one spot on it.", fontsize=9, color=INK2)
    ax.set_xlim(-1.3, 1.9); ax.set_ylim(-1.4, 1.6); ax.set_aspect("equal")
    ax.set_title("After dividing by length: every arrow has length 1", fontsize=10.5)
    save(fig, "ch1_unit.svg")

    # ---- Ideas 2 and 3: shadow of an arrow of length one on the floor = cosine = dot product ----
    angles = [0, 60, 90, 120, 180]
    fig, axs = plt.subplots(1, 5, figsize=(12, 3.5))
    for ax, deg in zip(axs, angles):
        r = np.radians(deg); b = np.array([np.cos(r), np.sin(r)])
        b = np.where(np.abs(b) < 1e-9, 0, b)
        ax.axhline(0, color=MUTED, lw=1)
        tt = np.linspace(0, np.pi, 100); ax.plot(np.cos(tt), np.sin(tt), color=GRID, lw=1.2)
        arrow(ax, (0, 0), (1, 0), BLUE, lw=2.4)
        arrow(ax, (0, 0), tuple(b * (0.999 if deg == 0 else 1)), INK2 if deg else AQUA, lw=2)
        if deg not in (0, 180):
            ax.plot([b[0], b[0]], [b[1], 0], color=MUTED, ls=":", lw=1.3)
        col = AQUA if b[0] > 0 else (ORANGE if b[0] < 0 else MUTED)
        if abs(b[0]) > 1e-9:
            ax.plot([0, b[0]], [-0.13, -0.13], color=col, lw=6, solid_capstyle="butt")
        ax.text(0, -0.42, f"shadow = {b[0]:+.1f}".replace("+0.0", "0").replace("-0.0", "0").replace("+", ""),
                ha="center", fontsize=13, fontweight="bold", color=INK)
        ax.text(0, -0.74, f"[1, 0] · [{b[0]:.2g}, {b[1]:.2g}] = {b[0]:.2g}".replace("-0,", "0,").replace("[0, -0]", "[0, 0]"),
                ha="center", fontsize=11, color=INK)
        ax.set_title(f"{deg}° apart", fontsize=13)
        ax.text(1.0, 0.07, "a", color=BLUE, fontsize=10); ax.text(b[0] * 1.12, b[1] * 1.12 + (0.06 if deg in (0, 180) else 0), "b", fontsize=10)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.9, 1.25); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("a lies flat on the floor, b is an arrow of length one, light shines straight down.\n"
                 "The shadow of b (coloured bar) is the cosine of the angle, and the dot product computes it.",
                 color=INK2, fontsize=12)
    save(fig, "ch1_cosine.svg")


def ch2b_equations():
    x = np.linspace(-1.5, 3.5, 2)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    cases = [
        ("one solution: the lines cross", [(2, 3, 6, BLUE, "2x + 3y = 6"), (-1, 1, 1, ORANGE, "−x + y = 1")], (0.6, 1.6)),
        ("no solution: parallel lines", [(2, 3, 6, BLUE, "2x + 3y = 6"), (4, 6, 3, ORANGE, "4x + 6y = 3")], None),
        ("infinitely many: the same line twice", [(2, 3, 6, BLUE, "2x + 3y = 6"), (4, 6, 12, ORANGE, "4x + 6y = 12")], None),
    ]
    for ax, (title, lines, pt) in zip(axs, cases):
        axes_through_origin(ax)
        for k, (a, b_, c, col, lab) in enumerate(lines):
            ls = "--" if (k == 1 and "same" in title) else "-"
            ax.plot(x, (c - a * x) / b_, color=col, lw=2.6 if k == 0 else 2, ls=ls, label=lab)
        if pt:
            ax.scatter([pt[0]], [pt[1]], s=80, color=INK, zorder=5)
            ax.annotate("(0.6, 1.6)", pt, xytext=(1.2, 2.4), fontsize=10, arrowprops=dict(arrowstyle="->", color=INK2))
        ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.2, 3.2); ax.set_title(title, fontsize=10.5)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlabel("x")
    axs[0].set_ylabel("y")
    save(fig, "ch2b_equations.svg")


def ch2_columns():
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    axes_through_origin(ax)
    arrow(ax, (0, 0), (2, 1), BLUE, lw=1.6, ls="--"); ax.text(1.1, -1.25, "1 scoop oats\n[2, 1]", color=BLUE, fontsize=9)
    arrow(ax, (0, 0), (1, 3), ORANGE, lw=1.6, ls="--"); ax.text(-0.25, 2.55, "1 scoop berries\n[1, 3]", color=ORANGE, fontsize=9, ha="right")
    arrow(ax, (0, 0), (6, 3), BLUE, lw=3)
    ax.text(4.0, 0.55, "3 scoops oats\n= 3 × [2, 1] = [6, 3]", color=BLUE, fontsize=10.5)
    arrow(ax, (6, 3), (8, 9), ORANGE, lw=3)
    ax.text(7.35, 5.0, "+ 2 scoops berries\n= 2 × [1, 3] = [2, 6]", color=ORANGE, fontsize=10.5)
    arrow(ax, (0, 0), (8, 9), AQUA, lw=2.2, ls="-")
    ax.scatter([8], [9], s=60, color=INK, zorder=6)
    ax.text(8.15, 9.3, "bowl: [8, 9]\n8 g protein, 9 g sugar", fontsize=10.5, fontweight="bold")
    ax.set_xlim(-2.6, 12.5); ax.set_ylim(-1.6, 10.8); ax.set_aspect("equal")
    ax.set_xlabel("protein (g)"); ax.set_ylabel("sugar (g)")
    ax.set_title("Columns view: the recipe [3, 2] says how much of each column to add", fontsize=10.5)
    save(fig, "ch2_columns.svg")

# ============================ Chapter 2 ============================
def ch2():
    tri = np.array([[0, 0], [1, 0], [0.4, 0.8]])
    mats = [(np.array([[2, 0], [0, 3]]), "stretch  [[2,0],[0,3]]"),
            (np.array([[-1, 0], [0, 1]]), "mirror  [[-1,0],[0,1]]"),
            (np.array([[0, -1], [1, 0]]), "rotate 90°  [[0,-1],[1,0]]")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.6))
    for ax, (M, title) in zip(axs, mats):
        axes_through_origin(ax)
        ax.add_patch(Polygon(tri, closed=True, fill=False, ec=INK2, ls="--", lw=1.4))
        ax.add_patch(Polygon(tri @ M.T, closed=True, color=BLUE, alpha=0.35, lw=0))
        ax.add_patch(Polygon(tri @ M.T, closed=True, fill=False, ec=BLUE, lw=1.6))
        e1, e2 = M[:, 0], M[:, 1]
        arrow(ax, (0, 0), tuple(e1), ORANGE, lw=2); arrow(ax, (0, 0), tuple(e2), AQUA, lw=2)
        ax.plot([], [], color=ORANGE, label=f"[1,0] lands on {[int(v) for v in e1]}")
        ax.plot([], [], color=AQUA, label=f"[0,1] lands on {[int(v) for v in e2]}")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=9)
        ax.set_xlim(-2.5, 3); ax.set_ylim(-1.2, 3.4); ax.set_aspect("equal"); ax.set_title(title, fontsize=10.5)
    fig.suptitle("dashed = before, shaded = after. Columns of the matrix = where the basic arrows land",
                 color=INK2, fontsize=10)
    save(fig, "ch2_transforms.svg")

    # |x| from relu
    x = np.linspace(-3, 3, 200)
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True)
    for ax, y, t, c in zip(axs, [np.maximum(0, x), np.maximum(0, -x), np.abs(x)],
                           ["ReLU(x)", "ReLU(-x)", "ReLU(x) + ReLU(-x) = |x|"], [BLUE, ORANGE, AQUA]):
        ax.plot(x, y, color=c, lw=2.5); axes_through_origin(ax); ax.set_title(t, fontsize=11)
    axs[0].text(-2.9, 2.5, "hinge 1", color=INK2); axs[1].text(0.8, 2.5, "hinge 2", color=INK2)
    axs[2].text(-1.3, 2.6, "a V shape!", color=INK2)
    save(fig, "ch2_relu_abs.svg")

    # network layer 3 -> 4
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(False); ax.axis("off")
    ins = [(0, y) for y in (2.5, 1.5, 0.5)]
    outs = [(3, y) for y in (3, 2, 1, 0)]
    for i, p in enumerate(ins):
        for j, q in enumerate(outs):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=1, alpha=0.55, zorder=1)
    for i, p in enumerate(ins):
        ax.add_patch(Circle(p, 0.25, color=AQUA, zorder=3)); ax.text(p[0] - 0.45, p[1], f"x{i+1}", ha="right", va="center")
    for j, q in enumerate(outs):
        ax.add_patch(Circle(q, 0.25, color=ORANGE, zorder=3))
        ax.text(q[0] + 0.4, q[1], f"ReLU( row {j+1} of W · x + b{j+1} )", va="center", fontsize=9.5)
    ax.text(0, -0.45, "3 inputs", ha="center", color=INK2); ax.text(3, -0.75, "4 detectors", ha="center", color=INK2)
    ax.text(1.5, 3.55, "12 lines = 12 weights (W is 4 × 3)\n+ 4 biases = 16 parameters", ha="center", color=INK)
    ax.set_xlim(-1, 7.2); ax.set_ylim(-1, 4.1); ax.set_aspect("equal")
    save(fig, "ch2_layer.svg")

    # determinant
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    for ax, M, title in zip(axs, [np.eye(2), np.array([[2, 1], [0, 1.5]]), np.array([[1, 2], [0.5, 1]])],
                            ["identity: area 1", "[[2,1],[0,1.5]]: det = 3", "[[1,2],[0.5,1]]: det = 0"]):
        axes_through_origin(ax)
        P = sq @ M.T
        ax.add_patch(Polygon(sq, fill=False, ec=INK2, ls="--", lw=1.2))
        ax.add_patch(Polygon(P, color=BLUE, alpha=0.35, lw=0)); ax.add_patch(Polygon(P, fill=False, ec=BLUE, lw=1.6))
        arrow(ax, (0, 0), tuple(M[:, 0]), ORANGE); arrow(ax, (0, 0), tuple(M[:, 1]), AQUA)
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3); ax.set_aspect("equal"); ax.set_title(title, fontsize=10.5)
    axs[2].text(0.2, 2.2, "squashed flat onto a line:\ninformation is lost", fontsize=9, color=INK2)
    fig.suptitle("The determinant = how much a matrix scales area (the unit square becomes the shaded shape)",
                 color=INK2, fontsize=10)
    save(fig, "ch2_determinant.svg")

    # eigenvectors
    M = np.array([[2, 1], [1, 2]])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    axes_through_origin(ax)
    for v, c, lab in [(np.array([1, 1]) / np.sqrt(2), AQUA, "eigenvector [1,1]: stretched ×3"),
                      (np.array([1, -1]) / np.sqrt(2), ORANGE, "eigenvector [1,-1]: stretched ×1"),
                      (np.array([1, 0]), VIOLET, "ordinary vector [1,0]: knocked off its line")]:
        w = M @ v
        ax.plot([-4 * v[0], 4 * v[0]], [-4 * v[1], 4 * v[1]], color=c, lw=0.8, alpha=0.4)
        arrow(ax, (0, 0), tuple(v), c, lw=1.6, ls="--")
        arrow(ax, (0, 0), tuple(w), c, lw=2.6)
        ax.text(w[0] + 0.1, w[1] + 0.12, lab, color=INK, fontsize=9)
    ax.set_xlim(-1.4, 4.6); ax.set_ylim(-1.6, 3.2); ax.set_aspect("equal")
    ax.set_title("A = [[2,1],[1,2]]: dashed = before, solid = after")
    save(fig, "ch2_eigen.svg")


# ============================ Chapter 3 ============================
def ch3():
    L = lambda w: (w - 5) ** 2
    w = np.linspace(-0.5, 10.5, 300)

    # the one-knob network: loss curve, current knob position, a nudge
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(w, L(w), color=INK, lw=2.2)
    ax.scatter([3], [4], s=70, color=ORANGE, zorder=5, edgecolor=SURF, linewidth=1.5)
    ax.annotate("the knob is at w = 3\nloss = 4", (3, 4), xytext=(-0.3, 15), fontsize=10, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.scatter([5], [0], s=70, color=AQUA, zorder=5, edgecolor=SURF, linewidth=1.5)
    ax.annotate("best setting: w = 5, loss = 0\n(the network doesn't know this)", (5, 0), xytext=(5.6, 8), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("the knob w"); ax.set_ylabel("loss L(w) = (w − 5)²"); ax.set_ylim(-2, 30)
    ax.set_title("A network with one knob: how wrong it is, for every setting of the knob", fontsize=10.5)
    save(fig, "ch3_knob.svg")

    # zoom in: secants from w = 3 settle on the tangent, slope -4
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ax.plot(w, L(w), color=INK, lw=2.4, label="loss L(w)")
    for w2, c in [(4, YELLOW), (3.5, AQUA)]:
        m = (L(w2) - 4) / (w2 - 3)
        t = np.linspace(1.2, 5.2, 2)
        ax.plot(t, 4 + m * (t - 3), color=c, lw=1.6, label=f"nudge to w = {w2}: slope {m:.1f}")
        ax.scatter([w2], [L(w2)], color=c, s=35, zorder=5)
    t = np.linspace(1.2, 5.2, 2)
    ax.plot(t, 4 - 4 * (t - 3), color=BLUE, lw=2.6, ls="--", label="tiny nudges settle on slope −4")
    ax.scatter([3], [4], color=ORANGE, s=60, zorder=6)
    ax.add_patch(Rectangle((2.6, 2.4), 0.8, 3.2, fill=False, ec=MUTED, lw=1.2, ls=":"))
    ax.set_xlim(0.8, 6.5); ax.set_ylim(-2, 14); ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlabel("the knob w"); ax.set_ylabel("loss")
    ax.set_title("Smaller and smaller nudges from w = 3", fontsize=10.5)
    ax = axs[1]
    wz = np.linspace(2.6, 3.4, 100)
    ax.plot(wz, L(wz), color=INK, lw=2.6, label="loss curve")
    ax.plot(wz, 4 - 4 * (wz - 3), color=BLUE, lw=1.6, ls="--", label="straight line, slope −4")
    ax.scatter([3], [4], color=ORANGE, s=60, zorder=6)
    ax.set_xlabel("the knob w (zoomed in)"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Zoomed in, the curve is nearly a straight line", fontsize=10.5)
    save(fig, "ch3_secants.svg")

    # the derivative is a slope-finder: tangents at several knob positions
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(w, L(w), color=INK, lw=2.2)
    for w0, c in [(1, ORANGE), (3, YELLOW), (5, MUTED), (7, AQUA), (9, BLUE)]:
        t = np.linspace(w0 - 1.1, w0 + 1.1, 2)
        ax.plot(t, L(w0) + 2 * (w0 - 5) * (t - w0), color=c, lw=2.4)
        ax.scatter([w0], [L(w0)], color=c, s=35, zorder=5)
        ax.text(w0 + (-1.25 if w0 < 5 else 1.25 if w0 > 5 else 0), L(w0) - (1.8 if w0 != 5 else 2.4),
                f"slope {2*(w0-5):+d}".replace("+0", "0"), ha="center", fontsize=9.5)
    ax.set_ylim(-4, 28); ax.set_xlabel("the knob w"); ax.set_ylabel("loss")
    ax.text(0.2, 25, "steep downhill:\nturn the knob up", fontsize=9, color=INK2)
    ax.text(7.6, 25, "uphill:\nturn it down", fontsize=9, color=INK2)
    ax.set_title("A different slope at every knob position: slope = 2(w − 5)", fontsize=10.5)
    save(fig, "ch3_tangents.svg")

    # why x² has derivative 2x: a growing square
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.grid(False); ax.axis("off")
    x0, h = 3.0, 0.6
    ax.add_patch(Rectangle((0, 0), x0, x0, color=BLUE_LIGHT, lw=0))
    ax.add_patch(Rectangle((x0, 0), h, x0, color=ORANGE, alpha=0.45, lw=0))
    ax.add_patch(Rectangle((0, x0), x0, h, color=ORANGE, alpha=0.45, lw=0))
    ax.add_patch(Rectangle((x0, x0), h, h, color=RED, alpha=0.6, lw=0))
    ax.add_patch(Rectangle((0, 0), x0 + h, x0 + h, fill=False, ec=INK2, lw=1.2))
    ax.text(x0 / 2, x0 / 2, "old square\narea x²", ha="center", va="center", fontsize=11)
    ax.text(x0 + h / 2, x0 / 2, "strip\nx × h", ha="center", va="center", fontsize=9.5, rotation=90)
    ax.text(x0 / 2, x0 + h / 2, "strip  x × h", ha="center", va="center", fontsize=9.5)
    ax.annotate("tiny corner h × h", (x0 + h / 2, x0 + h / 2), xytext=(x0 + 0.1, x0 + h + 0.45), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(x0 / 2, -0.3, "x", ha="center", fontsize=11); ax.text(x0 + h / 2, -0.3, "h", ha="center", fontsize=11)
    ax.text(-0.3, x0 / 2, "x", va="center", fontsize=11); ax.text(-0.3, x0 + h / 2, "h", va="center", fontsize=11)
    ax.set_xlim(-0.6, x0 + h + 0.6); ax.set_ylim(-0.6, x0 + h + 1.0); ax.set_aspect("equal")
    ax.set_title("Grow a square's side from x to x + h:\nthe area grows by two strips and a tiny corner", fontsize=10.5)
    save(fig, "ch3_square.svg")

    # integral rectangles
    g = lambda x: np.sin(x) + 1.5
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    xs = np.linspace(0, np.pi, 300)
    exact = 2 + 1.5 * np.pi
    for ax, n in zip(axs, [4, 8, 32]):
        w = np.pi / n
        mids = (np.arange(n) + 0.5) * w
        ax.bar(mids, g(mids), width=w, color=BLUE, alpha=0.35, edgecolor=BLUE, linewidth=0.8)
        ax.plot(xs, g(xs), color=INK, lw=2)
        ax.set_title(f"{n} rectangles: area ≈ {np.sum(g(mids)*w):.3f}", fontsize=10.5)
    fig.suptitle(f"Area under sin(x) + 1.5 from 0 to π. More, thinner rectangles → the exact area {exact:.3f}",
                 color=INK2, fontsize=10)
    save(fig, "ch3_integral.svg")

    # gradient field on a bowl
    fig, ax = plt.subplots(figsize=(5.6, 5))
    X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = X ** 2 + 2 * Y ** 2
    cs = ax.contour(X, Y, Z, levels=[1, 3, 6, 10, 15], colors=MUTED, linewidths=1)
    ax.clabel(cs, fontsize=8, fmt="%g")
    for px, py in [(2, 1), (-1.5, 1.5), (1, -1.6), (-2.2, -0.6), (0.4, 1.9)]:
        gx, gy = 2 * px, 4 * py
        s = 0.18
        arrow(ax, (px, py), (px + s * gx, py + s * gy), ORANGE, lw=2)
        arrow(ax, (px, py), (px - s * gx, py - s * gy), BLUE, lw=2)
        ax.scatter([px], [py], color=INK, s=18, zorder=5)
    ax.plot([], [], color=ORANGE, label="gradient ∇f: steepest uphill")
    ax.plot([], [], color=BLUE, label="−∇f: steepest downhill")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_title("f(x, y) = x² + 2y² seen from above\n(rings = equal height; arrows cross rings at right angles)", fontsize=10.5)
    save(fig, "ch3_gradient_field.svg")


# ============================ Chapter 5 ============================
def ch5():
    L = lambda w: (w - 5) ** 2
    fig, ax = plt.subplots(figsize=(6.2, 4))
    w = np.linspace(-1, 11, 300)
    ax.plot(w, L(w), color=INK, lw=2)
    ws = [0.0]
    for _ in range(8):
        ws.append(ws[-1] - 0.1 * 2 * (ws[-1] - 5))
    ws = np.array(ws)
    ax.plot(ws, L(ws), color=BLUE, lw=1.2, alpha=0.6)
    ax.scatter(ws, L(ws), color=BLUE, s=40, zorder=5, edgecolor=SURF)
    for i in range(4):
        ax.annotate(f"step {i}: w = {ws[i]:.2f}", (ws[i], L(ws[i])), textcoords="offset points",
                    xytext=(10, 2), fontsize=8.5)
    ax.annotate("minimum at w = 5", (5, 0), xytext=(6.8, 22), arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("w"); ax.set_ylabel("loss L(w) = (w − 5)²"); ax.set_title("Gradient descent, learning rate 0.1: the steps shrink as the slope flattens", fontsize=10.5)
    save(fig, "ch5_descent_1d.svg")

    fig, axs = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
    for ax, lr, c in zip(axs, [0.1, 0.5, 1.0, 1.1], [AQUA, BLUE, YELLOW, RED]):
        w = [0.0]
        for _ in range(10):
            w.append(w[-1] - lr * 2 * (w[-1] - 5))
        w = np.array(w)
        ax.plot(w, L(w), color=MUTED, lw=1)
        wl = np.linspace(-3, 13, 200)
        ax.plot(wl, L(wl), color=INK, lw=1.8)
        ax.plot(w[:7], L(w[:7]), color=c, lw=1.4, marker="o", ms=5)
        ax.set_title(f"lr = {lr}", fontsize=11)
        ax.set_xlim(-3, 13); ax.set_ylim(-3, 70)
    axs[0].set_xlabel("smooth approach"); axs[1].set_xlabel("one jump to 5"); axs[2].set_xlabel("bounces 0 ↔ 10 forever"); axs[3].set_xlabel("diverges")
    axs[0].set_ylabel("loss")
    save(fig, "ch5_learning_rates.svg")

    # 2D landscape: plain GD vs momentum on a narrow valley
    def grad(p):
        return np.array([0.2 * p[0], 4 * p[1]])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    X, Y = np.meshgrid(np.linspace(-11, 2, 300), np.linspace(-3, 3, 200))
    ax.contour(X, Y, 0.1 * X ** 2 + 2 * Y ** 2, levels=12, colors=MUTED, linewidths=0.8)
    p = np.array([-10.0, 2.0]); path = [p.copy()]
    for _ in range(40):
        p = p - 0.45 * grad(p); path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=ORANGE, marker="o", ms=3, lw=1.5, label="plain gradient descent (zig-zags)")
    p = np.array([-10.0, 2.0]); v = np.zeros(2); path = [p.copy()]
    for _ in range(40):
        v = 0.8 * v - 0.08 * grad(p); p = p + v; path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=BLUE, marker="o", ms=3, lw=1.5, label="with momentum (rolls along the valley)")
    ax.scatter([0], [0], marker="*", s=200, color=AQUA, zorder=6, label="minimum")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3); ax.set_aspect("equal")
    ax.set_title("A loss landscape from above: two parameters, 40 steps each", fontsize=10.5)
    ax.set_xlabel("parameter 1"); ax.set_ylabel("parameter 2")
    save(fig, "ch5_landscape.svg")

    # line fit progress
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 100)
    y = 3 * x + 1 + rng.normal(0, 0.1, 100)
    w, b, lr = 0.0, 0.0, 0.1
    snaps = {}
    for step in range(200):
        if step in (0, 5, 20, 199):
            snaps[step] = (w, b)
        err = w * x + b - y
        w -= lr * (2 * err * x).mean(); b -= lr * (2 * err).mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=14, color=MUTED, label="noisy data", zorder=2)
    xs = np.linspace(-1, 1, 2)
    for (step, (ww, bb)), c in zip(snaps.items(), [RED, YELLOW, AQUA, BLUE]):
        ax.plot(xs, ww * xs + bb, color=c, lw=2.2, label=f"step {step}: w = {ww:.2f}, b = {bb:.2f}")
    ax.legend(fontsize=9, loc="upper left"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Fitting y = w·x + b: the line swings into place")
    save(fig, "ch5_line_fit.svg")


# ============================ Chapter 7 ============================
def ch7():
    from itertools import product
    fig, axs = plt.subplots(1, 4, figsize=(11, 3), sharey=False)
    for ax, k in zip(axs, [1, 2, 3, 6]):
        dist = {}
        for roll in product(range(1, 7), repeat=k):
            dist[sum(roll)] = dist.get(sum(roll), 0) + 1
        xs = np.array(sorted(dist)); ps = np.array([dist[s] for s in xs]) / 6 ** k
        ax.bar(xs, ps, color=BLUE, width=0.8)
        ax.set_title(f"sum of {k} {'die' if k == 1 else 'dice'}", fontsize=11); ax.set_yticks([])
    axs[0].set_ylabel("probability")
    fig.suptitle("Add more dice and a bell appears (central limit theorem)", color=INK2, fontsize=10)
    save(fig, "ch7_dice.svg")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.linspace(-4, 4, 400)
    p = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    ax.fill_between(x, p, where=np.abs(x) <= 2, color=BLUE_LIGHT, lw=0)
    ax.fill_between(x, p, where=np.abs(x) <= 1, color="#86b6ef", lw=0)
    ax.plot(x, p, color=BLUE, lw=2.4)
    ax.text(0, 0.17, "68%\nwithin 1σ", ha="center", color=INK)
    ax.annotate("95% within 2σ", (1.6, 0.04), xytext=(2.6, 0.17), ha="center", color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.axvline(0, color=INK2, lw=1, ls="--"); ax.text(-0.1, 0.33, "mean μ", color=INK2, ha="right")
    ax.annotate("", (1, 0.245), (0, 0.245), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.text(0.5, 0.26, "σ", ha="center", color=ORANGE, fontsize=12)
    ax.set_xticks(range(-3, 4)); ax.set_xticklabels(["μ−3σ", "μ−2σ", "μ−σ", "μ", "μ+σ", "μ+2σ", "μ+3σ"])
    ax.set_yticks([]); ax.set_title("The Gaussian (normal) bell: area = probability")
    save(fig, "ch7_gaussian.svg")

    rng = np.random.default_rng(2)
    samples = rng.uniform(0, 1, (10_000, 12)).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(samples, bins=40, density=True, color=BLUE, alpha=0.55, edgecolor=SURF, linewidth=0.6, label="10,000 sums of 12 flat numbers")
    x = np.linspace(2, 10, 300)
    ax.plot(x, np.exp(-(x - 6) ** 2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, lw=2.2, label="Gaussian, mean 6, std 1")
    ax.legend(fontsize=9, loc="upper left"); ax.set_yticks([]); ax.set_title("What the Chapter 7 code produces")
    save(fig, "ch7_clt_hist.svg")


# ============================ Chapter 8 ============================
def ch8():
    p = np.linspace(0.005, 1, 400)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(p, -np.log2(p), color=BLUE, lw=2.4)
    for pp, lab in [(1, "certain: 0 bits"), (0.5, "coin flip: 1 bit"), (0.25, "1/4: 2 bits"), (1 / 16, "1/16: 4 bits")]:
        ax.scatter([pp], [-np.log2(pp)], color=ORANGE, s=40, zorder=5)
        ax.annotate(lab, (pp, -np.log2(pp)), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("probability p of what happened"); ax.set_ylabel("surprise −log₂ p (bits)")
    ax.set_title("Surprise: rare events carry more information")
    save(fig, "ch8_surprise.svg")

    q = np.linspace(0.001, 0.999, 400)
    H = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(q, H, color=BLUE, lw=2.4)
    ax.scatter([0.5, 0.99], [1, -(0.99 * np.log2(0.99) + 0.01 * np.log2(0.01))], color=ORANGE, s=40, zorder=5)
    ax.annotate("fair coin: 1 bit (max)", (0.5, 1), xytext=(0.5, 0.75), fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("99% heads: 0.08 bits", (0.99, 0.08), xytext=(0.6, 0.15), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("probability of heads"); ax.set_ylabel("entropy H (bits)")
    ax.set_title("Entropy of a coin: most surprising when fair")
    save(fig, "ch8_coin_entropy.svg")

    words = ["mat", "dog", "moon", "sofa"]
    logits = np.array([2.0, 0.5, -1.0, 1.0])
    e = np.exp(logits); probs = e / e.sum()
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, vals, t, c, fmt in [(axs[0], logits, "1. logits (any real number)", MUTED, "{:.1f}"),
                                 (axs[1], e, "2. exponentiate (all positive)", AQUA, "{:.2f}"),
                                 (axs[2], probs, "3. divide by total 12.12 (sum = 1)", BLUE, "{:.2f}")]:
        bars = ax.bar(words, vals, color=c, width=0.6)
        ax.axhline(0, color=MUTED, lw=1)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.04 * max(vals) if v >= 0 else v / 2,
                    fmt.format(v), ha="center", va="bottom" if v >= 0 else "center", fontsize=9,
                    color=INK if v >= 0 else SURF)
        ax.set_ylim(min(0, vals.min()) * 1.15, vals.max() * 1.2)
        ax.set_title(t, fontsize=10.5)
    save(fig, "ch8_softmax.svg")


# ============================ Chapter 9 ============================
def ch9():
    x = np.linspace(-4, 4, 300)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for ax, y, t, c, ylim in [(axs[0], np.maximum(0, x), "ReLU: max(0, z)", BLUE, (-1, 4)),
                               (axs[1], np.tanh(x), "tanh: squash to (−1, 1)", ORANGE, (-1.3, 1.3)),
                               (axs[2], 1 / (1 + np.exp(-x)), "sigmoid: squash to (0, 1)", AQUA, (-0.2, 1.2))]:
        axes_through_origin(ax); ax.plot(x, y, color=c, lw=2.5); ax.set_title(t, fontsize=11); ax.set_ylim(*ylim)
    axs[2].axhline(0.5, color=MUTED, ls=":", lw=1); axs[2].text(0.3, 0.38, "σ(0) = 0.5", fontsize=9)
    save(fig, "ch9_activations.svg")

    # train the XOR net exactly as in the book
    rng = np.random.default_rng(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    W1 = rng.normal(0, 1, (2, 8)); b1 = np.zeros((1, 8))
    W2 = rng.normal(0, 1, (8, 1)); b2 = np.zeros((1, 1))
    sig = lambda z: 1 / (1 + np.exp(-z))
    losses = []
    for step in range(5000):
        h = np.tanh(X @ W1 + b1); p = sig(h @ W2 + b2)
        losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        dz2 = (p - y) / len(X); dW2 = h.T @ dz2; db2 = dz2.sum(0, keepdims=True)
        dz1 = (dz2 @ W2.T) * (1 - h ** 2); dW1 = X.T @ dz1; db1 = dz1.sum(0, keepdims=True)
        W1 -= 0.1 * dW1; b1 -= 0.1 * db1; W2 -= 0.1 * dW2; b2 -= 0.1 * db2

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.2, 1.1]})
    ax = axs[0]
    for (a, b_), lab in zip(X, y.ravel()):
        ax.scatter(a, b_, s=160, color=BLUE if lab else ORANGE, marker="o" if lab else "X", zorder=5, edgecolor=SURF)
    t = np.linspace(-0.3, 1.3, 2)
    ax.plot(t, 1.5 - t, color=MUTED, ls="--", lw=1.4)
    ax.plot(t, 0.5 + 0.0 * t, color=MUTED, ls="--", lw=1.4)
    ax.text(0.5, -0.5, "every straight line gets\nat least one point wrong", ha="center", fontsize=9, color=INK2)
    ax.set_title("XOR: no line separates dots from crosses", fontsize=10.5)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.65, 1.35); ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2")

    ax = axs[1]
    G1, G2 = np.meshgrid(np.linspace(-0.35, 1.35, 200), np.linspace(-0.35, 1.35, 200))
    grid = np.c_[G1.ravel(), G2.ravel()]
    P = sig(np.tanh(grid @ W1 + b1) @ W2 + b2).reshape(G1.shape)
    im = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11), cmap="Blues")
    ax.contour(G1, G2, P, levels=[0.5], colors=INK, linewidths=1.6)
    for (a, b_), lab in zip(X, y.ravel()):
        ax.scatter(a, b_, s=160, color=BLUE if lab else ORANGE, marker="o" if lab else "X", zorder=5, edgecolor=SURF)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(output = 1)")
    ax.set_title("The trained net bends the boundary", fontsize=10.5); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("x1")

    ax = axs[2]
    ax.plot(losses, color=BLUE, lw=2)
    ax.axhline(np.log(2), color=MUTED, ls=":", lw=1.2); ax.text(4900, np.log(2) - 0.06, "ln 2 ≈ 0.69 = coin-flip guessing", fontsize=9, color=INK2, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss"); ax.set_title("Loss during training", fontsize=10.5)
    save(fig, "ch9_xor.svg")


# ============================ Chapter 10 ============================
def ch10():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    words = ["the", "cat", "sat"]; wts = [0.14, 0.58, 0.28]
    bars = ax.bar(words, wts, color=BLUE, width=0.55)
    for bb, v in zip(bars, wts):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 0.7); ax.set_ylabel("attention weight")
    ax.set_title('How much "sat" attends to each word\n(the worked example; weights sum to 1)', fontsize=10.5)

    ax = axs[1]
    rng = np.random.default_rng(3)
    toks = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(toks)
    S = rng.normal(0, 1.2, (n, n))
    S[np.triu_indices(n, 1)] = -np.inf
    Wt = np.exp(S - S.max(1, keepdims=True)); Wt /= Wt.sum(1, keepdims=True)
    im = ax.imshow(Wt, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "masked" if j > i else f"{Wt[i, j]:.2f}", ha="center", va="center", fontsize=7.5 if j > i else 8.5,
                    color=MUTED if j > i else (SURF if Wt[i, j] > 0.55 else INK))
    ax.set_xticks(range(n)); ax.set_xticklabels(toks); ax.set_yticks(range(n)); ax.set_yticklabels(toks)
    ax.set_xlabel("attends to (key)"); ax.set_ylabel("word (query)"); ax.grid(False)
    ax.set_title("A causal attention matrix (random weights):\neach row sums to 1, no peeking at the future", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "ch10_attention.svg")


# ============================ Intro timeline ============================
def timeline():
    events = [
        ("c. 300 BCE", "Euclid's Elements: geometry and distance", "la"),
        ("1614", "Napier publishes logarithms", "ca"),
        ("1637", "Descartes turns points into numbers (coordinates)", "la"),
        ("1654", "Pascal and Fermat found probability theory", "pr"),
        ("1684", "Leibniz publishes calculus (Newton's version: 1660s)", "ca"),
        ("1763", "Bayes' essay on updating beliefs is published", "pr"),
        ("1809", "Gauss: the bell curve and least squares", "pr"),
        ("1847", "Cauchy describes gradient descent", "ca"),
        ("1858", "Cayley's algebra of matrices", "la"),
        ("1943", "McCulloch and Pitts: the artificial neuron", "ai"),
        ("1948", "Shannon's information theory: entropy and the bit", "pr"),
        ("1956", "Dartmouth workshop names 'artificial intelligence'", "ai"),
        ("1958", "Rosenblatt's perceptron learns its weights", "ai"),
        ("1969", "Minsky and Papert: one layer cannot learn XOR", "ai"),
        ("1986", "Rumelhart, Hinton and Williams popularise backprop", "ca"),
        ("2012", "AlexNet wins ImageNet; deep learning takes off", "ai"),
        ("2013", "word2vec: king − man + woman ≈ queen", "la"),
        ("2014", "Adam optimiser; attention for translation", "ca"),
        ("2017", "The Transformer: 'Attention Is All You Need'", "ai"),
    ]
    colors = {"la": (BLUE, "linear algebra"), "ca": (ORANGE, "calculus and optimisation"),
              "pr": (AQUA, "probability and information"), "ai": (VIOLET, "neural networks and AI")}
    n = len(events)
    fig, ax = plt.subplots(figsize=(8, 0.42 * n + 1.2))
    ax.grid(False); ax.axis("off")
    ax.plot([0, 0], [0.4, n - 0.4], color=GRID, lw=3, zorder=1)
    for i, (year, text, k) in enumerate(events):
        y = n - 1 - i
        ax.scatter([0], [y], s=110, color=colors[k][0], zorder=3, edgecolor=SURF, linewidth=2)
        ax.text(-0.25, y, year, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(0.25, y, text, ha="left", va="center", fontsize=10.5, color=INK)
    for k, (c, lab) in colors.items():
        ax.scatter([], [], s=80, color=c, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9.5)
    ax.set_xlim(-2.2, 7); ax.set_ylim(-0.6, n - 0.4)
    ax.set_title("Where the ideas in this book come from", fontsize=12)
    save(fig, "intro_timeline.svg")


# ============================ Added topics ============================
def extra_topics():
    # ---- Ch2: transpose and tensors ----
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]; ax.axis("off"); ax.grid(False)
    M = np.array([[1, 0, 2], [3, 1, 0]])
    cols = [BLUE, ORANGE, AQUA]
    def grid(ax, x0, y0, mat, colfn, title):
        r, c = mat.shape
        for i in range(r):
            for j in range(c):
                ax.add_patch(Rectangle((x0 + j, y0 - i), 0.92, 0.92, color=colfn(i, j), alpha=0.35, lw=0))
                ax.text(x0 + j + 0.46, y0 - i + 0.46, str(mat[i, j]), ha="center", va="center", fontsize=13)
        ax.text(x0 + c / 2, y0 + 1.25, title, ha="center", fontsize=10.5)
    grid(ax, 0, 2, M, lambda i, j: cols[j], "W   (2 × 3)")
    grid(ax, 5.2, 2.5, M.T, lambda i, j: cols[i], "Wᵀ   (3 × 2)")
    ax.annotate("", (4.9, 2.2), (3.3, 2.2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    ax.text(4.1, 2.55, "flip", ha="center", color=INK2)
    ax.text(3.7, -1.3, "row i, column j  becomes  row j, column i\n(each coloured column of W becomes a row of Wᵀ)",
            ha="center", fontsize=9.5, color=INK2)
    ax.set_xlim(-0.3, 7.6); ax.set_ylim(-1.8, 4); ax.set_aspect("equal"); ax.set_title("Transpose", fontsize=11.5)

    ax = axs[1]; ax.axis("off"); ax.grid(False)
    for k, c in enumerate([BLUE, ORANGE, AQUA]):
        off = k * 0.55
        for i in range(4):
            for j in range(5):
                ax.add_patch(Rectangle((j * 0.6 + off, -i * 0.6 + off), 0.56, 0.56, facecolor=c, alpha=0.30, edgecolor=SURF, lw=1))
    ax.text(1.5, -2.55, "features (e.g. 768 numbers per token)", ha="center", fontsize=9)
    ax.text(-0.35, -0.75, "tokens", ha="right", va="center", fontsize=9, rotation=90)
    ax.annotate("", (4.15, 2.1), (3.05, 1.0), arrowprops=dict(arrowstyle="-|>", color=INK2))
    ax.text(3.75, 1.9, "batch\n(sentences)", fontsize=9, ha="right")
    ax.text(1.9, 2.75, "shape (batch, tokens, features) = (3, 4, 5)", ha="center", fontsize=9.5)
    ax.set_xlim(-0.9, 5); ax.set_ylim(-3, 3.2); ax.set_aspect("equal"); ax.set_title("A 3D tensor: a stack of matrices", fontsize=11.5)
    save(fig, "ch2_transpose_tensor.svg")

    # ---- Ch2: convolution ----
    img = np.zeros((6, 6)); img[:, 3:] = 9
    ker = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    out = np.array([[np.sum(img[i:i + 3, j:j + 3] * ker) for j in range(4)] for i in range(4)])
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [6, 3, 4]})
    for ax, data, title, cmap, vmin, vmax in [(axs[0], img, "image (6 × 6): dark | bright", "Greys_r", 0, 9),
                                               (axs[1], ker, "kernel (3 × 3)\nvertical-edge detector", "RdBu", -1.5, 1.5),
                                               (axs[2], out, "feature map (4 × 4)\nlights up at the edge", "Blues", 0, 30)]:
        ax.imshow(np.abs(data) if data is out else data, cmap=cmap, vmin=vmin, vmax=vmax)
        for (i, j), v in np.ndenumerate(data):
            light = (data is img and v < 5) or (data is out and abs(v) > 15)
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=10, color=SURF if light else INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title, fontsize=10.5)
    axs[0].add_patch(Rectangle((0.5, -0.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    axs[2].add_patch(Rectangle((0.5, -0.5), 1, 1, fill=False, ec=ORANGE, lw=3))
    fig.suptitle("Slide the kernel over the image; at each spot, multiply matching cells and add (a dot product). Orange window → orange cell.",
                 color=INK2, fontsize=9.5)
    save(fig, "ch2_convolution.svg")

    # ---- Ch2: SVD as rotate, stretch, rotate ----
    A = np.array([[1.5, 1.0], [0.0, 1.0]])
    U, S, Vt = np.linalg.svd(A)
    t = np.linspace(0, 2 * np.pi, 200); circ = np.vstack([np.cos(t), np.sin(t)])
    stages = [(np.eye(2), "1. unit circle"), (Vt, "2. rotate (Vᵀ)"), (np.diag(S) @ Vt, "3. stretch (Σ)"), (U @ np.diag(S) @ Vt, "4. rotate (U) = A")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (T, title) in zip(axs, stages):
        P = T @ circ; e1 = T @ np.array([1, 0]); e2 = T @ np.array([0, 1])
        axes_through_origin(ax)
        ax.fill(P[0], P[1], color=BLUE_LIGHT, alpha=0.8); ax.plot(P[0], P[1], color=BLUE, lw=2)
        arrow(ax, (0, 0), tuple(e1), ORANGE, lw=2); arrow(ax, (0, 0), tuple(e2), AQUA, lw=2)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.set_title(title, fontsize=10.5)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    axs[2].text(0, -2.05, f"stretch by {S[0]:.2f} and {S[1]:.2f}", ha="center", fontsize=9, color=INK2)
    fig.suptitle("SVD: every matrix is a rotation, then a stretch along perpendicular axes, then another rotation (here A = [[1.5, 1], [0, 1]])",
                 color=INK2, fontsize=9.5)
    save(fig, "ch2_svd.svg")

    import matplotlib.cbook as cbook, matplotlib.image as mpimg
    img = mpimg.imread(cbook.get_sample_data("grace_hopper.jpg")).astype(float)
    gray = img[..., :3].mean(axis=2) / 255.0
    U, S, Vt = np.linalg.svd(gray, full_matrices=False)
    m, n = gray.shape
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.6))
    for ax, k in zip(axs, [1, 5, 20, None]):
        if k is None:
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1); ax.set_title(f"original\n{m*n:,} numbers", fontsize=10)
        else:
            approx = (U[:, :k] * S[:k]) @ Vt[:k]
            ax.imshow(np.clip(approx, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"rank {k}\n{k*(m+n+1):,} numbers ({100*k*(m+n+1)/(m*n):.1f}%)", fontsize=10)
        ax.axis("off")
    fig.suptitle("Keeping only the biggest singular values: the best low-rank approximation of a photo (Grace Hopper)", color=INK2, fontsize=9.5)
    save(fig, "ch2_svd_image.svg")

    # ---- Ch5: local minima and saddles ----
    fig = plt.figure(figsize=(11, 3.9))
    ax = fig.add_subplot(1, 2, 1)
    x = np.linspace(-2.2, 3.2, 400)
    f = lambda x: 0.25 * x ** 4 - 0.45 * x ** 3 - 0.9 * x ** 2 + 1.2 * x + 2
    ax.plot(x, f(x), color=INK, lw=2.2)
    xs = np.linspace(-2.2, 3.2, 4000); d = np.gradient(f(xs), xs)
    crit = xs[1:][np.sign(d[1:]) != np.sign(d[:-1])]
    mins = sorted(crit, key=lambda c: f(c))
    gmin = [c for c in crit if f(c) == min(f(c) for c in crit)][0]
    for c in crit:
        if c == gmin:
            lab, col = "global minimum", AQUA
        elif f(c + 0.05) > f(c) and f(c - 0.05) > f(c):
            lab, col = "local minimum", ORANGE
        else:
            lab, col = "local maximum", MUTED
        ax.scatter([c], [f(c)], color=col, s=60, zorder=5, edgecolor=SURF)
        ax.annotate(lab, (c, f(c)), textcoords="offset points", xytext=(0, -18 if lab != "local maximum" else 10), ha="center", fontsize=9)
    ax.set_title("A bumpy loss: downhill steps can get stuck in the wrong valley", fontsize=10.5)
    ax.set_xlabel("parameter"); ax.set_ylabel("loss"); ax.set_ylim(-2.2, 6.5)
    ax = fig.add_subplot(1, 2, 2, projection="3d"); ax.set_facecolor(SURF)
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 40), np.linspace(-1.5, 1.5, 40))
    ax.plot_surface(X, Y, X ** 2 - Y ** 2, cmap="Blues", alpha=0.85, linewidth=0, antialiased=True)
    ax.scatter([0], [0], [0], color=ORANGE, s=60, depthshade=False)
    ax.text(0, 0, 0.6, "saddle point:\nflat, but not a minimum", fontsize=9)
    ax.set_title("A saddle: uphill one way, downhill the other", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=25, azim=-55)
    save(fig, "ch5_minima.svg")

    # ---- Ch6: confusion matrix + threshold trade-off ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]; ax.grid(False)
    cells = [[("True positive", 99, AQUA), ("False negative", 1, ORANGE)], [("False positive", 999, ORANGE), ("True negative", 98901, AQUA)]]
    for i in range(2):
        for j in range(2):
            lab, v, c = cells[i][j]
            ax.add_patch(Rectangle((j, 1 - i), 0.97, 0.97, color=c, alpha=0.25 if lab.startswith("True") else 0.35, lw=0))
            ax.text(j + 0.485, 1 - i + 0.6, f"{v:,}", ha="center", fontsize=15, fontweight="bold")
            ax.text(j + 0.485, 1 - i + 0.3, lab, ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks([0.485, 1.485]); ax.set_xticklabels(["test says sick", "test says healthy"])
    ax.set_yticks([1.485, 0.485]); ax.set_yticklabels(["really sick\n(100)", "really healthy\n(99,900)"])
    ax.tick_params(length=0); ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_title("Confusion matrix for the disease test", fontsize=10.5, pad=28)
    ax = axs[1]
    z = np.linspace(-4, 7, 400)
    neg = np.exp(-(z - 0) ** 2 / 2); pos = np.exp(-(z - 3) ** 2 / 2)
    ax.fill_between(z, neg, color=MUTED, alpha=0.3, lw=0); ax.plot(z, neg, color=INK2, lw=1.8, label="healthy (or ham)")
    ax.fill_between(z, pos, color=BLUE, alpha=0.3, lw=0); ax.plot(z, pos, color=BLUE, lw=1.8, label="sick (or spam)")
    ax.axvline(1.5, color=ORANGE, lw=2); ax.text(1.6, 1.08, "threshold", color=ORANGE, fontsize=9.5)
    ax.annotate("move right: fewer false alarms\n(precision up) but more misses\n(recall down)", (2.6, 0.85), xytext=(3.6, 0.78), fontsize=9,
                arrowprops=dict(arrowstyle="<-", color=INK2))
    ax.set_yticks([]); ax.set_xlabel("model's score"); ax.set_ylim(0, 1.25)
    ax.legend(loc="upper left", fontsize=9); ax.set_title("Every threshold trades precision against recall", fontsize=10.5)
    save(fig, "ch6_confusion.svg")

    # ---- Ch7: standardising reshapes the loss bowl ----
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (h1, h2), title, lr in [(axs[0], (25, 0.25), "raw features (scales differ ×10)", 0.039),
                                   (axs[1], (1, 1), "standardised features", 0.45)]:
        W1, W2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
        ax.contour(W1, W2, h1 * W1 ** 2 + h2 * W2 ** 2, levels=np.geomspace(0.05, 60, 12), colors=MUTED, linewidths=0.8)
        pnt = np.array([-2.5, 2.5]); path = [pnt.copy()]
        for _ in range(25):
            pnt = pnt - lr * 2 * np.array([h1, h2]) * pnt; path.append(pnt.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color=ORANGE, marker="o", ms=3.5, lw=1.4)
        ax.scatter([0], [0], marker="*", s=180, color=AQUA, zorder=5)
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"{title}\n25 steps of gradient descent", fontsize=10.5); ax.set_xlabel("weight 1"); ax.set_ylabel("weight 2")
    axs[0].text(-2.9, -2.8, "a narrow valley: the safe step size is tiny,\nso progress along weight 2 crawls", fontsize=8.5, color=INK2)
    axs[1].text(-2.9, -2.8, "a round bowl: every direction is\nequally steep, so it walks straight in", fontsize=8.5, color=INK2)
    save(fig, "ch7_standardise.svg")

    # ---- Ch7: correlation panels ----
    rng = np.random.default_rng(4)
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    x = rng.normal(0, 1, 150); xu = np.linspace(-2, 2, 150)
    sets = [(x, 0.95 * x + 0.3 * rng.normal(0, 1, 150)), (x, rng.normal(0, 1, 150)),
            (x, -0.7 * x + 0.7 * rng.normal(0, 1, 150)), (xu, xu ** 2 + 0.3 * rng.normal(0, 1, 150))]
    titles = ["strong positive", "none", "negative", "a clear pattern, yet r ≈ 0"]
    for ax, (a, b), t in zip(axs, sets, titles):
        r = np.corrcoef(a, b)[0, 1]
        ax.scatter(a, b, s=9, color=BLUE, alpha=0.7)
        ax.set_title(f"r = {r:+.2f}\n{t}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch7_correlation.svg")

    # ---- Ch8: likelihood of a coin ----
    p = np.linspace(0.001, 0.999, 400)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axs[0]; L = p ** 7 * (1 - p) ** 3
    ax.plot(p, L, color=BLUE, lw=2.4)
    for pp, c in [(0.5, MUTED), (0.7, ORANGE), (0.9, MUTED)]:
        ax.scatter([pp], [pp ** 7 * (1 - pp) ** 3], color=c, s=50, zorder=5)
        ax.annotate(f"p = {pp}: {pp**7*(1-pp)**3:.5f}", (pp, pp ** 7 * (1 - pp) ** 3), textcoords="offset points",
                    xytext=(-12 if pp < 0.7 else 8, 8), ha="right" if pp < 0.7 else "left", fontsize=8.5)
    ax.set_ylim(0, 0.0027); ax.set_xlabel("guess for P(heads)"); ax.set_ylabel("probability of the data")
    ax.set_title("Likelihood of 7 heads in 10 flips", fontsize=10.5)
    ax = axs[1]; ax.plot(p, 7 * np.log(p) + 3 * np.log(1 - p), color=AQUA, lw=2.4)
    ax.axvline(0.7, color=ORANGE, ls="--", lw=1.4); ax.text(0.72, -25, "peak at p = 0.7\n(same place)", fontsize=9)
    ax.set_ylim(-30, 0); ax.set_xlabel("guess for P(heads)"); ax.set_ylabel("log-likelihood")
    ax.set_title("Log-likelihood: same peak, friendlier numbers", fontsize=10.5)
    save(fig, "ch8_likelihood.svg")

    # ---- Ch9: underfit / good fit / overfit, and train vs validation ----
    rng = np.random.default_rng(7)
    xt = np.sort(rng.uniform(0, 1, 15)); yt = np.sin(2 * np.pi * xt) + rng.normal(0, 0.2, 15)
    xv = np.sort(rng.uniform(0, 1, 100)); yv = np.sin(2 * np.pi * xv) + rng.normal(0, 0.2, 100)
    def design(x, d):
        return np.vander(2 * x - 1, d + 1, increasing=True)
    def fit(d, lam=0.0):
        A = design(xt, d)
        return np.linalg.solve(A.T @ A + lam * np.eye(d + 1), A.T @ yt)
    xs = np.linspace(0, 1, 400)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    for ax, (d, lam, t) in zip(axs, [(1, 0, "degree 1: underfits"), (3, 0, "degree 3: about right"),
                                      (9, 0, "degree 9: overfits"), (9, 1e-3, "degree 9 + weight decay")]):
        w = fit(d, lam)
        ax.plot(xs, np.sin(2 * np.pi * xs), color=MUTED, ls="--", lw=1.2)
        ax.plot(xs, design(xs, d) @ w, color=BLUE if lam == 0 else AQUA, lw=2.2)
        ax.scatter(xt, yt, color=ORANGE, s=22, zorder=5, edgecolor=SURF)
        tr = np.mean((design(xt, d) @ w - yt) ** 2); va = np.mean((design(xv, d) @ w - yv) ** 2)
        ax.set_title(f"{t}\ntrain {tr:.3f} · new data {va:.3f}", fontsize=9.5)
        ax.set_ylim(-1.8, 1.8); ax.set_xticks([])
    axs[0].set_ylabel("y"); axs[0].text(0.02, -1.65, "dashed = true pattern", fontsize=8, color=INK2)
    save(fig, "ch9_fits.svg")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ds = range(0, 12); tr, va = [], []
    for d in ds:
        w = fit(d); tr.append(np.mean((design(xt, d) @ w - yt) ** 2)); va.append(np.mean((design(xv, d) @ w - yv) ** 2))
    ax.plot(ds, tr, color=BLUE, marker="o", ms=4, label="error on training data")
    ax.plot(ds, va, color=ORANGE, marker="o", ms=4, label="error on new data")
    ax.set_yscale("log"); ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("mean squared error (log scale)")
    best = int(np.argmin(va)); ax.axvline(best, color=MUTED, ls=":", lw=1.2)
    ax.text(best + 0.2, min(tr) * 1.5, "sweet spot", fontsize=9, color=INK2)
    ax.text(0.2, min(tr) * 1.5, "underfitting", fontsize=9, color=INK2); ax.text(8.6, min(tr) * 1.5, "overfitting", fontsize=9, color=INK2)
    ax.legend(fontsize=9, loc="upper left"); ax.set_title("Training error always falls; error on new data turns back up", fontsize=10.5)
    save(fig, "ch9_complexity.svg")

    # ---- Ch10: decoding strategies ----
    words = ["mat", "dog", "moon", "sofa"]; pr = np.array([0.61, 0.14, 0.03, 0.22])
    def topk(p, k):
        o = np.argsort(-p)[:k]; m = np.zeros_like(p); m[o] = p[o]; return m / m.sum()
    def topp(p, thr):
        o = np.argsort(-p); c = np.cumsum(p[o]); keep = o[:np.searchsorted(c, thr) + 1]
        m = np.zeros_like(p); m[keep] = p[keep]; return m / m.sum()
    greedy = np.zeros(4); greedy[0] = 1
    fig, axs = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    for ax, (vals, t, c) in zip(axs, [(pr, "model's probabilities\n(plain sampling)", BLUE), (greedy, "greedy\n(always the top word)", MUTED),
                                       (topk(pr, 2), "top-k, k = 2\n(keep the 2 best)", AQUA), (topp(pr, 0.9), "top-p, p = 0.9\n(smallest set covering 90%)", VIOLET)]):
        bars = ax.bar(words, vals, color=c, width=0.6)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
        ax.set_title(t, fontsize=10); ax.set_ylim(0, 1.15)
    axs[0].set_ylabel("chance of being picked")
    save(fig, "ch10_sampling.svg")


if __name__ == "__main__":
    timeline()
    ch0_extra()
    extra_topics()
    for fn in (ch2_columns, ch2b_equations, ch1_cosine, ch0, ch1, ch2, ch3, ch5, ch7, ch8, ch9, ch10):
        fn()
