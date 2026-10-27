"""README figures, light and dark versions, from results/ (and set-A run tables for the heatmap).

Usage: python -m analysis.figures
Writes docs/figures/<name>-light.png and <name>-dark.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

OUT = Path("docs/figures")
RES = Path("results")
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", box="#f0efec",
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", box="#2c2c2a",
                 ramp=["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]),
}


def style(t):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["muted"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def title(ax, t, main, sub):
    ax.set_title(main, loc="left", fontsize=12, fontweight="bold", color=t["ink"], pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=t["ink2"], va="bottom")


def save(fig, name, mode):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def overview(t, mode):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 3.95); ax.axis("off")

    def box(x, y, w, h, head, body, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=t["box"], ec=edge, lw=1.5))
        ax.text(x + w / 2, y + h - 0.22, head, ha="center", va="top", fontsize=10, fontweight="bold", color=t["ink"])
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.5, color=t["ink2"], linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.5, shrinkA=2, shrinkB=2))

    box(0.0, 1.15, 2.1, 1.7, "Agent conversations", "τ-bench banking\n20 models · 97 tasks\n7,760 runs\n(re-scored labels)", t["axis"])
    ys = [3.0, 2.05, 1.1, 0.15]
    box(3.0, ys[0], 4.0, 0.85, "Trace facts (code)", "tool counts · documents in context · repeats", t["s1"])
    box(3.0, ys[1], 4.0, 0.85, "Jev step judgments", "stuck · progress · premature completion · +5", t["s1"])
    box(3.0, ys[2], 4.0, 0.85, "User pushback (Jev)", "corrects · repeats · complains · asks for human", t["s1"])
    box(3.0, ys[3], 4.0, 0.85, "Task prior", "other models' failure rate on the same task", t["s2"])
    box(7.9, 1.15, 2.1, 1.7, "Outcome", "pass / fail\n(database grading)", t["axis"])
    for y in ys:
        arrow(2.1, 2.0, 3.0, y + 0.42)
        arrow(7.0, y + 0.42, 7.9, 2.0)
    ax.text(5.0, -0.35, "Question: does any trace signal predict failure better than knowing the task?",
            ha="center", fontsize=9, color=t["ink2"], style="italic")
    save(fig, "overview", mode)


def heatmap(t, mode):
    kept = json.loads(Path("local/data/tau/kept_A.json").read_text(encoding="utf-8"))
    rows = []
    for sub, meta in kept.items():
        r = pd.read_parquet(Path("local/work-rescored") / sub / "runs.parquet")
        g = r.groupby("task_id").reward.mean()
        name = meta.get("model", sub) + (" (no reasoning)" if "-none_" in sub else "")
        rows.append(g.rename(name))
    m = pd.DataFrame(rows)
    assert m.index.is_unique and len(m) == len(kept), "duplicate or missing model rows"
    m = m.iloc[np.argsort(-m.mean(axis=1).to_numpy(), kind="stable"), np.argsort(-m.mean(axis=0).to_numpy(), kind="stable")]
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(m.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(m.index)), m.index, fontsize=8)
    ax.set_xticks([]); ax.set_xlabel("97 tasks, sorted by mean pass rate across models  →  harder", color=t["ink2"])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("pass rate (4 trials)", color=t["ink2"]); cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"])
    title(ax, t, "The task decides most of the outcome",
          "Each cell: share of 4 trials a model passed on a task. Vertical bands = task difficulty; "
          "13 tasks fail for every model.")
    save(fig, "task-heatmap", mode)


def auroc(t, mode):
    c2 = pd.read_csv(RES / "c2_judgments.csv")
    c2 = c2[(c2.section == "auroc") & (c2.model == "MEAN OF MODELS") & (c2.k == 2)].set_index("feature_set")
    order = [("C", "Task prior (other models)"), ("C+F", "Prior + trace facts"), ("C+J", "Prior + Jev judgments"),
             ("C+J2", "Prior + Jev, 2 features"), ("F", "Trace facts alone"), ("J", "Jev judgments alone")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvline(c2.loc["C", "auroc"], color=t["s2"], lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0.5, color=t["axis"], lw=1, zorder=1)
    ax.text(0.505, 2.5, "chance", color=t["muted"], fontsize=8, ha="left", va="center")
    for i, (k, label) in enumerate(order):
        y = len(order) - 1 - i
        r = c2.loc[k]
        c = t["s2"] if k == "C" else t["s1"]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r.auroc], [y], s=64, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(r.ci_high + 0.008, y, f"{r.auroc:.3f}", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(range(len(order)), [l for _, l in reversed(order)])
    ax.set_xlim(0.45, 0.95); ax.set_xlabel("AUROC for predicting failure (95% interval, tasks resampled)")
    ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    title(ax, t, "Nothing in the trace beats knowing the task",
          "4 models, 1,550 runs, first 2 user turns. Trace signals carry some information alone, none on top of the prior.")
    save(fig, "auroc", mode)


def retrieval(t, mode):
    p = pd.read_csv(RES / "multi_model_pooled_A.csv")
    p = p[p.quantity.str.startswith("MH_risk_diff")]
    items = [("3.3 all_in_context", "mixed strata only", "All required docs in context — same task & model"),
             ("3.3 all_in_context", "all strata", "All required docs in context — all strata"),
             ("3.3 all_opened", "mixed strata only", "All required docs deliberately opened — same task & model"),
             ("3.3 all_opened", "all strata", "All required docs deliberately opened — all strata")]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.axvline(0, color=t["axis"], lw=1)
    for i, (sec, note, label) in enumerate(items):
        r = p[(p.section == sec) & (p.note == note)].iloc[0]
        y = len(items) - 1 - i
        c = t["s1"]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r.estimate], [y], s=64, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(r.ci_high + 0.01, y, f"{r.estimate:+.2f}", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(range(len(items)), [l for *_, l in reversed(items)], fontsize=9)
    ax.set_xlim(-0.5, 0.08); ax.set_xlabel("Change in failure risk (Mantel–Haenszel, 95% interval)")
    ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    title(ax, t, "Within a task, retrieval goes with success",
          "20 models. Comparing runs of the same task and model removes task difficulty. Association, not cause.")
    save(fig, "retrieval", mode)


def pushback(t, mode):
    w = pd.read_csv(RES / "c1_window_check.csv")
    w = w[w.measure.str.startswith("pushback per user message")].set_index("threshold")
    ths = [0.5, 0.8]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(ths)); bw = 0.34
    for j, (col, name, c) in enumerate((("passed", "Passed runs", t["s1"]), ("failed", "Failed runs", t["s2"]))):
        vals = [w.loc[th, col] for th in ths]
        bars = ax.bar(x + (j - 0.5) * (bw + 0.02), vals, bw, color=c, label=name, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=9, color=t["ink"])
    for i, th in enumerate(ths):
        r = w.loc[th]
        ax.text(x[i], -0.075, f"difference {r.failed_minus_passed:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                ha="center", fontsize=8, color=t["ink2"], transform=ax.get_xaxis_transform() if False else ax.transData)
    ax.set_xticks(x, ["threshold 0.5 (set in advance)", "threshold 0.8 (post hoc)"])
    ax.set_ylim(0, 0.36); ax.set_ylabel("pushback per user message")
    ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", labelcolor=t["ink"])
    ax.tick_params(axis="x", length=0, pad=18, colors=t["ink2"])
    title(ax, t, "Pushback is not an early warning of failure",
          "Before the agent's decisive action (4 models, 1,550 runs), failed conversations show no more pushback than passed ones.")
    save(fig, "pushback", mode)


def main():
    for mode, t in THEMES.items():
        style(t)
        overview(t, mode); heatmap(t, mode); auroc(t, mode); retrieval(t, mode); pushback(t, mode)
    print("\n".join(sorted(str(p) for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
