"""Regenerates the four manuscript figures as vector PDFs (fonts embedded).

Reads only the derived-result JSON files shipped in ../results/ (no raw data
needed). Run from this folder:  python make_figures.py
Output: Fig1.pdf ... Fig4.pdf in $FIG_OUT (default: parent folder)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("FIG_OUT", HERE.parent))

plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType fonts (editable, Elsevier-friendly)
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
})

DATASETS = ["CICIDS2017", "UNSW-NB15"]
PANELS = ["(a)", "(b)"]
SCEN = {
    "S0_behavioral_only": ("S0: Behavioral only", "tab:gray"),
    "S1_naive": ("S1: Raw identifiers", "tab:red"),
    "S2_anonymized": ("S2: IP/port generalization", "tab:orange"),
    "S3_reduced": ("S3: Feature reduction", "tab:green"),
    "S4_dp_clipped_eps100": ("S4: Laplace perturbation ($\\varepsilon_{total}$=100)", "tab:blue"),
}
# label offsets (points) per dataset/scenario, tuned so each label sits next to its marker
OFFSETS = {
    "CICIDS2017": {"S0_behavioral_only": (-8, 12), "S1_naive": (6, 12), "S2_anonymized": (8, -18),
                   "S3_reduced": (8, -16), "S4_dp_clipped_eps100": (9, 6)},
    "UNSW-NB15": {"S0_behavioral_only": (10, -16), "S1_naive": (-8, 12), "S2_anonymized": (8, 8),
                  "S3_reduced": (-6, -18), "S4_dp_clipped_eps100": (9, 6)},
}


def load(name):
    """Find a derived-result JSON either in ../results/ or in any ../pipeline/<stage>/02_outputs/."""
    bases = [HERE.parent / "results"] + sorted((HERE.parent / "pipeline").glob("*/02_outputs"))
    for base in bases:
        if (base / name).exists():
            with open(base / name, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(name)


def fig1_pipeline():
    def box(ax, x, y, w, h, text, fc, ec, fs=10.5, weight="normal", style="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.3, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                weight=weight, style=style)

    def arrow(ax, p1, p2, color, ls="-", rad=0.0, lw=1.3, alpha=1.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                                     color=color, linestyle=ls, alpha=alpha,
                                     connectionstyle=f"arc3,rad={rad}"))

    BLUE, GREEN = "#1b5e8a", "#0b8a5f"
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.5); ax.axis("off")

    # data sources and preprocessing
    box(ax, 1.9, 6.75, 2.2, 0.6, "CICIDS2017", "#fff5e8", "#d9902f", weight="bold")
    box(ax, 4.5, 6.75, 2.2, 0.6, "UNSW-NB15", "#fff5e8", "#d9902f", weight="bold")
    box(ax, 2.2, 5.6, 4.2, 0.7, "Preprocessing & stratified split", "#eaf1f5", BLUE)
    arrow(ax, (3.0, 6.75), (3.7, 6.3), BLUE)
    arrow(ax, (5.6, 6.75), (4.9, 6.3), BLUE)

    # feature-importance ranking R: computed once on S0 train, used only by S3
    box(ax, 7.1, 5.55, 3.6, 0.8,
        "Feature-importance ranking $R$\n(attack model on $S_0$ train;\nused only to define $S_3$)",
        "#e1f5ee", GREEN, fs=9.5)

    # five conditions
    w, gap, x0, ys, hs = 1.95, 0.2, 0.225, 3.85, 0.85
    labels = [("S0", "Behavioral only"), ("S1", "Raw identifiers"), ("S2", "IP / port generalization"),
              ("S3", "Feature reduction"), ("S4", "Clipped Laplace ($\\varepsilon$)")]
    xs = [x0 + i * (w + gap) for i in range(5)]
    for x, (a, b) in zip(xs, labels):
        box(ax, x, ys, w, hs, f"$\\mathbf{{{a}}}$\n{b}", "#fdeded", "#cc4a4a", fs=10.5)
        arrow(ax, (4.3, 5.6), (x + w / 2, ys + hs), BLUE, rad=0.0, alpha=0.9)
    # R -> S3 only
    arrow(ax, (xs[3] + w / 2 + 0.55, 5.55), (xs[3] + w / 2 + 0.55, ys + hs), GREEN, ls="--", lw=1.5)
    ax.text(xs[3] + w / 2 + 0.62, 5.05, "feature set", fontsize=8.5, color=GREEN, style="italic", va="center")

    # per-condition detection + attack
    yd, hd = 2.2, 1.1
    for x, (a, _) in zip(xs, labels):
        box(ax, x, yd, w, hd,
            f"Detection: RF · XGB · MLP\nAttack: classifier $A_{{{a[1]}}}$\n(trained on {a})",
            "#e3f2fc", BLUE, fs=9.0)
        arrow(ax, (x + w / 2, ys), (x + w / 2, yd + hd), BLUE)

    # outputs
    yo, ho = 0.3, 0.85
    box(ax, 0.9, yo, 4.4, ho, "Utility metrics\nF1 · AUC-ROC · FPR", "#dcf7ec", GREEN, fs=10.5)
    box(ax, 5.7, yo, 4.4, ho, "Re-identification risk\nTop-1 / Top-5 vs. baseline", "#dcf7ec", GREEN, fs=10.5)
    for x in xs:
        cx = x + w / 2
        arrow(ax, (cx, yd), (3.1, yo + ho), BLUE, rad=0.0, lw=1.0, alpha=0.8)
        arrow(ax, (cx, yd), (7.9, yo + ho), GREEN, rad=0.0, lw=1.0, alpha=0.8)

    ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.6, marker=">", ms=6, label="Detection flow"),
                       Line2D([0], [0], color=GREEN, lw=1.6, marker=">", ms=6, label="Attack / risk flow"),
                       Line2D([0], [0], color=GREEN, lw=1.6, ls="--", label="Feature set definition")],
              loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=9.5)
    fig.savefig(OUT / "Fig1.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_feature_importance():
    attack = load("baseline_attack_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds, panel in zip(axes, DATASETS, PANELS):
        ranking = attack[ds]["feature_importance_ranking"][:12][::-1]
        ax.barh([f for f, _ in ranking], [v for _, v in ranking], color="mediumorchid")
        ax.set_xlabel("RF impurity-based importance ($S_0$ attack model)")
        ax.set_title(f"{panel} {ds}", loc="left", fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "Fig2.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_epsilon_sweep():
    dp = load("scenario4_dp_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds, panel in zip(axes, DATASETS, PANELS):
        sweep = dp[ds]["epsilon_sweep_xgboost"]
        eps = [p["epsilon_total"] for p in sweep]
        f1 = [p["detection_xgboost"]["f1"] for p in sweep]
        risk = [p["privacy_risk"]["top1_accuracy"] * 100 for p in sweep]
        ax2 = ax.twinx()
        l1, = ax.plot(eps, f1, "o-", color="tab:blue", label="Detection F1", lw=2, ms=7)
        l2, = ax2.plot(eps, risk, "s--", color="tab:red", label="Attack top-1 risk (%)", lw=2, ms=7)
        ax.set_xscale("log")
        ax.set_xlabel("$\\varepsilon_{total}$ (nominal composed perturbation budget, log scale)")
        ax.set_ylabel("Detection F1", color="tab:blue")
        ax2.set_ylabel("Attack top-1 risk (%)", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"{panel} {ds}", loc="left", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(handles=[l1, l2], loc="center right", fontsize=9.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "Fig3.pdf", bbox_inches="tight")
    plt.close(fig)


def fig4_pareto():
    comp = load("cross_scenario_comparison.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, ds, panel in zip(axes, DATASETS, PANELS):
        for scen, (label, color) in SCEN.items():
            f1 = np.array(comp[ds]["f1_by_scenario"][scen])
            risk = np.array(comp[ds]["risk_by_scenario"][scen]) * 100
            ax.errorbar(risk.mean(), f1.mean(), xerr=risk.std(), yerr=f1.std(), fmt="o", ms=10,
                        capsize=3, elinewidth=1.2, color=color, label=label)
            ax.annotate(label.split(":")[0], (risk.mean(), f1.mean()), textcoords="offset points",
                        xytext=OFFSETS[ds][scen], fontsize=10, fontweight="bold")
        ax.set_xlabel("Re-identification attack risk, top-1 (%)")
        ax.set_ylabel("Detection F1 (XGBoost)")
        ax.set_title(f"{panel} {ds}", loc="left", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.annotate("preferred\ndirection", xy=(0.06, 0.92), xycoords="axes fraction", ha="left",
                    va="top", fontsize=8.5, style="italic", color="dimgray",
                    arrowprops=dict(arrowstyle="->", color="dimgray"), xytext=(0.28, 0.74))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "Fig4.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig2_feature_importance()
    fig3_epsilon_sweep()
    fig4_pareto()
    print("Saved Fig1-4.pdf to", OUT)
