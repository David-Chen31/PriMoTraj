#!/usr/bin/env python3
"""Plot Nature-style Porto Pareto figures for the ICDM paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 5.9,
        "axes.linewidth": 0.72,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelcolor": "#222222",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "gate": "#009E73",
    "lstm": "#0F4D92",
    "patch": "#D55E00",
    "time": "#8E6BBE",
    "dlinear": "#B64342",
    "phys_dark": "#4D4D4D",
    "phys_mid": "#7F7F7F",
    "phys_light": "#B7B7B7",
    "grid": "#DADADA",
    "text": "#222222",
}

STYLE = {
    "LiteMoTrajGate": {"color": COLORS["gate"], "marker": "*", "size": 98, "z": 7, "label": "PriMoTraj"},
    "PriMoTraj": {"color": COLORS["gate"], "marker": "*", "size": 98, "z": 7, "label": "PriMoTraj"},
    "LSTM": {"color": COLORS["lstm"], "marker": "o", "size": 45, "z": 6, "label": "LSTM"},
    "PatchTST": {"color": COLORS["patch"], "marker": "s", "size": 45, "z": 5, "label": "PatchTST"},
    "TimeMixer": {"color": COLORS["time"], "marker": "D", "size": 42, "z": 5, "label": "TimeMixer"},
    "DLinear": {"color": COLORS["dlinear"], "marker": "v", "size": 42, "z": 4, "label": "DLinear"},
    "CV-KF": {"color": COLORS["phys_dark"], "marker": "X", "size": 39, "z": 3, "label": "physics priors"},
    "CVMix": {"color": COLORS["phys_dark"], "marker": "h", "size": 39, "z": 3, "label": "physics priors"},
    "CV": {"color": COLORS["phys_mid"], "marker": ">", "size": 34, "z": 2, "label": "physics priors"},
    "Last": {"color": COLORS["phys_light"], "marker": "<", "size": 34, "z": 2, "label": "physics priors"},
}

LATENCY_NAME = {
    "PhysicsLast": "Last",
    "PhysicsCV": "CV",
    "PhysicsCVMix": "CVMix",
    "PhysicsCVKF": "CV-KF",
}

LABEL_OFFSETS = {
    ("params_plot", "PriMoTraj"): (-6, -1, "right"),
    ("params_plot", "LiteMoTrajGate"): (5, -1),
    ("params_plot", "LSTM"): (5, 1),
    ("params_plot", "PatchTST"): (5, 1),
    ("params_plot", "DLinear"): (5, 0),
    ("cpu_b1_p50_ms", "PriMoTraj"): (5, -1),
    ("cpu_b1_p50_ms", "LiteMoTrajGate"): (5, -1),
    ("cpu_b1_p50_ms", "LSTM"): (5, 1),
    ("cpu_b1_p50_ms", "PatchTST"): (5, 1),
    ("cpu_b1_p50_ms", "DLinear"): (5, 0),
}


def load_data(main_csv: Path, latency_csv: Path) -> pd.DataFrame:
    main = pd.read_csv(main_csv)
    lat = pd.read_csv(latency_csv)
    lat = lat[(lat["device"] == "cpu") & (lat["batch_size"] == 1) & (lat["latency_scope"] == "end_to_end")].copy()
    lat["model"] = lat["model"].map(lambda x: LATENCY_NAME.get(x, x))
    lat["pred_len"] = lat["pred_len"].astype(int)
    lat["cpu_b1_p50_ms"] = lat["p50_ms"].astype(float)
    merged = main.merge(lat[["model", "pred_len", "cpu_b1_p50_ms"]], on=["model", "pred_len"], how="left")
    merged["params_plot"] = merged["params"].clip(lower=1)
    merged["pred_len"] = merged["pred_len"].astype(int)
    return merged


def style_axis(ax, grid_axis="both") -> None:
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.42, alpha=0.55, zorder=0)
    ax.tick_params(length=2.6, width=0.65, pad=1.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.72)


def draw_point(ax, row, x_col: str):
    cfg = STYLE[row["model"]]
    h = int(row["pred_len"])
    ade_std = pd.to_numeric(row.get("ade_std"), errors="coerce")
    if pd.notna(ade_std) and ade_std > 0:
        ax.errorbar(
            row[x_col], row["ade_mean"], yerr=ade_std,
            fmt="none", ecolor=cfg["color"], elinewidth=0.68,
            capsize=2.0, capthick=0.68, alpha=0.82,
            zorder=cfg["z"] - 0.2,
        )
    face = cfg["color"] if h == 6 else "white"
    alpha = 0.94 if row["model"] in {"PriMoTraj", "LiteMoTrajGate", "LSTM", "PatchTST", "TimeMixer", "DLinear"} else 0.64
    ax.scatter(
        row[x_col],
        row["ade_mean"],
        s=cfg["size"],
        marker=cfg["marker"],
        facecolor=face,
        edgecolor=cfg["color"],
        linewidth=0.95,
        alpha=alpha,
        zorder=cfg["z"],
    )


def annotate_anchor(ax, row, x_col: str) -> None:
    if row["model"] not in {"PriMoTraj", "LiteMoTrajGate", "LSTM", "PatchTST", "DLinear"}:
        return
    if int(row["pred_len"]) != 12:
        return
    _off = LABEL_OFFSETS.get((x_col, row["model"]), (5, 1))
    dx, dy = _off[0], _off[1]
    ha = _off[2] if len(_off) > 2 else "left"
    text = STYLE[row["model"]]["label"]
    ax.annotate(
        text,
        (row[x_col], row["ade_mean"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=5.35,
        color=COLORS["text"],
        ha=ha,
        va="center",
        zorder=8,
    )


def plot_panel(ax, df: pd.DataFrame, x_col: str, xlabel: str, panel_label: str) -> None:
    for model in STYLE:
        group = df[df["model"].eq(model)].sort_values("pred_len")
        if group.empty:
            continue
        if len(group) > 1:
            ax.plot(group[x_col], group["ade_mean"], color=STYLE[model]["color"], lw=0.52, alpha=0.24, zorder=1)
        for _, row in group.iterrows():
            draw_point(ax, row, x_col)
            annotate_anchor(ax, row, x_col)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ADE (m)")
    ax.set_ylim(220, 1080)
    ax.text(-0.12, 1.025, panel_label, transform=ax.transAxes, fontsize=7.7, fontweight="bold", va="bottom")
    style_axis(ax)


def add_group_notes(axes):
    axes[0].text(0.058, 0.065, "physics priors", transform=axes[0].transAxes,
                 fontsize=5.55, color="#606060", ha="left", va="bottom")
    axes[1].text(0.020, 0.065, "physics priors", transform=axes[1].transAxes,
                 fontsize=5.55, color="#606060", ha="left", va="bottom")


def make_legend(fig) -> None:
    handles = [
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor=COLORS["gate"], markeredgecolor=COLORS["gate"], markersize=6.2),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["lstm"], markeredgecolor=COLORS["lstm"], markersize=4.4),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["patch"], markeredgecolor=COLORS["patch"], markersize=4.2),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=COLORS["time"], markeredgecolor=COLORS["time"], markersize=4.0),
        Line2D([0], [0], marker="v", linestyle="none", markerfacecolor=COLORS["dlinear"], markeredgecolor=COLORS["dlinear"], markersize=4.3),
        Line2D([0], [0], marker="h", linestyle="none", markerfacecolor=COLORS["phys_dark"], markeredgecolor=COLORS["phys_dark"], markersize=4.2, alpha=0.72),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#5A5A5A", markeredgecolor="#5A5A5A", markersize=4.0),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#5A5A5A", markersize=4.0),
    ]
    labels = ["PriMoTraj", "LSTM", "PatchTST", "TimeMixer", "DLinear", "physics", "H=6", "H=12"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.52, -0.006),
               columnspacing=0.90, handletextpad=0.34)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_csv", default="benchmarks/results/p0_main_table_v2.csv")
    parser.add_argument("--latency_csv", default="benchmarks/results/efficiency_profile_full_cpu_b1_v4.csv")
    parser.add_argument("--out_dir", default="benchmarks/figures")
    parser.add_argument("--prefix", default="formal_pareto_v4_porto-15s-s12")
    args = parser.parse_args()

    df = load_data(Path(args.main_csv), Path(args.latency_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.00), dpi=220)
    plot_panel(axes[0], df, "params_plot", "Trainable parameters", "a")
    plot_panel(axes[1], df[df["cpu_b1_p50_ms"].notna()], "cpu_b1_p50_ms", "CPU b=1 latency (ms)", "b")
    axes[0].set_xlim(0.75, 5.0e5)
    axes[1].set_xlim(0.028, 2.6)
    axes[0].set_xticks([1, 1e2, 1e4, 1e5])
    axes[1].set_xticks([0.03, 0.1, 0.3, 1.0, 2.0])
    axes[1].set_xticklabels(["0.03", "0.1", "0.3", "1", "2"])
    add_group_notes(axes)
    make_legend(fig)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.985, bottom=0.188, hspace=0.36)

    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{args.prefix}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=600)
        print(f"saved={path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
