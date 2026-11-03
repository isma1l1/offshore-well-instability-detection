import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

DATASET = Path("dataset")
CUTOFF = datetime(2017, 4, 24)
EXCLUDED_WELLS = {8, 10}
HIGHLIGHT_WELL = "WELL-00019"

FOLDERS = {0: "0", 1: "4"}
CLASS_NAMES = {0: "normal", 1: "unstable"}
COLORS = {0: "#5B9E6E", 1: "#C8414B"}
SENSORS = {"P-TPT": "bar", "T-TPT": "°C", "P-MON-CKP": "bar"}

PAIRS = {
    1: ("WELL-00001_20170202020343", "WELL-00001_20170316210000"),
    2: ("WELL-00002_20131214150127", "WELL-00002_20131209000000"),
    4: ("WELL-00004_20140807110112", "WELL-00004_20140805070029"),
}

STD_MINUTES = 3
ZOOM_MINUTES = 180
ZOOM_START = 0.0
FEATURES = ["std3m_med30m", "std3m_mean30m"]
N_FILES = 20


def load(stem, folder, columns=None):
    return pd.read_parquet(DATASET / folder / f"{stem}.parquet", columns=columns)


def to_unit(x, sensor):
    return x / 1e5 if sensor.startswith("P-") else x


def centered(tab, sensor):
    x = to_unit(tab[sensor], sensor)[tab["class"].notna()]
    return x - x.median()


def rolling_std(tab, sensor, minutes=STD_MINUTES):
    x = to_unit(tab[sensor], sensor).rolling(f"{minutes}min", min_periods=minutes * 30).std()
    return x[tab["class"].notna()].dropna()


def crop(x, minutes, start):
    if minutes is None:
        return x
    n = minutes * 60
    first = int(start * max(len(x) - n, 0))
    return x.iloc[first:first + n]


def plot_pairs(pairs, transform, what, minutes=ZOOM_MINUTES, start=ZOOM_START):
    tables = {stem: load(stem, FOLDERS[cls])
              for files in pairs.values() for stem, cls in zip(files, (0, 1))}
    zoom = "whole file" if minutes is None else f"{minutes}-min zoom at {start:.0%} of the file"
    for sensor, unit in SENSORS.items():
        fig, axs = plt.subplots(len(pairs), 2, figsize=(12, 2.4 * len(pairs)),
                                sharey="row", layout="constrained", squeeze=False)
        for row, (well, files) in enumerate(pairs.items()):
            for col, (stem, cls) in enumerate(zip(files, (0, 1))):
                ax = axs[row, col]
                signal = crop(transform(tables[stem], sensor), minutes, start)
                if signal.notna().any():
                    ax.plot(signal, color=COLORS[cls], linewidth=0.8)
                else:
                    ax.text(0.5, 0.5, f"{sensor} not available", ha="center", va="center",
                            transform=ax.transAxes)
                ax.set_title(f"Well {well} · {stem} · {CLASS_NAMES[cls]}", loc="left", fontsize=10)
                ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            axs[row, 0].set_ylabel(f"{sensor} {what} ({unit})")
        fig.suptitle(f"{sensor} {what}: normal vs unstable, same well ({zoom})", fontweight="bold")
        plt.show()


def sample_files(n_files, wells=None, seed=0):
    rng = random.Random(seed)
    sample = {}
    for cls, folder in FOLDERS.items():
        by_well = {}
        for path in sorted((DATASET / folder).glob("WELL-*.parquet")):
            well = int(path.stem[5:10])
            if well in EXCLUDED_WELLS or (wells is not None and well not in wells):
                continue
            if datetime.strptime(path.stem[11:19], "%Y%m%d") < CUTOFF:
                by_well.setdefault(well, []).append(path.stem)
        for well, stems in by_well.items():
            sample[(well, cls)] = rng.sample(stems, min(n_files, len(stems)))
    return sample


def plot_random_pairs(n_pairs=10, wells=(1, 2, 4), transform=rolling_std, what=f"{STD_MINUTES}-min std"):
    sample = sample_files(n_pairs, wells)
    for k in range(n_pairs):
        pairs = {w: (sample[(w, 0)][k], sample[(w, 1)][k]) for w in wells
                 if k < min(len(sample.get((w, 0), [])), len(sample.get((w, 1), [])))}
        if pairs:
            plot_pairs(pairs, transform, what)


def file_features(stem, folder):
    tab = load(stem, folder, columns=["P-TPT", "class"])
    std3m = (tab["P-TPT"] / 1e5).rolling(180).std()
    feats = pd.DataFrame({
        "std3m_med30m": std3m.rolling(1800).median(),
        "std3m_mean30m": std3m.rolling(1800).mean(),
    })
    feats["class"] = tab["class"].map({0: 0, 4: 1, 104: 1})
    feats["well"] = stem[:10]
    return feats.dropna(subset=["class"]).iloc[::20]


def plot_feature_distributions(df, features, floor=1e-3):
    wells = sorted(df["well"].unique())
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 2.6 * len(features) + 0.8),
                             sharex=True, squeeze=False)
    for ax, feat in zip(axes[:, 0], features):
        for i, well in enumerate(wells):
            groups = {c: df.loc[(df["well"] == well) & (df["class"] == c), feat] for c in (0, 1)}
            groups = {c: v[v > floor] for c, v in groups.items() if len(v)}
            for cls, values in groups.items():
                shift = 0 if len(groups) == 1 else (-0.18 if cls == 0 else 0.18)
                ax.boxplot(values, positions=[i + shift], widths=0.32, whis=(5, 95), showfliers=False,
                           patch_artist=True, manage_ticks=False,
                           boxprops={"facecolor": COLORS[cls], "edgecolor": "none"},
                           whiskerprops={"color": "0.45"}, capprops={"color": "0.45"},
                           medianprops={"color": "black", "linewidth": 1.5})
        if HIGHLIGHT_WELL in wells:
            j = wells.index(HIGHLIGHT_WELL)
            ax.axvspan(j - 0.45, j + 0.45, color="tab:orange", alpha=0.12, linewidth=0)
        ax.set_yscale("log")
        ax.set_ylabel(f"{feat}\n(bar)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[-1, 0]
    ax.set_xticks(range(len(wells)), [f"Well {w[-2:]}" for w in wells])
    ax.set_xlim(-0.6, len(wells) - 0.4)
    for label in ax.get_xticklabels():
        if label.get_text().endswith(HIGHLIGHT_WELL[-2:]):
            label.set_color("tab:orange")
            label.set_fontweight("bold")
    axes[0, 0].legend(handles=[Patch(color=COLORS[c], label=CLASS_NAMES[c]) for c in (0, 1)],
                      loc="upper left", ncol=2, frameon=False)
    fig.suptitle(f"P-TPT agitation per well, normal vs unstable (≤ {N_FILES} files per well and class)",
                 fontweight="bold")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_pairs(PAIRS, centered, "− median")
    plot_pairs(PAIRS, rolling_std, f"{STD_MINUTES}-min std")

    files = sample_files(N_FILES)
    eda = pd.concat([file_features(stem, FOLDERS[cls])
                     for (well, cls), stems in files.items() for stem in stems])
    plot_feature_distributions(eda, FEATURES)
