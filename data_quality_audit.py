from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATASET = Path("dataset")
LABELS = ["class", "state"]
MAX_BAD_SHARE = 0.7
STATUSES = ["good", "frozen"]
COLORS = {"good": "#4E754C", "frozen": "#9F4B59"}
TITLES = {"0": "Sensor data quality in normal operation",
          "4": "Sensor data quality during flow instability"}


def frozen_variables(tab):
    df = tab.drop(columns=LABELS)
    mask = (df.diff() == 0).iloc[1:].all()
    return df.columns[mask]


def missing_variables(tab):
    df = tab.drop(columns=LABELS)
    mask = df.isna().all()
    return df.columns[mask]


def pos_clear_variable(tab):
    excluded = set(frozen_variables(tab)) | set(missing_variables(tab))
    return tab.columns[~tab.columns.isin(excluded)]


def missing_rate(tab):
    cols = pos_clear_variable(tab)
    return tab[cols].isna().mean()


def frozen_rate(tab, min_run=60):
    cols = pos_clear_variable(tab)
    rates = {}
    for col in cols:
        x = tab[col]
        new_run = x.ne(x.shift())
        run_id = new_run.cumsum()
        run_length = run_id.map(run_id.value_counts())
        rates[col] = ((run_length >= min_run) & x.notna()).mean()
    return pd.Series(rates)


def collect_quality(folder, max_bad_share):
    rows = []
    for path in sorted((DATASET / folder).glob("WELL-*.parquet")):
        tab = pd.read_parquet(path)
        tab = tab[tab["class"].notna()]
        sensors = [c for c in pos_clear_variable(tab) if c not in LABELS]
        missing, frozen = missing_rate(tab), frozen_rate(tab)
        for sensor in sensors:
            if missing[sensor] + frozen[sensor] >= max_bad_share:
                continue
            rows.append({"well": path.stem[:10], "file": path.stem, "sensor": sensor,
                         "hours": len(tab) / 3600,
                         "missing": missing[sensor], "frozen": frozen[sensor]})
    quality = pd.DataFrame(rows)
    quality["good"] = 1 - quality["missing"] - quality["frozen"]
    return quality


def summarize(quality):
    def weighted_share(group):
        w = group["hours"]
        return pd.Series({s: 100 * (group[s] * w).sum() / w.sum() for s in STATUSES})

    summary = quality.groupby(["well", "sensor"]).apply(weighted_share)
    summary["n_files"] = quality.groupby(["well", "sensor"])["file"].nunique()
    return summary


def plot_quality(summary, quality, folder):
    wells = sorted(summary.index.get_level_values("well").unique())
    files_per_well = quality.groupby("well")["file"].nunique()
    ncols = 3
    nrows = -(-len(wells) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharex=True, squeeze=False)

    for ax, well in zip(axs.flat, wells):
        s = summary.loc[well].sort_values("good")
        labels = [f"{sensor}  ({n}/{files_per_well[well]})" for sensor, n in s["n_files"].items()]
        left = 0
        for status in STATUSES:
            ax.barh(labels, s[status], left=left, color=COLORS[status], label=status)
            left = left + s[status]
        for y, good in enumerate(s["good"]):
            ax.text(2, y, f"{good:.0f} %", va="center", fontsize=7, color="white")
        ax.set_title(f"{well}  ({files_per_well[well]} files)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", labelsize=7)

    for ax in axs.flat[len(wells):]:
        ax.axis("off")
    for ax in axs[-1, :]:
        ax.set_xlabel("Time with usable vs frozen signal (%)")
    handles, names = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle(f"Folder {folder}: {TITLES[folder]}", fontsize=13)
    fig.text(0.01, 0.005,
             f"Kept: (file, sensor) pairs where the sensor is neither missing nor frozen over the whole file "
             f"and missing + frozen < {MAX_BAD_SHARE:.0%} of the time. Frozen = run of identical values >= 60 s. "
             f"Shares weighted by file duration. (k/n) = files kept / files of the well.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.show()


for folder in ["0", "4"]:
    quality = collect_quality(folder, MAX_BAD_SHARE)
    summary = summarize(quality)
    print(summary.round(1))
    plot_quality(summary, quality, folder)
