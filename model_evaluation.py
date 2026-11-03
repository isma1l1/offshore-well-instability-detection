from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score)

DATASET = Path("dataset")
features = ["std3m_med30m", "std3m_mean30m"]

parts = []
for folder in ["0", "4"]:
    for file_path in sorted((DATASET / folder).glob("*.parquet")):
        b = pd.read_parquet(file_path, columns=["P-TPT", "class"])
        std3m = (b["P-TPT"] / 1e5).rolling(180).std()
        b["std3m_med30m"] = std3m.rolling(1800, min_periods=900).median()
        b["std3m_mean30m"] = std3m.rolling(1800, min_periods=900).mean()
        b["well"] = file_path.stem[:10]
        parts.append(b.dropna(subset="class")[features + ["class", "well"]].iloc[::20])

a_ref = pd.concat(parts).sort_index()
a_ref["class"] = a_ref["class"].replace({104: 1, 4: 1})


train_Fold1 = a_ref.loc[:"2014-09-23"]
val_Fold1 = a_ref.loc["2014-10-08":"2017-02-21"]

train_Fold2 = a_ref.loc[:"2017-02-21"]
val_Fold2 = a_ref.loc["2017-03-08":"2017-04-23"]

train = a_ref.loc[:"2017-04-23"]
test = a_ref.loc["2017-05-08":]


def score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0)}


GRID = {"max_depth": [2, 3, 5, None],
        "min_samples_leaf": [1, 0.001, 0.01, 0.05],
        "max_features": [1, 2, None]}


def make_model(params):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", monotonic_cst=[1] * len(features),
                                  n_jobs=-1, random_state=1, **params)


def best_f2_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f2 = 5 * prec * rec / (4 * prec + rec + 1e-12)
    return thr[np.argmax(f2[:-1])]


def grid_search(train, val, grid):
    results = []
    for values in product(*grid.values()):
        params = dict(zip(grid, values))
        model = make_model(params).fit(train[features], train["class"])
        proba = model.predict_proba(val[features])[:, 1]
        threshold = best_f2_threshold(val["class"], proba)
        results.append({**params, "threshold": threshold,
                        **score(val["class"], (proba >= threshold).astype(int))})
    return pd.DataFrame(results).sort_values("f2", ascending=False)


def best_params(results):
    best = results.iloc[0]
    leaf = best["min_samples_leaf"]
    params = {"max_depth": None if pd.isna(best["max_depth"]) else int(best["max_depth"]),
              "min_samples_leaf": int(leaf) if leaf >= 1 else float(leaf),
              "max_features": None if pd.isna(best["max_features"]) else int(best["max_features"])}
    return params, float(best["threshold"])


HOURS_PER_ROW = 20 / 3600


def predict_fold(train, val, params, threshold):
    model = make_model(params).fit(train[features], train["class"])
    proba = model.predict_proba(val[features])[:, 1]
    return val.assign(alarm=(proba >= threshold).astype(int))


def per_well_table(val):
    false_alarms = ((val["class"] == 0) & (val["alarm"] == 1)).sum()
    rows = []
    for (well, cls), g in val.groupby(["well", "class"]):
        rows.append({"well": well[-2:],
                     "class": "unstable" if cls == 1 else "normal",
                     "hours": round(len(g) * HOURS_PER_ROW, 1),
                     "alarm rate": g["alarm"].mean(),
                     "share of false alarms": g["alarm"].sum() / false_alarms if cls == 0 and false_alarms else float("nan")})
    return pd.DataFrame(rows).set_index(["well", "class"])


def fold_report(folds, params, threshold):
    scores, wells = {}, {}
    for name, train, val in folds:
        val = predict_fold(train, val, params, threshold)
        scores[name] = score(val["class"], val["alarm"])
        wells[name] = per_well_table(val)
    return pd.DataFrame(scores).T, pd.concat(wells, names=["fold"])


def plot_scores(scores):
    cols = ["precision", "recall", "f1", "false_alarm_rate"]
    ax = scores[cols].T.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Scores")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_per_well(wells):
    folds = wells.index.get_level_values("fold").unique()
    fig, axes = plt.subplots(1, len(folds), figsize=(6 * len(folds), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fold in zip(axes, folds):
        w = wells.loc[fold].reset_index()
        labels = "well " + w["well"] + "\n" + w["class"]
        colors = w["class"].map({"unstable": "tab:green", "normal": "tab:red"})
        bars = ax.bar(labels, w["alarm rate"], color=colors)
        ax.bar_label(bars, labels=[f"{r:.0%}\n{h:.0f} h" for r, h in zip(w["alarm rate"], w["hours"])], fontsize=8)
        ax.set_title(fold)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("alarm rate")
    fig.legend(handles=[Patch(color="tab:green", label="unstable: detection rate (higher = better)"),
                        Patch(color="tab:red", label="normal: false-alarm rate (lower = better)")],
               loc="upper center", ncol=2)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


tr1, va1 = train_Fold1.dropna(subset=features), val_Fold1.dropna(subset=features)
tr2, va2 = train_Fold2.dropna(subset=features), val_Fold2.dropna(subset=features)
train, test = train.dropna(subset=features), test.dropna(subset=features)

results = grid_search(tr1, va1, GRID)
print("Top 10 on fold-1 validation:")
print(results.head(10).round(3).to_string(index=False))

params, THRESHOLD = best_params(results)
print("\nBest (chosen on fold 1):", params, "| threshold:", round(THRESHOLD, 3))

scores, wells = fold_report([("fold 1", tr1, va1), ("fold 2", tr2, va2)], params, THRESHOLD)
print("\nScores per fold:")
print(scores.round(3).to_string())
print("\nPer well (alarm rate = detection on unstable rows, false alarms on normal rows):")
print(wells.round(3).to_string())
plot_scores(scores)
plot_per_well(wells)

scores, wells = fold_report([("test", train, test)], params, THRESHOLD)
print("\nTest scores:")
print(scores.round(3).to_string())
print("\nPer well (alarm rate = detection on unstable rows, false alarms on normal rows):")
print(wells.round(3).to_string())
plot_scores(scores)
plot_per_well(wells)
