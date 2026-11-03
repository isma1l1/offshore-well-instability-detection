from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier, export_text

DATASET = Path("dataset")
WELLS = ["WELL-00001", "WELL-00004", "WELL-00002", "WELL-00005"]
N_FILES = 30

SENSORS = ["P-TPT"]
THRESHOLD = {"P-TPT": 0.1}
NAMES = ["std180_q90_30", "std180_max30", "std180_frac30"]
FEATURES = [f"{s}_{n}" for s in SENSORS for n in NAMES]


def add_features(df):
    new = {}
    for s in SENSORS:
        x = df[s] / 1e5 if s.startswith("P-") else df[s]
        std180 = x.rolling(180).std()
        r = std180.rolling(1800, min_periods=900)
        new[f"{s}_std180_q90_30"] = r.quantile(0.9)
        new[f"{s}_std180_max30"] = r.max()
        new[f"{s}_std180_frac30"] = (std180 > THRESHOLD[s]).rolling(1800, min_periods=900).mean()
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df.dropna(subset=["class"] + FEATURES)


parts = []
for folder in ["4", "0"]:
    for well in WELLS:
        for path in sorted((DATASET / folder).glob(f"{well}_*.parquet"))[:N_FILES]:
            b = add_features(pd.read_parquet(path))
            b["well"] = well
            parts.append(b)
data = pd.concat(parts)


data["y"] = data["class"].replace({4: 1})
rows = []
for idx_train, idx_val in LeaveOneGroupOut().split(data[FEATURES], data["y"], groups=data["well"]):
    train, val = data.iloc[idx_train], data.iloc[idx_val]
    model = DecisionTreeClassifier(class_weight="balanced", max_depth=5, min_samples_leaf=1000, random_state=0)
    model.fit(train[FEATURES], train["y"])
    pred = model.predict(val[FEATURES])
    y = val["y"].to_numpy()
    rows.append({
        "val_well": val["well"].iloc[0],
        "recall": pred[y == 1].mean(),
        "false_alarm_rate": pred[y == 0].mean(),
    })
print(pd.DataFrame(rows))


rows = []
for well in WELLS:
    for folder in ["0", "4"]:
        for path in sorted((DATASET / folder).glob(f"{well}_*.parquet"))[:N_FILES]:
            b = pd.read_parquet(path).dropna(subset=["class"])
            p = b["P-TPT"] / 1e5
            rows.append({
                "well": well, "folder": folder, "file": path.stem,
                "frozen_share": (p.diff() == 0).mean(),
                "std3min": p.rolling(180).std().median(),
                "amp1h": (p.rolling(3600).max() - p.rolling(3600).min()).median(),
                "state": b["state"].mode().iloc[0] if b["state"].notna().any() else None,
            })
inventory = pd.DataFrame(rows)
print(inventory.groupby(["well", "folder"])[["frozen_share", "std3min", "amp1h"]].median().round(3))
print(inventory.groupby(["well", "folder"])["state"].value_counts())


data["class"] = data["class"].replace({104: 1, 4: 1})
model = DecisionTreeClassifier(class_weight="balanced", max_depth=5, min_samples_leaf=1000,
                               random_state=1, monotonic_cst=[1] * len(FEATURES))
model.fit(data[FEATURES], data["class"])
print("depth:", model.get_depth(), "| leaves:", model.get_n_leaves())
print(dict(zip(FEATURES, model.feature_importances_.round(3))))
print(export_text(model, feature_names=FEATURES, max_depth=3))
