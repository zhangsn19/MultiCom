from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features
from .metrics import classification_metrics
from .schema import INT_TO_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe OOF aggregation for MultiCom.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing agent_votes.csv and pilot_notes.csv.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_lr(c: float, class_weight, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def align_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = prob[:, i]
    return out


def choose_spec(X_train: pd.DataFrame, y_train: np.ndarray, inner_folds: int, seed: int) -> tuple[float, object]:
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    weight_grid: list[object] = [None, "balanced", {0: 1.0, 1: 1.15, 2: 1.0}]
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None
    for c in c_grid:
        for weight in weight_grid:
            oof = np.zeros(len(y_train), dtype=int)
            for tr, va in inner.split(X_train, y_train):
                model = make_lr(c, weight, seed)
                model.fit(X_train.iloc[tr], y_train[tr])
                oof[va] = model.predict(X_train.iloc[va])
            m = classification_metrics(y_train, oof)
            key = (m["balanced_accuracy"], m["accuracy"], m["min_recall"], -m["cross_error"])
            if best is None or key > best[0]:
                best = (key, c, weight)
    assert best is not None
    return best[1], best[2]


def nested_oof(df: pd.DataFrame, cols: list[str], folds: int, inner_folds: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[cols].copy()
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros((len(df), 3), dtype=float)
    rows = []
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        c, weight = choose_spec(X.iloc[tr].reset_index(drop=True), y[tr], inner_folds, seed + fold)
        model = make_lr(c, weight, seed + fold)
        model.fit(X.iloc[tr], y[tr])
        pred[te] = model.predict(X.iloc[te])
        prob[te] = align_prob(model, X.iloc[te])
        rows.append(
            {
                "fold": fold,
                "c": c,
                "class_weight": json.dumps(weight) if isinstance(weight, dict) else (weight or "none"),
                **{f"test_{k}": v for k, v in classification_metrics(y[te], pred[te]).items()},
            }
        )
    return pred, prob, rows


def weighted_vote(matrix: np.ndarray, weights: list[float]) -> np.ndarray:
    score = np.zeros((matrix.shape[0], 3), dtype=float)
    idx = np.arange(matrix.shape[0])
    for j, weight in enumerate(weights):
        score[idx, matrix[:, j].astype(int)] += float(weight)
    return score.argmax(axis=1).astype(int)


def final_hard_ensemble(df: pd.DataFrame) -> np.ndarray:
    anchors = df[["summary_meta", "summary_meta_struct", "full_meta", "blend"]].to_numpy(dtype=int)
    pred = weighted_vote(anchors, [1.0, 0.75, 1.0, 2.0])
    vote_nmr_share = df["vote_somewhat_helpful"].to_numpy(dtype=float) / df["n_votes"].clip(lower=1).to_numpy(dtype=float)
    mean_under = df["mean_changes_reader_understanding"].to_numpy(dtype=float)
    promote = (
        (pred == 1)
        & (df["blend"].to_numpy(dtype=int) == df["summary_meta_struct"].to_numpy(dtype=int))
        & (df["blend"].to_numpy(dtype=int) != 1)
        & (vote_nmr_share >= 0.5625)
        & (mean_under <= 29.79375)
    )
    pred[promote] = df.loc[promote, "blend"].to_numpy(dtype=int)
    return pred


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "oof_aggregation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    y = df["true_label_3way"].to_numpy(dtype=int)
    predictions = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    summary_rows = []
    fold_rows = []

    runs = {
        "summary_meta": feature_sets["summary"],
        "summary_meta_struct": feature_sets["summary"],
        "full_meta": feature_sets["full_agent_plus_metadata"],
    }
    for name, cols in runs.items():
        pred, prob, folds = nested_oof(df, cols, args.folds, args.inner_folds, args.seed)
        predictions[name] = pred
        for label_id, label in INT_TO_LABEL.items():
            predictions[f"{name}_prob_{label.lower()}"] = prob[:, label_id]
        summary_rows.append({"method": name, "n_features": len(cols), **classification_metrics(y, pred)})
        for row in folds:
            row["method"] = name
            fold_rows.append(row)

    # Rationale/metadata blend proxy: combine the summary-structured and full views.
    blend_prob = (
        predictions[[f"summary_meta_struct_prob_{INT_TO_LABEL[i].lower()}" for i in range(3)]].to_numpy()
        + predictions[[f"full_meta_prob_{INT_TO_LABEL[i].lower()}" for i in range(3)]].to_numpy()
    ) / 2.0
    predictions["blend"] = blend_prob.argmax(axis=1)
    summary_rows.append({"method": "blend", "n_features": -1, **classification_metrics(y, predictions["blend"].to_numpy(int))})

    working = df.merge(predictions[["noteId", "summary_meta", "summary_meta_struct", "full_meta", "blend"]], on="noteId", how="left")
    final_pred = final_hard_ensemble(working)
    predictions["multicom_final"] = final_pred
    summary_rows.append({"method": "multicom_final", "n_features": -1, **classification_metrics(y, final_pred)})

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "folds": args.folds,
                "inner_folds": args.inner_folds,
                "seed": args.seed,
                "feature_sets": {k: len(v) for k, v in feature_sets.items()},
                "promotion_rule": {
                    "vote_nmr_share_min": 0.5625,
                    "mean_changes_reader_understanding_max": 29.79375,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

