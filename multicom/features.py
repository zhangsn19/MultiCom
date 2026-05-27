from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .schema import RAW_LABEL_SCORE


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_entropy(df: pd.DataFrame, cols: list[str], out_col: str) -> None:
    arr = df[cols].fillna(0.0).clip(1e-9, 1.0).to_numpy(dtype=float)
    df[out_col] = -(arr * np.log(arr)).sum(axis=1)


def build_features(run_dir) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    run_dir = run_dir.resolve()
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["raw_label"] = votes["parsed_rating"].fillna(votes.get("helpfulnessLevel", "")).astype(str)
    votes["raw_score"] = votes["raw_label"].map(RAW_LABEL_SCORE)
    votes = votes[votes["raw_score"].isin([0.0, 0.5, 1.0])].copy()

    numeric_cols = [
        "predicted_rating_score",
        "raw_score",
        "agree",
        "disagree",
        "helpfulClear",
        "helpfulGoodSources",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "helpfulUnbiasedLanguage",
        "notHelpfulIncorrect",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulMissingKeyPoints",
        "notHelpfulHardToUnderstand",
        "notHelpfulArgumentativeOrBiased",
        "notHelpfulIrrelevantSources",
        "notHelpfulOpinionSpeculation",
        "notHelpfulNoteNotNeeded",
        "confidence",
        "changes_reader_understanding",
    ]
    for col in numeric_cols:
        votes[col] = safe_num(votes, col)

    votes["is_h"] = (votes["raw_label"] == "HELPFUL").astype(float)
    votes["is_sh"] = (votes["raw_label"] == "SOMEWHAT_HELPFUL").astype(float)
    votes["is_nh"] = (votes["raw_label"] == "NOT_HELPFUL").astype(float)

    grouped = votes.groupby("noteId", as_index=False)
    base = grouped.agg(
        tweetId=("tweetId", "first"),
        currentStatus=("currentStatus", "first"),
        true_label_3way=("true_label_3way", "first"),
        true_label_text=("true_label_text", "first"),
        n_votes=("agent_id", "size"),
        vote_helpful=("is_h", "sum"),
        vote_somewhat_helpful=("is_sh", "sum"),
        vote_not_helpful=("is_nh", "sum"),
        mean_raw_score=("raw_score", "mean"),
        std_raw_score=("raw_score", "std"),
        mean_confidence=("confidence", "mean"),
        std_confidence=("confidence", "std"),
        mean_changes_reader_understanding=("changes_reader_understanding", "mean"),
        agree_rate=("agree", "mean"),
        disagree_rate=("disagree", "mean"),
        helpful_clear_rate=("helpfulClear", "mean"),
        helpful_good_sources_rate=("helpfulGoodSources", "mean"),
        helpful_addresses_claim_rate=("helpfulAddressesClaim", "mean"),
        helpful_important_context_rate=("helpfulImportantContext", "mean"),
        helpful_unbiased_language_rate=("helpfulUnbiasedLanguage", "mean"),
        not_helpful_incorrect_rate=("notHelpfulIncorrect", "mean"),
        not_helpful_sources_missing_or_unreliable_rate=("notHelpfulSourcesMissingOrUnreliable", "mean"),
        not_helpful_missing_key_points_rate=("notHelpfulMissingKeyPoints", "mean"),
        not_helpful_hard_to_understand_rate=("notHelpfulHardToUnderstand", "mean"),
        not_helpful_argumentative_or_biased_rate=("notHelpfulArgumentativeOrBiased", "mean"),
        not_helpful_irrelevant_sources_rate=("notHelpfulIrrelevantSources", "mean"),
        not_helpful_opinion_speculation_rate=("notHelpfulOpinionSpeculation", "mean"),
        not_helpful_note_not_needed_rate=("notHelpfulNoteNotNeeded", "mean"),
    )
    base["true_label_3way"] = pd.to_numeric(base["true_label_3way"], errors="coerce").astype(int)
    base["std_raw_score"] = base["std_raw_score"].fillna(0.0)
    base["std_confidence"] = base["std_confidence"].fillna(0.0)
    for name in ["helpful", "somewhat_helpful", "not_helpful"]:
        base[f"share_{name}"] = base[f"vote_{name}"] / base["n_votes"].clip(lower=1)
    add_entropy(base, ["share_not_helpful", "share_somewhat_helpful", "share_helpful"], "raw_vote_entropy")
    base["helpful_vs_not_margin"] = (base["share_helpful"] - base["share_not_helpful"]).abs()
    base["positive_mass"] = base["share_helpful"] + 0.5 * base["share_somewhat_helpful"]
    base["negative_mass"] = base["share_not_helpful"] + 0.5 * base["share_somewhat_helpful"]
    base["resolved_raw_vote_share"] = base["share_helpful"] + base["share_not_helpful"]
    base["helpful_reason_mean"] = base[
        [
            "helpful_clear_rate",
            "helpful_good_sources_rate",
            "helpful_addresses_claim_rate",
            "helpful_important_context_rate",
            "helpful_unbiased_language_rate",
        ]
    ].mean(axis=1)
    base["not_helpful_reason_mean"] = base[
        [
            "not_helpful_incorrect_rate",
            "not_helpful_sources_missing_or_unreliable_rate",
            "not_helpful_missing_key_points_rate",
            "not_helpful_hard_to_understand_rate",
            "not_helpful_argumentative_or_biased_rate",
            "not_helpful_irrelevant_sources_rate",
            "not_helpful_opinion_speculation_rate",
            "not_helpful_note_not_needed_rate",
        ]
    ].mean(axis=1)

    total_conf = votes.groupby("noteId")["confidence"].sum().replace(0, np.nan)
    for raw_label, name in [
        ("NOT_HELPFUL", "not_helpful"),
        ("SOMEWHAT_HELPFUL", "somewhat_helpful"),
        ("HELPFUL", "helpful"),
    ]:
        weighted = votes[votes["raw_label"] == raw_label].groupby("noteId")["confidence"].sum() / total_conf
        base = base.merge(weighted.rename(f"conf_weighted_share_{name}").reset_index(), on="noteId", how="left")

    label_map = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
    votes["raw_label_id"] = votes["raw_label"].map(label_map)
    label_pivot = votes.pivot_table(index="noteId", columns="agent_id", values="raw_label_id", aggfunc="first")
    label_pivot = label_pivot.add_prefix("agent_raw_label__").reset_index()
    base = base.merge(label_pivot, on="noteId", how="left")
    for col in [c for c in base.columns if c.startswith("agent_raw_label__")]:
        for label_id, label_name in [(0, "nh"), (1, "sh"), (2, "h")]:
            base[f"{col}__is_{label_name}"] = (base[col] == label_id).astype(float)

    for metric in [
        "confidence",
        "changes_reader_understanding",
        "helpfulGoodSources",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulMissingKeyPoints",
        "notHelpfulNoteNotNeeded",
    ]:
        pivot = votes.pivot_table(index="noteId", columns="agent_id", values=metric, aggfunc="first")
        base = base.merge(pivot.add_prefix(f"agent_{metric}__").reset_index(), on="noteId", how="left")

    pilot_path = run_dir / "pilot_notes.csv"
    if pilot_path.exists():
        full = add_note_metadata(base, pd.read_csv(pilot_path, low_memory=False))
    else:
        full = base.copy()

    non_features = {"noteId", "tweetId", "currentStatus", "true_label_3way", "true_label_text"}
    summary_features = [
        c
        for c in base.columns
        if c not in non_features
        and not (c.startswith("agent_raw_label__") and "__is_" not in c)
        and not c.startswith("agent_confidence__")
        and not c.startswith("agent_changes_reader_understanding__")
        and not c.startswith("agent_helpful")
        and not c.startswith("agent_notHelpful")
    ]
    full_features = [
        c
        for c in base.columns
        if c not in non_features and not (c.startswith("agent_raw_label__") and "__is_" not in c)
    ]
    metadata_features = [c for c in full.columns if c not in base.columns and c != "noteId"]
    return full.sort_values("noteId").reset_index(drop=True), {
        "summary": summary_features,
        "full_agent": full_features,
        "full_agent_plus_metadata": full_features + metadata_features,
    }


def add_note_metadata(base: pd.DataFrame, pilot: pd.DataFrame) -> pd.DataFrame:
    pilot = pilot.copy()
    pilot["noteId"] = pilot["noteId"].astype(str)
    meta_cols = [
        "noteId",
        "year",
        "classification",
        "primary_topic",
        "isMediaNote",
        "isCollaborativeNote",
        "misleadingManipulatedMedia",
        "misleadingFactualError",
        "misleadingOutdatedInformation",
        "misleadingMissingImportantContext",
        "misleadingUnverifiedClaimAsFact",
        "misleadingSatire",
        "notMisleadingOther",
        "notMisleadingFactuallyCorrect",
        "notMisleadingOutdatedButNotWhenWritten",
        "notMisleadingClearlySatire",
        "notMisleadingPersonalOpinion",
        "topic_count",
    ]
    meta = pilot[[c for c in meta_cols if c in pilot.columns]].drop_duplicates("noteId").copy()
    categorical = [c for c in ["classification", "primary_topic"] if c in meta.columns]
    if categorical:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        arr = enc.fit_transform(meta[categorical].fillna("UNKNOWN"))
        enc_df = pd.DataFrame(arr, columns=[f"meta_{x}" for x in enc.get_feature_names_out(categorical)])
        meta = pd.concat([meta.drop(columns=categorical).reset_index(drop=True), enc_df], axis=1)
    for col in meta.columns:
        if col != "noteId":
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    return base.merge(meta, on="noteId", how="left")

