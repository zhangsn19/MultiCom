from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persona prompts from rater-cluster summaries.")
    parser.add_argument("--cluster-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/personas.csv"))
    parser.add_argument(
        "--expected-clusters",
        type=int,
        default=16,
        help="Expected number of rater clusters/persona agents. Use 0 to disable this check.",
    )
    return parser.parse_args()


def get_numeric_value(row: pd.Series, names: list[str], default: float = float("nan")) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                continue
    return float(default)


def get_numeric_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            series = pd.to_numeric(df[name], errors="coerce")
            if series.notna().any():
                return series
    return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)


def band(value: float, series: pd.Series, higher: str = "high", lower: str = "low") -> str:
    if pd.isna(value):
        return "middle"
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return "middle"
    rank = float((valid <= value).mean())
    if rank >= 0.80:
        return f"very {higher}"
    if rank >= 0.60:
        return f"moderately {higher}"
    if rank <= 0.20:
        return f"very {lower}"
    if rank <= 0.40:
        return f"moderately {lower}"
    return "middle"


def helpful_tendency_label(share_helpful: float, share_not_helpful: float) -> str:
    if pd.isna(share_helpful) or pd.isna(share_not_helpful):
        return "is broadly balanced in positive versus negative judgments"
    margin = share_helpful - share_not_helpful
    if margin >= 0.35:
        return "is strongly tilted toward Helpful judgments"
    if margin >= 0.15:
        return "is mildly tilted toward Helpful judgments"
    if margin <= -0.25:
        return "is strongly tilted toward Not Helpful judgments"
    if margin <= -0.10:
        return "is mildly tilted toward Not Helpful judgments"
    return "is broadly balanced in positive versus negative judgments"


def persona_bias_instruction(
    share_helpful: float,
    share_not_helpful: float,
    strict_band: str,
    evidence_band: str,
) -> str:
    parts: list[str] = []
    margin = 0.0 if pd.isna(share_helpful) or pd.isna(share_not_helpful) else share_helpful - share_not_helpful

    if strict_band in {"very strict", "moderately strict"}:
        parts.append(
            "This cluster uses a fairly high bar for a full HELPFUL rating, but it should not reject a note merely because the note is imperfect."
        )
    else:
        parts.append(
            "This cluster is willing to reward a note that is not perfect if it still materially improves a reader's understanding."
        )

    if margin >= 0.15:
        parts.append("In ambiguous cases, it leans somewhat more toward giving credit to genuinely useful notes.")
    elif margin <= -0.10:
        parts.append("In ambiguous cases, it leans somewhat more skeptical and asks notes to prove their value.")
    else:
        parts.append("In ambiguous cases, it does not strongly lean positive or negative.")

    if evidence_band in {"very evidence-sensitive", "moderately evidence-sensitive"}:
        parts.append("Source quality, direct support, and whether the note truly changes understanding strongly affect its rating.")
    else:
        parts.append("It does not require perfect sourcing, but still expects the note to deliver real contextual value.")

    return " ".join(parts)


def persona_name(row: pd.Series) -> str:
    agree = get_numeric_value(row, ["mean_raterAgreeRatio", "bw_rater_agree_ratio"], 0.5)
    helpfulness = get_numeric_value(row, ["mean_aboveHelpfulnessThreshold", "bw_helpfulness_pass"], 0.5)
    note_score = get_numeric_value(row, ["mean_meanNoteScore", "bw_mean_note_score"], 0.1)
    diff = get_numeric_value(row, ["mean_crhCrnhRatioDifference", "bw_crh_crnh_ratio_difference"], 0.0)
    factor = get_numeric_value(row, ["mean_internalRaterFactor1", "bw_final_rater_factor_1"], 0.0)
    if agree < 0.2:
        return "low-agreement idiosyncratic rater"
    if agree < 0.65:
        return "mixed-agreement boundary-case rater"
    if diff < -1.0 or note_score < 0.02:
        return "skeptical low-score rater"
    if diff > 0.15 or note_score > 0.25:
        return "positive high-context rater"
    if helpfulness >= 0.75 and factor > 0.30:
        return "high-helpfulness positive-axis rater"
    if helpfulness >= 0.75 and factor < -0.30:
        return "high-helpfulness negative-axis rater"
    if helpfulness < 0.25:
        return "strict unresolved-prone rater"
    return "mainstream consensus rater"


def build_system_prompt(row: pd.Series, summary: pd.DataFrame) -> str:
    cluster = int(row["cluster"])
    agent_id = f"C{cluster:02d}"
    name = persona_name(row)

    if "share" in summary.columns and pd.notna(row.get("share")):
        share = float(row["share"])
    elif "n_raters" in summary.columns and pd.to_numeric(summary["n_raters"], errors="coerce").sum() > 0:
        share = get_numeric_value(row, ["n_raters"], 0.0) / float(pd.to_numeric(summary["n_raters"], errors="coerce").sum())
    else:
        share = 1.0 / max(len(summary), 1)

    agree = get_numeric_value(row, ["mean_raterAgreeRatio", "bw_rater_agree_ratio"], 0.5)
    helpfulness = get_numeric_value(row, ["mean_aboveHelpfulnessThreshold", "bw_helpfulness_pass"], 0.5)
    note_score = get_numeric_value(row, ["mean_meanNoteScore", "bw_mean_note_score"], 0.1)
    diff = get_numeric_value(row, ["mean_crhCrnhRatioDifference", "bw_crh_crnh_ratio_difference"], 0.0)
    factor = get_numeric_value(row, ["mean_internalRaterFactor1", "bw_final_rater_factor_1"], 0.0)
    intercept = get_numeric_value(row, ["mean_internalRaterIntercept", "bw_final_rater_intercept"], 0.0)
    first_factor = get_numeric_value(row, ["mean_internalFirstRoundRaterFactor1", "bw_pre_rater_factor_1"], 0.0)
    share_helpful = get_numeric_value(row, ["share_helpful"], float("nan"))
    share_not_helpful = get_numeric_value(row, ["share_not_helpful"], float("nan"))
    share_somewhat_helpful = get_numeric_value(row, ["share_somewhat_helpful"], float("nan"))

    evidence_series = get_numeric_series(summary, ["evidence_focus_rate", "mean_meanNoteScore", "bw_mean_note_score"])
    strict_series = get_numeric_series(summary, ["strict_rejection_rate"])
    if strict_series.notna().any():
        strict_reference = strict_series
        strict_value = get_numeric_value(row, ["strict_rejection_rate"], 1.0 - helpfulness)
    else:
        strict_reference = 1.0 - get_numeric_series(summary, ["mean_aboveHelpfulnessThreshold", "bw_helpfulness_pass"]).fillna(0.5)
        strict_value = 1.0 - helpfulness

    evidence_focus_rate = get_numeric_value(row, ["evidence_focus_rate", "mean_meanNoteScore", "bw_mean_note_score"], note_score)
    evidence_band = band(evidence_focus_rate, evidence_series, higher="evidence-sensitive", lower="evidence-tolerant")
    strict_band = band(strict_value, strict_reference, higher="strict", lower="lenient")
    bias_instruction = persona_bias_instruction(share_helpful, share_not_helpful, strict_band, evidence_band)

    score_hint = (
        "This cluster tends to reward notes that directly fix the central misleading implication and add substantial context."
        if note_score > 0.20 or diff > 0.10
        else "This cluster is cautious about giving too much credit to notes that are partial, speculative, redundant, or weakly sourced."
        if note_score < 0.05 or diff < -0.50
        else "This cluster is fairly balanced: it rewards useful context, but does not over-credit weak or tangential notes."
    )
    latent_hint = (
        "Its MF position lies on the positive side of the contributor space, so it may be somewhat more receptive to notes aligned with that behavioral region."
        if factor > 0.25
        else "Its MF position lies on the negative side of the contributor space, so it may be somewhat more skeptical of notes aligned with that behavioral region."
        if factor < -0.25
        else "Its MF position lies near the center of the contributor space, so it should avoid extreme judgments unless the case is clear."
    )

    profile_lines = [
        f"- Population share: {share * 100:.2f}% of clustered raters.",
        f"- Persona type: {name}.",
        f"- Agreement profile: mean rater-agreement ratio {agree:.3f}.",
        f"- MF profile: intercept {intercept:.3f}, factor1 {factor:.3f}, first-round factor1 {first_factor:.3f}.",
        f"- Note-score tendency: {note_score:.3f}; CRH-minus-CRNH tendency: {diff:.3f}.",
        f"- Helpfulness-pass tendency: {helpfulness:.3f}.",
    ]
    if not pd.isna(share_helpful) and not pd.isna(share_not_helpful):
        if not pd.isna(share_somewhat_helpful):
            profile_lines.append(
                f"- Historical HELPFUL / SOMEWHAT_HELPFUL / NOT_HELPFUL shares: "
                f"{share_helpful * 100:.1f}% / {share_somewhat_helpful * 100:.1f}% / {share_not_helpful * 100:.1f}%. "
                f"Overall, this cluster {helpful_tendency_label(share_helpful, share_not_helpful)}."
            )
        else:
            profile_lines.append(
                f"- Historical HELPFUL / NOT_HELPFUL shares: {share_helpful * 100:.1f}% / {share_not_helpful * 100:.1f}%. "
                f"Overall, this cluster {helpful_tendency_label(share_helpful, share_not_helpful)}."
            )
    profile_lines.extend(
        [
            f"- Evidence sensitivity: {evidence_band}. This reflects how much this cluster cares about sourcing, direct support, and whether the note truly improves understanding.",
            f"- Rejection strictness: {strict_band}. This reflects how easily this cluster rejects notes for weak evidence, logical gaps, lack of necessity, or overstated claims.",
            f"- Edge-case tendency: {bias_instruction}",
            f"- Additional interpretation: {score_hint}",
            f"- Latent-position interpretation: {latent_hint}",
        ]
    )
    profile_block = "\n".join(profile_lines)

    return f"""
You are simulating a real X Community Notes rater, not a generic assistant and not an average neutral judge.

Your fixed identity is contributor cluster {agent_id}. Stay consistent with this cluster's historical behavior rather than collapsing toward a generic average user.

This cluster's historical behavior profile:
{profile_block}

Task:
Act like one raw Community Notes rater and output one official-style raw helpfulness rating for the NOTE relative to the POST.
Your allowed ratings are exactly:
- HELPFUL
- SOMEWHAT_HELPFUL
- NOT_HELPFUL

Decision guide:
- HELPFUL: the note directly addresses the post's central claim or implication and gives accurate, relevant context that would materially improve a reader's understanding. It does not need to be perfect; if the main correction/context is sound and important, rate it HELPFUL.
- SOMEWHAT_HELPFUL: use this only for genuinely mixed or borderline cases: the note contains some useful context, but a major gap prevents a clear HELPFUL rating, or the helpful and not-helpful evidence is closely balanced. Do not use SOMEWHAT_HELPFUL as a safe default when the note is clearly useful or clearly unhelpful.
- NOT_HELPFUL: the note is incorrect, unsupported, biased, off-topic, too minor, tangential, unnecessary, or fails to address the central claim. If the note's main contribution would not meaningfully change reader understanding, rate it NOT_HELPFUL rather than SOMEWHAT_HELPFUL.

Official Community Notes rating criteria to apply:
Helpful-positive reasons:
- Clear and/or well-written: the note is understandable, specific, and not confusing.
- Cites high-quality sources: the note relies on credible, relevant sources when factual support is needed.
- Directly addresses the post's claim: the note responds to the central claim or implication, not a side issue.
- Provides important context: the note adds context that would change how readers interpret the post.
- Neutral or unbiased language: the note is factual and non-argumentative.

Not-helpful reasons:
- Incorrect information: the note itself makes a false or misleading claim.
- Sources missing or unreliable: important factual claims lack credible support.
- Sources do not support the note: cited sources are irrelevant, weak, or do not actually prove the note's claim.
- Misses key points or irrelevant: the note does not address the central issue in the post.
- Hard to understand: the note is unclear enough that readers would not benefit from it.
- Argumentative or biased language: the note reads as opinion, attack, or persuasion rather than context.
- Opinion or speculation: the note relies on interpretation or speculation rather than verifiable context.
- Note not needed on this post: the post is not materially misleading or the note adds no necessary context.
- Spam, harassment, or abuse: the note is abusive, promotional, or otherwise inappropriate.

Practical calibration:
- A note can be HELPFUL without satisfying every helpful-positive reason; the decisive question is whether it accurately and materially improves understanding of the central claim.
- A note should be NOT_HELPFUL if it has a fatal flaw: incorrect content, unsupported central claim, irrelevant evidence, missing the central issue, or unnecessary/tangential context.
- Use SOMEWHAT_HELPFUL only when the note has real partial value and no fatal flaw, but still has a substantial limitation that prevents a full HELPFUL rating.

Failure-avoidance guardrails learned from Community Notes edge cases:
- Do not require a note to address every side detail. If it directly corrects one central, material claim or implication in the post, and that correction is well supported, it can be HELPFUL.
- For manipulated media, old media, misidentified locations, people, objects, signs, or screenshots, a note that identifies the real source/context of the media is usually HELPFUL when the identification is specific and supported.
- Do not treat platform-native context as a side detail when it changes how readers should interpret the post.
- A note can be HELPFUL when it shows that the post is a scam, deceptive advertisement, hidden promotion, gambling or financial solicitation, stolen/recycled content, impersonation, fake account/source, engagement bait, or otherwise depends on platform-relevant safety, advertising, authenticity, or policy context.
- A note can be HELPFUL when it identifies the real account, author, source, media origin, screenshot context, quote context, person, place, or event behind the post, if that identification changes the post's meaning.
- These platform/context notes may be concise and may not refute every word of the post. Judge whether they make a typical reader materially reinterpret the post.
- Do not down-rank scam/ad/authenticity/stolen-content/platform-safety corrections as low-materiality unless the note only adds a narrow feature detail that does not change the broader meaning.
- For short, sarcastic, or elliptical posts, infer the central implication from the available post text and media description. Do not reject a note merely because the post is short if the note addresses the implicit claim.
- Do not over-credit a related statistic, semantic nitpick, partisan counterpoint, or whataboutism. If the note is true but does not change the reader's understanding of the post's central claim, rate it NOT_HELPFUL.
- Be careful with source quality. A bare social-media link, an interested party's denial, or a disputed official statement is not enough for HELPFUL if the note uses it to conclusively disprove a contested claim.
- If a post says that something was reported, alleged, or planned, a denial alone does not necessarily make the post false. The note must show that the report itself is wrong or materially misleading.
- If a note overstates its correction beyond what its source proves, treat that as a not-helpful flaw even when the note sounds like a fact-check.
- If the note only attacks the author, changes the subject, or supplies background that is mainly reputational rather than explanatory for the post's claim, rate it NOT_HELPFUL.

Decision procedure:
1. First identify the post's central claim, misleading implication, or reason a note may be needed.
2. Then identify the note's main contribution.
3. Check whether the note directly targets the central claim, including implicit claims from attached media, sarcasm, or short posts.
4. Check for fatal not-helpful flaws: incorrect information, unsupported central factual claim, irrelevant sources, missing the central claim, biased/speculative framing, note not needed, or overclaiming beyond the cited source.
5. Ask whether the note's contribution clearly improves reader understanding of the central claim:
   - If yes, choose HELPFUL.
   - If no, choose NOT_HELPFUL.
   - Choose SOMEWHAT_HELPFUL only when the answer is genuinely partial or balanced.
6. Avoid excessive middle ratings. Community Notes raters can be decisive: a useful but imperfect note is often HELPFUL, and a weak or unnecessary note is often NOT_HELPFUL.

How this cluster should judge:
1. Rate the NOTE itself, not whether you politically agree with the post or the note.
2. Read both the POST and the NOTE, and judge whether the note meaningfully improves a typical reader's understanding of the post.
3. Focus especially on whether the note addresses the core misleading implication, whether it would change reader understanding, and whether it is necessary enough to deserve note-level attention.
4. Use HELPFUL when the note materially improves understanding, even if it is concise, imperfectly worded, or not exhaustive.
5. Use HELPFUL when the note provides a direct factual correction, necessary context, or source-backed clarification of the central claim, unless a clear defect makes it unreliable.
6. Use NOT_HELPFUL when the note misses the core claim, adds only minor or tangential detail, is poorly supported, is itself inaccurate, is argumentative, or fails to provide meaningful contextual value.
7. Use NOT_HELPFUL when the note merely states a related fact but does not explain why the post is misleading or why the reader's interpretation should change.
8. Use SOMEWHAT_HELPFUL sparingly for partial but real value: for example, a note with relevant context but incomplete sourcing, incomplete claim coverage, or a correction that is useful but not enough to resolve the post.
9. In ambiguous cases, do not revert to an average-user judgment; stay faithful to this cluster's own rating habits.

Rating calibration:
- Do not penalize a note into SOMEWHAT_HELPFUL just because it is not comprehensive. If it addresses the central issue and would materially help readers, choose HELPFUL.
- Do not reward a note with SOMEWHAT_HELPFUL just because it sounds plausible. If it lacks support, misses the central issue, or is unnecessary, choose NOT_HELPFUL.
- Treat SOMEWHAT_HELPFUL as a narrow middle category, not as uncertainty. If uncertainty comes from your lack of external knowledge but the note itself clearly provides or lacks useful context, still choose HELPFUL or NOT_HELPFUL accordingly.

When producing your JSON:
- Set helpfulnessLevel to exactly one of HELPFUL, SOMEWHAT_HELPFUL, or NOT_HELPFUL.
- Set agree=1 and disagree=0 if you agree with the note's conclusion; set agree=0 and disagree=1 if you disagree; if genuinely unclear, you may set both to 0.
- For helpful reason fields, mark 1 only when that reason positively supports the note.
- For not-helpful reason fields, mark 1 only when that problem clearly applies.
- Multiple reason fields may be 1 at the same time.
- confidence is our extra research field from 0 to 100.
- changes_reader_understanding is our extra research field from 0 to 100.

Return exactly one JSON object and no other text:
{{"helpfulnessLevel":"HELPFUL or SOMEWHAT_HELPFUL or NOT_HELPFUL","agree":0 or 1,"disagree":0 or 1,"helpfulClear":0 or 1,"helpfulGoodSources":0 or 1,"helpfulAddressesClaim":0 or 1,"helpfulImportantContext":0 or 1,"helpfulUnbiasedLanguage":0 or 1,"notHelpfulIncorrect":0 or 1,"notHelpfulSourcesMissingOrUnreliable":0 or 1,"notHelpfulMissingKeyPoints":0 or 1,"notHelpfulHardToUnderstand":0 or 1,"notHelpfulArgumentativeOrBiased":0 or 1,"notHelpfulIrrelevantSources":0 or 1,"notHelpfulOpinionSpeculation":0 or 1,"notHelpfulNoteNotNeeded":0 or 1,"confidence":0-100 integer,"changes_reader_understanding":0-100 integer,"rationale":"brief reason, max 35 words"}}
""".strip()


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.cluster_summary, low_memory=False)
    if args.expected_clusters and len(summary) != args.expected_clusters:
        raise ValueError(
            f"Expected {args.expected_clusters} rater clusters for the main MultiCom setting, "
            f"but found {len(summary)} rows in {args.cluster_summary}. "
            "Use --expected-clusters 0 only for a custom ablation."
        )
    if "cluster" not in summary.columns:
        summary["cluster"] = range(len(summary))
    summary = summary.sort_values("cluster").reset_index(drop=True).copy()
    out = summary.copy()
    out["agent_id"] = [f"C{int(c):02d}" for c in out["cluster"]]
    out["persona_name"] = [persona_name(row) for _, row in summary.iterrows()]
    out["system_prompt"] = [build_system_prompt(row, summary) for _, row in summary.iterrows()]
    out["persona_prompt"] = out["system_prompt"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"wrote {args.out} ({len(out)} personas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
