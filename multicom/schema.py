from __future__ import annotations

LABEL_TO_INT = {"NOT_HELPFUL": 0, "NEEDS_MORE_RATINGS": 1, "HELPFUL": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

STATUS_TO_LABEL = {
    "CURRENTLY_RATED_NOT_HELPFUL": 0,
    "NEEDS_MORE_RATINGS": 1,
    "CURRENTLY_RATED_HELPFUL": 2,
}

RATING_TO_ID = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
RATING_NAME = {0: "nh", 1: "nmr", 2: "h"}
RAW_LABEL_SCORE = {"NOT_HELPFUL": 0.0, "SOMEWHAT_HELPFUL": 0.5, "HELPFUL": 1.0}

OFFICIAL_HELPFUL_REASON_KEYS = [
    "helpfulClear",
    "helpfulGoodSources",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
]

OFFICIAL_NOT_HELPFUL_REASON_KEYS = [
    "notHelpfulIncorrect",
    "notHelpfulSourcesMissingOrUnreliable",
    "notHelpfulMissingKeyPoints",
    "notHelpfulHardToUnderstand",
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulIrrelevantSources",
    "notHelpfulOpinionSpeculation",
    "notHelpfulNoteNotNeeded",
]

REQUIRED_AGENT_OUTPUT_KEYS = [
    "helpfulnessLevel",
    "agree",
    "disagree",
    *OFFICIAL_HELPFUL_REASON_KEYS,
    *OFFICIAL_NOT_HELPFUL_REASON_KEYS,
    "confidence",
    "changes_reader_understanding",
    "rationale",
]

