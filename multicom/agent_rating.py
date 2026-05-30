from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from .schema import (
    OFFICIAL_HELPFUL_REASON_KEYS,
    OFFICIAL_NOT_HELPFUL_REASON_KEYS,
    RATING_TO_ID,
    REQUIRED_AGENT_OUTPUT_KEYS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persona-guided multi-agent note ratings.")
    parser.add_argument("--notes", type=Path, required=True, help="Evaluation CSV with noteId, post_text, note_text, and true labels.")
    parser.add_argument("--personas", type=Path, required=True, help="CSV from multicom.personas.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/agent_votes.csv"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.1"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=420)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional note limit for smoke tests.")
    return parser.parse_args()


def extract_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    raise ValueError(f"No JSON object found in model output: {text[:300]}")


def normalize_rating(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"HELPFUL", "CURRENTLY_RATED_HELPFUL"}:
        return "HELPFUL"
    if text in {"SOMEWHAT_HELPFUL", "PARTLY_HELPFUL", "PARTIALLY_HELPFUL"}:
        return "SOMEWHAT_HELPFUL"
    if text in {"NOT_HELPFUL", "NOTHELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"}:
        return "NOT_HELPFUL"
    return "UNKNOWN"


def coerce_binary(value: object) -> int:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return 0


def coerce_score(value: object) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except Exception:
        return 0.0


def build_prompt(note: pd.Series, persona: pd.Series) -> list[dict[str, str]]:
    system = str(persona.get("system_prompt", persona.get("persona_prompt", ""))).strip()
    if not system:
        system = (
            "You are a simulated Community Notes rater. "
            "Return only one JSON object with the required fields. "
            "Do not include markdown fences or extra prose."
        )
    created = str(note.get("createdAtUTC", "")).strip()
    topic = str(note.get("primary_topic", "")).strip()
    user = f"""
Post creation context: {created}
Author-selected note topic: {topic}

POST:
{note.get("post_text", note.get("tweet_text", ""))}

COMMUNITY NOTE:
{note.get("note_text", note.get("summary", ""))}

Evaluate only the note's usefulness for the post. Do not infer the official status from metadata.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running agent ratings.")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        req = Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def normalize_output(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["helpfulnessLevel"] = normalize_rating(payload.get("helpfulnessLevel"))
    out["parsed_rating"] = out["helpfulnessLevel"]
    out["predicted_rating_score"] = RATING_TO_ID.get(out["parsed_rating"], -1)
    for key in ["agree", "disagree", *OFFICIAL_HELPFUL_REASON_KEYS, *OFFICIAL_NOT_HELPFUL_REASON_KEYS]:
        out[key] = coerce_binary(payload.get(key))
    out["confidence"] = coerce_score(payload.get("confidence"))
    out["changes_reader_understanding"] = coerce_score(payload.get("changes_reader_understanding"))
    out["rationale"] = str(payload.get("rationale", ""))[:2000]
    return out


def rate_one(note: pd.Series, persona: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    messages = build_prompt(note, persona)
    raw = call_chat_completion(messages, args.model, args.base_url, args.temperature, args.max_tokens, args.max_retries)
    payload = extract_json_blob(raw)
    out = normalize_output(payload)
    out.update(
        {
            "noteId": str(note["noteId"]),
            "tweetId": str(note.get("tweetId", "")),
            "currentStatus": str(note.get("currentStatus", "")),
            "true_label_3way": note.get("true_label_3way", ""),
            "true_label_text": note.get("true_label_text", ""),
            "agent_id": str(persona["agent_id"]),
            "cluster": persona.get("cluster", ""),
            "persona_name": persona.get("persona_name", ""),
        }
    )
    return out


def main() -> int:
    args = parse_args()
    notes = pd.read_csv(args.notes, dtype={"noteId": str}, low_memory=False)
    personas = pd.read_csv(args.personas, low_memory=False)
    if args.limit:
        notes = notes.head(args.limit).copy()
    tasks = [(note, persona) for _, note in notes.iterrows() for _, persona in personas.iterrows()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "noteId",
        "tweetId",
        "currentStatus",
        "true_label_3way",
        "true_label_text",
        "agent_id",
        "cluster",
        "persona_name",
        "helpfulnessLevel",
        "parsed_rating",
        "predicted_rating_score",
        "agree",
        "disagree",
        *OFFICIAL_HELPFUL_REASON_KEYS,
        *OFFICIAL_NOT_HELPFUL_REASON_KEYS,
        "confidence",
        "changes_reader_understanding",
        "rationale",
    ]
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(rate_one, note, persona, args) for note, persona in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                writer.writerow(future.result())
                if i % 100 == 0 or i == len(futures):
                    print(f"completed {i}/{len(futures)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
