from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from .schema import STATUS_TO_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Community Notes files for MultiCom.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract downloaded Community Notes zip files.")
    p_extract.add_argument("--raw-dir", type=Path, required=True)
    p_extract.add_argument("--out-dir", type=Path, required=True)

    p_eval = sub.add_parser("make-eval", help="Build a model input CSV from notes/status/post-text files.")
    p_eval.add_argument("--notes-tsv", type=Path, required=True)
    p_eval.add_argument("--status-tsv", type=Path, required=True)
    p_eval.add_argument("--posts-csv", type=Path, required=True, help="User-provided noteId/tweetId/post_text table.")
    p_eval.add_argument("--out", type=Path, default=Path("data/eval_notes.csv"))
    p_eval.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def extract(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in sorted(raw_dir.rglob("*.zip")):
        target_dir = out_dir / zip_path.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = target_dir / Path(member.filename).name
                with archive.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                print(f"extracted {target}")


def first_existing(columns: list[str], candidates: list[str]) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"none of {candidates} found in columns")


def make_eval(notes_tsv: Path, status_tsv: Path, posts_csv: Path, out: Path, limit: int) -> None:
    notes = pd.read_csv(notes_tsv, sep="\t", dtype=str, low_memory=False)
    status = pd.read_csv(status_tsv, sep="\t", dtype=str, low_memory=False)
    posts = pd.read_csv(posts_csv, dtype=str, low_memory=False)

    note_id_col = first_existing(list(notes.columns), ["noteId", "note_id"])
    note_text_col = first_existing(list(notes.columns), ["summary", "noteText", "note_text"])
    status_note_id_col = first_existing(list(status.columns), ["noteId", "note_id"])
    status_col = first_existing(
        list(status.columns),
        ["currentStatus", "currentLabel", "currentStatusWithLockedStatus", "mostRecentStatus"],
    )
    posts_note_id_col = "noteId" if "noteId" in posts.columns else note_id_col

    df = notes[[note_id_col, note_text_col] + [c for c in ["tweetId", "classification"] if c in notes.columns]].copy()
    df = df.rename(columns={note_id_col: "noteId", note_text_col: "note_text"})
    status_keep = status[[status_note_id_col, status_col]].rename(
        columns={status_note_id_col: "noteId", status_col: "currentStatus"}
    )
    df = df.merge(status_keep.drop_duplicates("noteId"), on="noteId", how="left")
    posts = posts.rename(columns={posts_note_id_col: "noteId"})
    df = df.merge(posts[[c for c in ["noteId", "tweetId", "post_text"] if c in posts.columns]], on="noteId", how="left")
    df["true_label_3way"] = df["currentStatus"].map(STATUS_TO_LABEL)
    df["true_label_text"] = df["true_label_3way"].map({0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"})
    df = df[df["true_label_3way"].notna() & df["note_text"].notna() & df["post_text"].notna()].copy()
    if limit:
        df = df.head(limit).copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {out} ({len(df)} rows)")


def main() -> int:
    args = parse_args()
    if args.cmd == "extract":
        extract(args.raw_dir, args.out_dir)
    elif args.cmd == "make-eval":
        make_eval(args.notes_tsv, args.status_tsv, args.posts_csv, args.out, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

