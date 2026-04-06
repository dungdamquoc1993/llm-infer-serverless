#!/usr/bin/env python3
"""
Clean noise from JSONL dataset produced by export_dataset.py.

Noise found in assistant turns (page echo messages):
  1. Facebook "replied to an ad" notifications
  2. Facebook auto-greeting lines  ("Hi X! Please let us know…")
  3. CRM auto-label lines          ("Auto-label added: …")
  4. CRM lead-stage lines          ("Lead stage set to …")
  5. Facebook spam notification    ("This message was automatically moved to spam.")
  6. Facebook post/story links     ("X đã trả lời … Xem bài viết(https://…)")
  7. Order-dump spreadsheet lines  (lines with 3+ tab-separated fields)
  8. Duplicate adjacent lines within one turn

Strategy
--------
- Clean each assistant turn line by line.
- If a turn becomes empty after cleaning → skip that turn (not drop the whole sample).
- After skipping turns, merge consecutive same-role turns (keeps sample valid).
- Trim trailing user turns so every sample ends with an assistant turn.
- Drop samples that end up with no assistant turn or no user turn.
- Backup originals to .bak before writing.

Input/Output: dataset/train.jsonl and dataset/val.jsonl  (resolved from repo root)
"""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _DATA_ROOT.parent
DATASET_DIR = REPO_ROOT / "dataset"
TRAIN_FILE = DATASET_DIR / "train.jsonl"
VAL_FILE = DATASET_DIR / "val.jsonl"

# ── Noise line patterns (applied per line after splitting on \n) ───────────────
# Each pattern is tested with re.search on a stripped line.
_NOISE: list[re.Pattern] = [
    # 1. "Long Sana replied to an ad."
    re.compile(r"^.{1,80} replied to an ad\.$"),

    # 2a. EN greeting: "Hi Sana! Please let us know how we can help you."
    re.compile(r"^Hi .{1,60}! Please let us know how we can help you\.$"),

    # 2b. VI greeting: "Hi Sana! Hãy cho Chúng Tôi biết Chúng Tôi có thể giúp được gì."
    re.compile(r"^Hi .{1,60}! Hãy cho Chúng Tôi biết Chúng Tôi có thể giúp được gì\.$"),

    # 2c. Longer EN/VI greeting variant with name embedded
    re.compile(r"^Chào .{1,60} hãy cho chúng tôi biết .{0,60}$", re.IGNORECASE),

    # 3. CRM auto-label
    re.compile(r"^Auto-label added:"),

    # 4. CRM lead stage
    re.compile(r"^Lead stage set to"),

    # 5. Spam notification
    re.compile(r"^This message was automatically moved to spam\.$"),

    # 6a. Vietnamese post-link line  "X đã trả lời về một bài viết. Xem bài viết(https://…)"
    re.compile(r"Xem bài viết\(https?://"),

    # 6b. "X đã trả lời về một bài viết." standalone
    re.compile(r"^.{1,80} đã trả lời về một bài viết\.$"),

    # 7. Order-dump: lines with 3+ tabs  (spreadsheet copy-paste)
    re.compile(r"^[^\t]*\t[^\t]*\t[^\t]*\t"),
]


def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE)


def clean_content(raw: str) -> str:
    """
    Remove noise lines and deduplicate adjacent identical lines.
    Returns the cleaned string (may be empty if everything was noise).
    """
    # Normalise line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines
        if _is_noise(stripped):
            continue
        kept.append(stripped)

    # Deduplicate adjacent identical lines
    deduped: list[str] = []
    for line in kept:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return "\n".join(deduped)


def clean_sample(sample: dict) -> dict | None:
    """
    Clean one sample. Returns None if the sample should be dropped.

    Processing order:
      1. Clean every assistant turn; skip (not drop) turns that become empty.
      2. Merge consecutive same-role turns produced by skipped turns.
      3. Trim trailing user turns.
      4. Validate: must have ≥1 user turn and ≥1 assistant turn.
    """
    messages: list[dict] = sample.get("messages", [])
    cleaned: list[dict] = []

    for msg in messages:
        role = msg["role"]
        if role == "assistant":
            new_content = clean_content(msg["content"])
            if not new_content:
                # Drop this turn; merging will fix consecutive user turns
                continue
            cleaned.append({**msg, "content": new_content})
        else:
            # system / user — pass through unchanged
            cleaned.append(msg)

    # Merge consecutive same-role turns (can happen after dropping empty assistant turns)
    merged: list[dict] = []
    for msg in cleaned:
        if (
            merged
            and merged[-1]["role"] == msg["role"]
            and msg["role"] not in ("system",)
        ):
            merged[-1] = {
                **merged[-1],
                "content": merged[-1]["content"] + "\n" + msg["content"],
            }
        else:
            merged.append(msg)

    # Trim trailing user turns (sample must end with assistant)
    while merged and merged[-1]["role"] == "user":
        merged.pop()

    # Validate
    roles = [m["role"] for m in merged]
    if roles.count("assistant") < 1 or roles.count("user") < 1:
        return None

    return {**sample, "messages": merged}


def process_file(path: Path, stats: Counter) -> list[dict]:
    samples_in: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples_in.append(json.loads(line))

    samples_out: list[dict] = []
    for s in samples_in:
        result = clean_sample(s)
        if result is None:
            stats["dropped"] += 1
            continue
        original_str = json.dumps(s, ensure_ascii=False)
        cleaned_str = json.dumps(result, ensure_ascii=False)
        if cleaned_str != original_str:
            stats["modified"] += 1
        else:
            stats["unchanged"] += 1
        samples_out.append(result)

    return samples_out


def write_jsonl(path: Path, samples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue

        bak = path.with_suffix(".jsonl.bak")
        shutil.copy2(path, bak)
        print(f"Backed up  → {bak.name}")

        stats: Counter = Counter()
        cleaned = process_file(path, stats)
        write_jsonl(path, cleaned)

        total_in = stats["dropped"] + stats["modified"] + stats["unchanged"]
        print(f"{path.name}")
        print(f"  Input    : {total_in} samples")
        print(f"  Unchanged: {stats['unchanged']}")
        print(f"  Modified : {stats['modified']}")
        print(f"  Dropped  : {stats['dropped']}")
        print(f"  Output   : {len(cleaned)} samples")
        print()

    print("Done ✓")


if __name__ == "__main__":
    main()
