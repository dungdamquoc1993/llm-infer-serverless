#!/usr/bin/env python3
"""
Export Facebook Messenger conversations from PostgreSQL to JSONL (ChatML format).

Target page : Giày Đá Bóng - Yêu Bóng Đá Shop (fan_page_id: 693218564214489)
Output      : <repo>/dataset/train.jsonl  +  val.jsonl (chạy từ bất kỳ cwd)
"""

import json
import os
import random
from pathlib import Path
from collections import Counter

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

_DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _DATA_ROOT.parent

load_dotenv(REPO_ROOT / ".env")
load_dotenv(_DATA_ROOT / ".env", override=True)

# ── Config ────────────────────────────────────────────────────────────────────
FAN_PAGE_ID = "693218564214489"
FAN_PAGE_NAME = "Giày Đá Bóng - Yêu Bóng Đá Shop"

OUTPUT_DIR = REPO_ROOT / "dataset"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"

VAL_RATIO = 0.05  # 5 % for validation
MAX_MSGS_PER_SAMPLE = 30  # slide window if conversation longer than this
SLIDE_STRIDE = 15  # step between windows
MIN_ASSISTANT_TURNS = 1  # drop sample if page hasn't replied at least once
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    f'Bạn là nhân viên tư vấn bán hàng của Fanpage "{FAN_PAGE_NAME}" '
    "trên Facebook Messenger.\n\n"
)
# ─────────────────────────────────────────────────────────────────────────────


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5434)),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )


def fetch_conversation_ids(conn: psycopg2.extensions.connection) -> list[str]:
    """Return IDs of conversations that have at least one page reply."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT fcm.id
            FROM facebook_conversation_messages fcm
            JOIN messages m ON m.conversation_id = fcm.id
            WHERE fcm.fan_page_id = %s
              AND fcm.deleted_at IS NULL
              AND m.deleted_at IS NULL
              AND m.is_echo = TRUE          -- page has replied at least once
              AND m.text IS NOT NULL
              AND m.text <> ''
            ORDER BY fcm.id
            """,
            (FAN_PAGE_ID,),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_messages(conn: psycopg2.extensions.connection, conv_id: str) -> list[dict]:
    """Return text messages for a conversation ordered by time."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT is_echo, text, metadata, facebook_timestamp
            FROM messages
            WHERE conversation_id = %s
              AND deleted_at IS NULL
              AND text IS NOT NULL
              AND text <> ''
            ORDER BY facebook_timestamp ASC
            """,
            (conv_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def _is_ai(msg: dict) -> bool:
    """True if this message was sent by the AI agent (not a human page admin)."""
    meta = msg.get("metadata")
    if not meta:
        return False
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return False
    return isinstance(meta, dict) and meta.get("sent_by") == "ai_agent"


def build_turns(messages: list[dict]) -> list[dict]:
    """
    Convert raw DB rows to ChatML turns.
    - Skip AI-generated messages.
    - Merge consecutive messages from the same role.
    - Returns [{"role": "user"|"assistant", "content": str}, ...]
    """
    turns: list[dict] = []
    for msg in messages:
        if _is_ai(msg):
            continue
        role = "assistant" if msg["is_echo"] else "user"
        content = msg["text"].strip()
        if not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content  # merge consecutive
        else:
            turns.append({"role": role, "content": content})
    return turns


def turns_to_samples(turns: list[dict]) -> list[dict]:
    """
    Convert a turn list to one or more ChatML training samples.
    - Each sample must start with a user turn and end with an assistant turn.
    - Long conversations are split with a sliding window.
    """
    if not turns:
        return []

    # Drop leading assistant turns (page sent first — unusual)
    while turns and turns[0]["role"] == "assistant":
        turns = turns[1:]

    if len(turns) < 2:
        return []

    def make_sample(segment: list[dict]) -> dict | None:
        # Must end with assistant
        end = len(segment)
        while end > 0 and segment[end - 1]["role"] == "user":
            end -= 1
        segment = segment[:end]
        if len(segment) < 2:
            return None
        n_assistant = sum(1 for t in segment if t["role"] == "assistant")
        if n_assistant < MIN_ASSISTANT_TURNS:
            return None
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + segment}

    samples: list[dict] = []

    if len(turns) <= MAX_MSGS_PER_SAMPLE:
        s = make_sample(turns)
        if s:
            samples.append(s)
    else:
        pos = 0
        while pos < len(turns):
            end = min(pos + MAX_MSGS_PER_SAMPLE, len(turns))
            segment = list(turns[pos:end])
            # Drop leading assistant in window
            while segment and segment[0]["role"] == "assistant":
                segment = segment[1:]
            s = make_sample(segment)
            if s:
                samples.append(s)
            if end >= len(turns):
                break
            pos += SLIDE_STRIDE

    return samples


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to database…")
    conn = get_conn()

    print("Fetching conversation IDs…")
    conv_ids = fetch_conversation_ids(conn)
    print(f"  → {len(conv_ids)} conversations with at least one page reply")

    all_samples: list[dict] = []
    stats = Counter()

    for conv_id in conv_ids:
        messages = fetch_messages(conn, conv_id)
        if not messages:
            stats["empty"] += 1
            continue

        turns = build_turns(messages)
        samples = turns_to_samples(turns)

        if not samples:
            stats["no_valid_sample"] += 1
            continue

        all_samples.extend(samples)
        stats["ok"] += 1

    conn.close()

    # ── Stats ─────────────────────────────────────────────────────────────────
    turn_counts = [len(s["messages"]) - 1 for s in all_samples]  # exclude system
    print(f"\n{'─'*50}")
    print(f"Conversations processed : {stats['ok']}")
    print(f"Conversations skipped   : {stats['empty'] + stats['no_valid_sample']}")
    print(f"Total samples           : {len(all_samples)}")
    print(
        f"Turns/sample  min={min(turn_counts)}  max={max(turn_counts)}  "
        f"avg={sum(turn_counts)/len(turn_counts):.1f}"
    )
    print(f"{'─'*50}")

    # ── Shuffle + split ───────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    val_n = max(1, int(len(all_samples) * VAL_RATIO))
    val_samples = all_samples[:val_n]
    trn_samples = all_samples[val_n:]

    # ── Write JSONL ───────────────────────────────────────────────────────────
    def write_jsonl(path: Path, data: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_FILE, trn_samples)
    write_jsonl(VAL_FILE, val_samples)

    print(f"\nTrain → {TRAIN_FILE}  ({len(trn_samples)} samples)")
    print(f"Val   → {VAL_FILE}  ({len(val_samples)} samples)")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
