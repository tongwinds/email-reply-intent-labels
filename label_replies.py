"""
Label each sales email reply by buying intent using an OpenRouter model.

Basic usage:
    python label_replies.py --input replies.csv --limit 20
    python label_replies.py --input replies.csv
    python label_replies.py --input replies.csv --retry-failed
    python label_replies.py --input replies.csv --fresh --limit 20   (start over)

Files it needs:
    replies.csv           the export of the 840 replies
    labeling_guide.md     the labeling guide (definitions and examples)
    .env                  one line:  OPENROUTER_API_KEY=your_key   (never commit it)

Files it creates:
    labeled_replies.csv   id, text, label, reason
    failed.csv            replies that failed (to retry them)
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "inclusionai/ling-3.2-flash:free"
VALID_LABELS = ["High", "Medium", "Low", "None", "Auto-reply"]
RESULTS_FILE = "labeled_replies.csv"
FAILED_FILE = "failed.csv"


def load_env(path=".env"):
    """Read the .env file without extra libraries. An existing environment variable also works."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def guess_text_column(columns):
    for hint in ("reply", "body", "text", "message", "response", "content"):
        for c in columns:
            if hint in c.lower():
                return c
    return None


def read_rows(path, text_column):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        col = text_column or guess_text_column(columns)
        if col is None or col not in columns:
            sys.exit(f"Could not find the text column. Available columns: {columns}. "
                     f"Use --text-column NAME.")
        rows = []
        for i, row in enumerate(reader, start=1):
            rows.append({"id": row.get("id") or row.get("reply_id") or str(i), "text": row[col]})
    return rows


def build_messages(guide, text):
    system = (
        "You classify sales email replies by how ready the sender is to buy. "
        "Follow this labeling guide:\n\n"
        f"{guide}\n\n"
        f"Answer ONLY with valid JSON, no extra text, in this exact format: "
        f'{{"label": "<one of {VALID_LABELS}>", "reason": "<reason of 20 words or fewer>"}}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Customer reply:\n\"\"\"\n{text}\n\"\"\""},
    ]


def parse_answer(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in the answer: {raw[:120]}")
    data = json.loads(match.group(0))
    label = str(data.get("label", "")).strip().capitalize()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label}")
    return label, str(data.get("reason", "")).strip()


def classify(key, model, guide, text, retries=4):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": build_messages(guide, text), "temperature": 0}
    wait = 5
    last_error = None
    for _ in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 429:  # rate limit on free models: wait longer and retry
                time.sleep(wait)
                wait *= 2
                continue
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            return parse_answer(raw)
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(wait)
            wait *= 2
    raise RuntimeError(str(last_error))


def load_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Free OpenRouter model (ends in :free)")
    p.add_argument("--guide", default="labeling_guide.md")
    p.add_argument("--text-column", default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N replies (for testing)")
    p.add_argument("--sleep", type=float, default=3.0, help="Seconds to wait between calls")
    p.add_argument("--retry-failed", action="store_true", help="Only re-run replies in failed.csv")
    p.add_argument("--fresh", action="store_true",
                   help="Delete previous results and failures first (use after editing the guide)")
    args = p.parse_args()

    load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is missing from your .env file")

    if args.fresh:
        for path in (RESULTS_FILE, FAILED_FILE):
            if os.path.exists(path):
                os.remove(path)
        print("Previous results deleted. Starting fresh.")

    with open(args.guide, encoding="utf-8") as f:
        guide = f.read()

    rows = read_rows(args.input, args.text_column)

    if args.retry_failed:
        failed_ids = load_ids(FAILED_FILE)
        rows = [r for r in rows if r["id"] in failed_ids]
        if os.path.exists(FAILED_FILE):
            os.remove(FAILED_FILE)
    if args.limit:
        rows = rows[: args.limit]

    done = load_ids(RESULTS_FILE)
    rows = [r for r in rows if r["id"] not in done]
    print(f"To label: {len(rows)}")

    new_results = not os.path.exists(RESULTS_FILE)
    new_failed = not os.path.exists(FAILED_FILE)
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as fo, \
         open(FAILED_FILE, "a", encoding="utf-8", newline="") as ff:
        out = csv.writer(fo)
        fail = csv.writer(ff)
        if new_results:
            out.writerow(["id", "text", "label", "reason"])
        if new_failed:
            fail.writerow(["id", "error"])

        for n, row in enumerate(rows, start=1):
            try:
                label, reason = classify(key, args.model, guide, row["text"])
                out.writerow([row["id"], row["text"], label, reason])
                fo.flush()
                print(f"[{n}/{len(rows)}] {row['id']} -> {label}")
            except Exception as e:  # noqa: BLE001
                fail.writerow([row["id"], str(e)[:200]])
                ff.flush()
                print(f"[{n}/{len(rows)}] {row['id']} FAILED: {e}")
            time.sleep(args.sleep)

    print("Done. Check labeled_replies.csv and failed.csv")


if __name__ == "__main__":
    main()
