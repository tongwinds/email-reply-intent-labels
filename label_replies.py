"""
Label sales email replies by buying intent using a free OpenRouter model.

Free OpenRouter models allow only 20 requests per minute and 50 requests per day
(until you have bought 10 credits). To label 840 replies inside that limit, this
script sends several replies per request (--batch-size, default 20), so 840 replies
need about 42 requests.

Usage:
    python label_replies.py --check                         (see your remaining limits)
    python label_replies.py --input replies.csv --fresh --limit 20   (test, 1 request)
    python label_replies.py --input replies.csv             (full run, resumes where it stopped)
    python label_replies.py --input replies.csv --retry-failed

Files it needs:
    replies.csv           the export of the 840 replies
    labeling_guide.md     the labeling guide (definitions and examples)
    .env                  one line:  OPENROUTER_API_KEY=your_key   (never commit it)

Files it creates:
    labeled_replies.csv   id, text, label, reason
    failed.csv            id, error  (replies that failed, to retry them)
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
KEY_URL = "https://openrouter.ai/api/v1/key"
DEFAULT_MODEL = "inclusionai/ling-3.2-flash:free"
VALID_LABELS = ["High", "Medium", "Low", "None", "Auto-reply"]
RESULTS_FILE = "labeled_replies.csv"
FAILED_FILE = "failed.csv"


class DailyLimitReached(Exception):
    """The free daily request limit is used up. Stop and continue tomorrow."""


def load_env(path=".env"):
    """Read the .env file without extra libraries. An existing environment variable also works."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def check_limits(key):
    """Print the limits OpenRouter reports for this key (never prints the key itself)."""
    r = requests.get(KEY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    print("HTTP status:", r.status_code)
    try:
        data = r.json().get("data", r.json())
    except ValueError:
        print(r.text[:300])
        return
    for field in ("is_free_tier", "usage", "limit", "limit_remaining", "free_model_daily_requests"):
        if field in data:
            print(f"{field}: {data[field]}")


def is_id_column(name):
    low = name.lower()
    return low == "id" or low.endswith("_id") or low.endswith(" id")


def guess_text_column(columns):
    """Pick the column that holds the reply text, never an id column."""
    candidates = [c for c in columns if not is_id_column(c)]
    for hint in ("body", "text", "message", "content", "response", "reply"):
        for c in candidates:
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
            rows.append({"id": str(row.get("id") or row.get("reply_id") or i), "text": row[col]})
    print(f"Using text column: {col}")
    if rows and rows[0]["text"].strip() == rows[0]["id"]:
        sys.exit(f"The text column '{col}' contains the ids, not the replies. "
                 f"Use --text-column with the right name. Columns: {columns}")
    return rows


def build_messages(guide, batch):
    system = (
        "You classify sales email replies by how ready the sender is to buy. "
        "Follow this labeling guide:\n\n"
        f"{guide}\n\n"
        "You will receive several replies, each with an id. Answer ONLY with a valid JSON "
        "array, no extra text, with one object per reply, in the same order, in this exact "
        f'format: [{{"id": "<the reply id>", "label": "<one of {VALID_LABELS}>", '
        '"reason": "<reason of 20 words or fewer>"}]'
    )
    replies = "\n\n".join(f'--- id: {r["id"]} ---\n{r["text"]}' for r in batch)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Label these {len(batch)} replies:\n\n{replies}"},
    ]


def parse_batch_answer(raw, batch):
    """Return ({id: (label, reason)}, {id: error}) for the replies in this batch."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in the answer: {raw[:120]}")
    items = json.loads(raw[start:end + 1])
    wanted = [r["id"] for r in batch]
    by_id = {}
    for item in items:
        if isinstance(item, dict) and "id" in item:
            by_id[str(item["id"]).strip()] = item
    # If ids are missing but the count matches, trust the order.
    if not by_id and len(items) == len(wanted):
        by_id = {wanted[i]: items[i] for i in range(len(items)) if isinstance(items[i], dict)}

    ok, bad = {}, {}
    for rid in wanted:
        item = by_id.get(rid)
        if item is None:
            bad[rid] = "missing from the model answer"
            continue
        label = str(item.get("label", "")).strip().capitalize()
        if label not in VALID_LABELS:
            bad[rid] = f"invalid label: {label}"
            continue
        ok[rid] = (label, str(item.get("reason", "")).strip())
    return ok, bad


def classify_batch(key, model, guide, batch, retries=2):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": build_messages(guide, batch), "temperature": 0}
    last_error = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=120)
        except requests.RequestException as e:
            last_error = f"network error: {e}"
            time.sleep(10)
            continue

        if r.status_code == 429:
            text = r.text[:300]
            low = text.lower()
            if "per-day" in low or "per day" in low or "daily" in low:
                raise DailyLimitReached(text)
            last_error = f"HTTP 429 (per-minute limit or busy provider): {text}"
            time.sleep(65)
            continue
        if r.status_code >= 400:
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(10)
            continue
        try:
            raw = r.json()["choices"][0]["message"]["content"]
            return parse_batch_answer(raw, batch)
        except Exception as e:  # noqa: BLE001
            last_error = f"bad answer: {e}"
            time.sleep(5)
    raise RuntimeError(last_error)


def load_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Free OpenRouter model (ends in :free)")
    p.add_argument("--guide", default="labeling_guide.md")
    p.add_argument("--text-column", default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N replies (for testing)")
    p.add_argument("--batch-size", type=int, default=20, help="Replies sent per request")
    p.add_argument("--sleep", type=float, default=4.0, help="Seconds to wait between requests")
    p.add_argument("--retry-failed", action="store_true", help="Only re-run replies in failed.csv")
    p.add_argument("--fresh", action="store_true",
                   help="Delete previous results and failures first (use after editing the guide)")
    p.add_argument("--check", action="store_true", help="Show your OpenRouter limits and exit")
    args = p.parse_args()

    load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is missing from your .env file")

    if args.check:
        check_limits(key)
        return
    if not args.input:
        sys.exit("Use --input replies.csv (or --check to see your limits)")

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
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    print(f"To label: {len(rows)} replies in {len(batches)} requests")

    new_results = not os.path.exists(RESULTS_FILE)
    new_failed = not os.path.exists(FAILED_FILE)
    labeled = 0
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as fo, \
         open(FAILED_FILE, "a", encoding="utf-8", newline="") as ff:
        out = csv.writer(fo)
        fail = csv.writer(ff)
        if new_results:
            out.writerow(["id", "text", "label", "reason"])
        if new_failed:
            fail.writerow(["id", "error"])

        for n, batch in enumerate(batches, start=1):
            text_by_id = {r["id"]: r["text"] for r in batch}
            try:
                ok, bad = classify_batch(key, args.model, guide, batch)
            except DailyLimitReached as e:
                print(f"\nDaily free limit reached after {labeled} labeled replies.")
                print(f"OpenRouter said: {e}")
                print("Run the same command again after the limit resets. It resumes where it stopped.")
                break
            except Exception as e:  # noqa: BLE001
                for r in batch:
                    fail.writerow([r["id"], str(e)[:200]])
                ff.flush()
                print(f"[request {n}/{len(batches)}] FAILED: {e}")
                time.sleep(args.sleep)
                continue

            for rid, (label, reason) in ok.items():
                out.writerow([rid, text_by_id[rid], label, reason])
            for rid, err in bad.items():
                fail.writerow([rid, err])
            fo.flush()
            ff.flush()
            labeled += len(ok)
            print(f"[request {n}/{len(batches)}] labeled {len(ok)}, failed {len(bad)}")
            time.sleep(args.sleep)

    print(f"Done. Labeled this run: {labeled}. Check labeled_replies.csv and failed.csv")


if __name__ == "__main__":
    main()
