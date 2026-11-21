"""
Etiqueta cada respuesta de email por intencion de compra usando OpenRouter.

Uso basico:
    python label_replies.py --input replies.csv --model NOMBRE_DEL_MODELO:free --limit 20
    python label_replies.py --input replies.csv --model NOMBRE_DEL_MODELO:free
    python label_replies.py --input replies.csv --model NOMBRE_DEL_MODELO:free --retry-failed

Archivos que necesita:
    replies.csv           tu export de las 840 respuestas
    labeling_guide.md     tu guia de etiquetas (definiciones y ejemplos)
    .env                  una linea:  OPENROUTER_API_KEY=tu_llave   (NO lo subas a GitHub)

Archivos que genera:
    labeled_replies.csv   id, texto, label, reason
    failed.csv            las que fallaron (para repetirlas)
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
VALID_LABELS = ["High", "Medium", "Low", "None"]
RESULTS_FILE = "labeled_replies.csv"
FAILED_FILE = "failed.csv"


def load_env(path=".env"):
    """Lee .env sin librerias extra. En Colab tambien funciona si ya hay variable de entorno."""
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
            sys.exit(f"No encontre la columna del texto. Columnas disponibles: {columns}. "
                     f"Usa --text-column NOMBRE.")
        rows = []
        for i, row in enumerate(reader, start=1):
            rows.append({"id": row.get("id") or row.get("reply_id") or str(i), "text": row[col]})
    return rows


def build_messages(guide, text):
    system = (
        "Eres un asistente que clasifica respuestas de email de ventas segun que tan listo "
        "esta quien responde para comprar. Sigue esta guia de etiquetas:\n\n"
        f"{guide}\n\n"
        f"Responde SOLO con JSON valido, sin texto extra, con este formato exacto: "
        f'{{"label": "<una de {VALID_LABELS}>", "reason": "<razon de maximo 20 palabras>"}}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Respuesta del cliente:\n\"\"\"\n{text}\n\"\"\""},
    ]


def parse_answer(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Sin JSON en la respuesta: {raw[:120]}")
    data = json.loads(match.group(0))
    label = str(data.get("label", "")).strip().capitalize()
    if label not in VALID_LABELS:
        raise ValueError(f"Etiqueta invalida: {label}")
    return label, str(data.get("reason", "")).strip()


def classify(key, model, guide, text, retries=4):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": build_messages(guide, text), "temperature": 0}
    wait = 5
    last_error = None
    for _ in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 429:  # limite de uso de los modelos gratis
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


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="inclusionai/ling-3.2-flash:free",
                   help="Modelo gratis de OpenRouter (termina en :free)")
    p.add_argument("--guide", default="labeling_guide.md")
    p.add_argument("--text-column", default=None)
    p.add_argument("--limit", type=int, default=None, help="Solo las primeras N (para probar)")
    p.add_argument("--sleep", type=float, default=3.0, help="Segundos entre llamadas")
    p.add_argument("--retry-failed", action="store_true", help="Solo repite las de failed.csv")
    args = p.parse_args()

    load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Falta OPENROUTER_API_KEY en tu .env")

    with open(args.guide, encoding="utf-8") as f:
        guide = f.read()

    rows = read_rows(args.input, args.text_column)

    if args.retry_failed:
        failed_ids = load_done(FAILED_FILE)
        rows = [r for r in rows if r["id"] in failed_ids]
        os.remove(FAILED_FILE)
    if args.limit:
        rows = rows[: args.limit]

    done = load_done(RESULTS_FILE)
    rows = [r for r in rows if r["id"] not in done]
    print(f"Por etiquetar: {len(rows)}")

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
                print(f"[{n}/{len(rows)}] {row['id']} FALLO: {e}")
            time.sleep(args.sleep)

    print("Listo. Revisa labeled_replies.csv y failed.csv")


if __name__ == "__main__":
    main()
