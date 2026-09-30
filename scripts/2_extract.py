"""Step 2: extract (head, relation, tail) triples from each passage with an open model.

Usage:  pip install ollama  &&  ollama pull qwen3.5:9b  &&  python scripts/2_extract.py
Output: kg/triples_to_curate.csv  (fill the "keep" column with 1 or 0)
"""
import csv
import glob
import json
import os

import ollama

MODEL = "qwen3.5:9b"
# the Kaggle dataset if there is one, otherwise the local file
PASSAGES = (glob.glob("/kaggle/input/**/passages.jsonl", recursive=True) or ["data/passages.jsonl"])[0]
RELATIONS = ["introduces", "covers_language", "belongs_to_family", "designed_for",
             "evaluated_on", "measured_by", "derived_from", "has_statistic"]

PROMPT = """Extract ALL facts about multilingual NLP from the text as (head, relation, tail) triples.
Use only these relations: introduces (head is THIS_PAPER), covers_language, belongs_to_family,
designed_for, evaluated_on, measured_by, derived_from, has_statistic (tail is a number with unit).
Only use facts stated in the text and copy names exactly as written.
Return JSON: {"triples": [{"head": "...", "relation": "...", "tail": "..."}]}"""

passages = [json.loads(line) for line in open(PASSAGES, encoding="utf-8")]
os.makedirs("kg", exist_ok=True)

with open("kg/triples_to_curate.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["keep", "head", "relation", "tail", "paper_id", "split", "passage_id"])

    for i, p in enumerate(passages, 1):
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": p["text"]}],
            format="json", think=False,
            # fixed settings for reproducibility; num_predict stops an answer that loops forever
            options={"temperature": 0, "seed": 42, "presence_penalty": 0, "num_predict": 2048},
        )
        try:
            triples = json.loads(response.message.content)["triples"]
        except (json.JSONDecodeError, KeyError):
            triples = []  # e.g. an answer cut off by num_predict

        text = p["text"].lower()
        kept = 0
        for t in triples:
            head, relation, tail = t.get("head", ""), t.get("relation", ""), t.get("tail", "")
            if relation == "introduces":
                head = p["title"]
            # keep only allowed relations whose head and tail really occur in the passage
            if relation in RELATIONS and head and tail and \
                    (head == p["title"] or head.lower() in text) and tail.lower() in text:
                writer.writerow(["", head, relation, tail, p["paper_id"], p["split"], p["passage_id"]])
                kept += 1
        f.flush()
        print(f"[{i}/{len(passages)}] {p['passage_id']}: {kept} triples")
