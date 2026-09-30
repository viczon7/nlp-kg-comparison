"""Step 4: draft multiple-choice questions from the passages (not from the KG), to check by hand.

Usage:  python scripts/4_draft_questions.py   (Ollama running, qwen3.5:9b pulled)
Output: eval/questions_draft.csv  (";"-separated: fill "keep" with 1/0, fix wording where needed)
"""
import csv
import glob
import json
import os

import ollama

MODEL = "qwen3.5:9b"  # the extraction model, not one of the tested Gemma models
PASSAGES = (glob.glob("/kaggle/input/**/passages.jsonl", recursive=True) or ["data/passages.jsonl"])[0]

PROMPT = """Write 2 multiple-choice questions that test facts stated in the text about multilingual NLP.
Rules:
- The answer is a short fact copied exactly from the text: a name, number, language, dataset, model or metric.
- The question must make sense without the text: name the dataset, model or benchmark,
  never write "this paper", "the authors" or "the proposed method".
- Give 4 options: the FIRST is the correct answer, the other 3 are wrong options of the same kind
  (e.g. other languages or numbers).
Return JSON: {"questions": [{"question": "...", "options": ["correct answer", "wrong", "wrong", "wrong"]}]}"""

passages = [json.loads(line) for line in open(PASSAGES, encoding="utf-8")]
os.makedirs("eval", exist_ok=True)

with open("eval/questions_draft.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["keep", "question", "gold", "wrong1", "wrong2", "wrong3", "paper_id", "split", "passage_id"])

    for i, p in enumerate(passages, 1):
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": p["text"]}],
            format="json", think=False,
            options={"temperature": 0, "seed": 42, "presence_penalty": 0, "num_predict": 1024},
        )
        try:
            questions = json.loads(response.message.content)["questions"]
        except (json.JSONDecodeError, KeyError):
            questions = []

        kept = 0
        for q in questions:
            question, options = q.get("question", ""), q.get("options", [])
            answer = options[0] if options else ""  # the first option is the correct one
            wrong = [w for w in dict.fromkeys(options[1:]) if w and w.lower() != answer.lower()][:3]
            # keep only questions whose answer really occurs in the passage
            if question and answer and len(wrong) == 3 and answer.lower() in p["text"].lower():
                writer.writerow(["", question, answer, *wrong, p["paper_id"], p["split"], p["passage_id"]])
                kept += 1
        f.flush()
        print(f"[{i}/{len(passages)}] {p['passage_id']}: {kept} questions")
