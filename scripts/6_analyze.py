"""Step 6: all numbers of the poster, computed from the evaluation results.

Input
  eval/results.csv, eval/questions_final.csv, kg/kg_final.csv
Output
  printed: accuracy by condition and old/new papers, no-answer rates,
  accuracy when the answer is / is not in the retrieved facts

Usage:  python scripts/6_analyze.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
res = pd.read_csv(ROOT / "eval/results.csv")
questions = pd.read_csv(ROOT / "eval/questions_final.csv", sep=";", dtype=str, encoding="utf-8-sig")
questions = questions[questions["keep"] == "1"].reset_index(drop=True)
kg = pd.read_csv(ROOT / "kg/kg_final.csv", dtype=str, encoding="utf-8-sig")


def name(s):  # same matching as 5_evaluate.py
    return re.sub(r"\s+\d+(\.\d+)*$", "", s).lower()


def parse(reply):  # fixed pattern of 5_evaluate.py; the run's \b[ABCD]\b read "BRAVE-D" (question 162) as D
    return (re.findall(r"(?<![\w-])[ABCD](?![\w-])", reply) or ["-"])[0]


def in_facts(q):  # is the gold answer in the facts that E2B + KG was given?
    facts = [f"{h} | {r} | {t}" for h, r, t in kg[["head", "relation", "tail"]].values
             if name(h) in q["question"].lower() or name(t) in q["question"].lower()][:40]
    return any(q["gold"].lower() in f.lower() for f in facts)


res["condition"] = res["condition"].map({"small": "E2B", "small+KG": "E2B + KG", "mid": "12B"})
res["answer"] = res["reply"].fillna("").map(parse)
res["correct"] = (res["answer"] == res["gold"]).astype(int)
covered = questions.apply(in_facts, axis=1)
res["in_facts"] = res["question"].map(covered).map({True: "answer in facts", False: "not in facts"})

print("Accuracy (%)")
print((100 * res.pivot_table(index="condition", columns="split", values="correct", aggfunc="mean", margins=True)).round(1))
print("\nAccuracy (%) by whether the retrieved facts hold the answer")
print((100 * res.pivot_table(index="condition", columns="in_facts", values="correct", aggfunc="mean")).round(1))
print(f"answer in facts: {covered.sum()} of {len(covered)} questions "
      f"(old {covered[questions['split'] == 'old'].mean():.0%}, new {covered[questions['split'] == 'new'].mean():.0%})")

print()
for c, d in res.groupby("condition"):
    print(f"{c}: no answer {100 * (d['answer'] == '-').mean():.1f}%")
