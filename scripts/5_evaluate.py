import glob
import random
import re

import ollama
import pandas as pd


def find(name):
    return (glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True))[0]


def name(s):  # "MultiBLiMP 1.0" -> "multiblimp", so it also matches questions without the version
    return re.sub(r"\s+\d+(\.\d+)*$", "", s).lower()


def parse(reply):  # first standalone letter; the run used r"\b[ABCD]\b", which also matched the D in "BRAVE-D"
    return (re.findall(r"(?<![\w-])[ABCD](?![\w-])", reply) or ["-"])[0]


questions = pd.read_csv(find("questions_final.csv"), sep=";", dtype=str, encoding="utf-8-sig")
questions = questions[questions["keep"] == "1"].reset_index(drop=True)   # accepted questions only
kg = pd.read_csv(find("kg_final.csv"), dtype=str, encoding="utf-8-sig")

CONDITIONS = [("small", "gemma4:e2b", False), ("small+KG", "gemma4:e2b", True), ("mid", "gemma4:12b", False)]

results = []
for i, q in questions.iterrows():
    options = [q["gold"], q["wrong1"], q["wrong2"], q["wrong3"]]
    random.Random(i).shuffle(options)                      # fixed option order per question
    gold = "ABCD"[options.index(q["gold"])]
    text = q["question"] + "\n" + "\n".join(f"{l}) {o}" for l, o in zip("ABCD", options))

    # KG facts: triples whose head or tail is mentioned in the question
    facts = [f"{t['head']} | {t['relation']} | {t['tail']}" for _, t in kg.iterrows()
             if name(t["head"]) in q["question"].lower() or name(t["tail"]) in q["question"].lower()]

    for condition, model, use_kg in CONDITIONS:
        prompt = text + "\n\nIf you are not sure, pick the most likely option. Answer with the letter (A, B, C or D) only."
        if use_kg and facts:
            prompt = "Facts:\n" + "\n".join(facts[:40]) + "\n\n" + prompt
        reply = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], think=False,
                            options={"temperature": 0, "seed": 42, "num_predict": 30}).message.content
        answer = parse(reply)
        results.append({"question": i, "condition": condition, "split": q["split"],
                        "gold": gold, "answer": answer, "correct": int(answer == gold), "reply": reply})
    print(f"[{i + 1}/{len(questions)}]", " ".join(f"{r['condition']}={r['correct']}" for r in results[-len(CONDITIONS):]))

results = pd.DataFrame(results)
results.to_csv("results.csv", index=False)
print(results.pivot_table(index="condition", columns="split", values="correct", aggfunc="mean", margins=True))
