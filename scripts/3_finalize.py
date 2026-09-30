"""Step 3: turn the curated sheet into the final knowledge graph.

Input
  kg/triples_to_curate.csv   with "keep" filled in (1 = correct, 0 = wrong)
Output
  kg/kg_final.csv            head, relation, tail + the papers it comes from

A triple counts as "old" knowledge if any source paper is old, otherwise "new".

Usage:  python scripts/3_finalize.py
"""
import csv
from collections import Counter
from pathlib import Path

KG = Path(__file__).resolve().parent.parent / "kg"


def main():
    with open(KG / "triples_to_curate.csv", encoding="utf-8-sig") as f:
        header = f.readline()
        f.seek(0)
        # Excel with a German/Russian locale saves CSV with ";" instead of ","
        rows = list(csv.DictReader(f, delimiter=";" if header.count(";") > header.count(",") else ","))
    checked = [r for r in rows if r["keep"].strip()]
    kept = [r for r in checked if r["keep"].strip().lower() in ("1", "y", "yes", "x")]

    triples = {}  # (head, relation, tail) -> source papers and their old/new labels
    for r in kept:
        head, tail = r["head"].strip(), r["tail"].strip()
        t = triples.setdefault((head.lower(), r["relation"], tail.lower()),
                               {"head": head, "relation": r["relation"], "tail": tail, "papers": {}})
        t["papers"][r["paper_id"]] = r["split"]

    with open(KG / "kg_final.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["head", "relation", "tail", "split", "papers"])
        for t in triples.values():
            split = "old" if "old" in t["papers"].values() else "new"
            w.writerow([t["head"], t["relation"], t["tail"], split, ";".join(sorted(t["papers"]))])

    print(f"curated {len(checked)}/{len(rows)} candidates, {len(kept)} marked correct "
          f"(extractor precision {len(kept) / max(len(checked), 1):.0%})")
    print(f"final KG: {len(triples)} unique triples -> kg/kg_final.csv")
    print("by relation:", dict(Counter(t["relation"] for t in triples.values()).most_common()))
    print("old/new    :", dict(Counter("old" if "old" in t["papers"].values() else "new"
                                       for t in triples.values())))


if __name__ == "__main__":
    main()
