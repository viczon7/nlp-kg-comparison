import csv
import json
import re
import statistics
import unicodedata
import urllib.request
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdf"
MAX_WORDS = 500
SETUP_MAX_WORDS = 1000
CUTOFF = "2025-01"

# ACL Anthology id
PAPERS = [
    ("2026.tacl-1.10", "2025-04"),
    ("2026.tacl-1.19", "2025-10"),
    ("2025.tacl-1.75", "2025-08"),
    ("2026.tacl-1.56", "2026-02"),
    ("2026.cl-1.6", "2024-04"),
    ("2025.cl-2.1", "2024-09"),
    ("2025.tacl-1.48", "2024-09"),
    ("2025.tacl-1.24", "2025-02"),
    ("2025.tacl-1.51", "2024-03"),
    ("2026.tacl-1.4", "2025-09"),
    ("2026.cl-1.9", "2024-06"),
    ("2026.tacl-1.9", "2024-08"),
    ("2026.cl-1.7", "2024-07"),
    ("2025.tacl-1.4", "2024-01"),
    ("2026.cl-2.2", "2025-11"),
    ("2025.emnlp-main.834", "2025-06"),
    ("2025.acl-long.1112", "2025-02"),
    ("2025.naacl-long.12", "2024-08"),
    ("2025.acl-long.701", "2025-02"),
    ("N16-1083", "2016-06"),
    ("L14-1232", "2014-05"),
    ("L14-1746", "2014-05"),
    ("2025.acl-srw.4", "2024-06"),
]

HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*[A-Z].{1,90}|Abstract|References)$")
BOLD_RE = re.compile(r"bold|medi|semi|black|heavy", re.I)
NOISE_RE = re.compile(r"Proceedings of|©|Association for Computational Linguistics|"
                      r"Transactions of the|\S+@\S+\.\w+|^\d{1,4}$")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kg-poster/1.0 (academic use)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def bib_field(bib: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*\"(.*?)\",?\s*\n", bib, re.S)
    return re.sub(r"[{}]|\s+(?=\s)", "", m.group(1)).replace("\n", " ").strip() if m else ""


def pdf_lines(pdf: Path):
    """Yield (text, is_heading) per line, in reading order, without footnotes/headers."""
    doc = pymupdf.open(pdf)
    flags = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE
    pages = [p.get_text("dict", flags=flags)["blocks"] for p in doc]
    sizes = [s["size"] for b in pages for bl in b for l in bl.get("lines", []) for s in l["spans"]]
    body = statistics.mode(round(s, 1) for s in sizes)
    for blocks in pages:
        for block in blocks:
            prev_number = None
            first = "".join(s["text"] for l in block.get("lines", [])[:1] for s in l["spans"]).strip()
            if re.match(r"^(Figure|Table|Fig\.)\s*\d+\s*[:.]", first):
                continue  # figure/table captions interrupt the running text
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = unicodedata.normalize("NFKC", "".join(s["text"] for s in line["spans"]))
                text = re.sub(r"\s+", " ", text).strip()
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & 16 or BOLD_RE.search(s["font"]) for s in spans)
                if bold and size >= body - 0.5 and re.fullmatch(r"\d+\.?", text):
                    prev_number = text
                    continue
                if size < body - 1.5 or NOISE_RE.search(text):
                    continue  # footnotes, page numbers, venue footers, e-mails
                if prev_number:
                    text, prev_number = f"{prev_number} {text}", None
                yield text, bool(bold and HEADING_RE.match(text) and size >= body - 0.5), block["number"]


def read_sections(pdf: Path) -> list[tuple[str, str]]:
    """(title, text) per top-level section, subsections included; "front" = title, authors, abstract."""
    sections, lines, last_block = [("front", [])], None, None
    for text, is_heading, block_no in pdf_lines(pdf):
        lines = sections[-1][1]
        if is_heading:
            if text.startswith("References"):
                break
            number = re.match(r"^(\d+)", text)
            current_number = re.match(r"^(\d+)", sections[-1][0])
            if number and not (current_number and number.group(1) == current_number.group(1)):
                sections.append((text, []))  # new top-level section ("3 Data"), not "3.1 ..."
            continue
        # same block, or a sentence running on into the next column/page
        continues = lines and (block_no == last_block or not re.search(r"[.!?:]$", lines[-1]))
        if continues:
            lines[-1] = lines[-1][:-1] + text if lines[-1].endswith("-") and text[:1].islower() \
                else f"{lines[-1]} {text}"
        else:
            lines.append(text)
        last_block = block_no
    return [(title, "\n".join(lines)) for title, lines in sections]


def pick_parts(sections: list[tuple[str, str]]) -> dict:
    """Abstract (front matter), Introduction, and one setup section (>= 150 words), by priority:
    experiments/setup > data/benchmark/corpus/evaluation > method > first section after the intro."""
    def find(pattern):
        return next(((t, s) for t, s in sections[1:] if re.search(pattern, t, re.I)
                     and not re.search(r"related|background|introduction|conclusion|references", t, re.I)
                     and len(s.split()) >= 150), None)
    intro = next((s for t, s in sections if re.match(r"^1\.?\s*Introduction", t)), "")
    setup = find(r"experiment|set-?up") or find(r"data|benchmark|corpus|evaluation") \
        or find(r"method") or find(r".")
    setup_text = " ".join(setup[1].split(" ")[:SETUP_MAX_WORDS]) if setup else ""
    return {"abstract": sections[0][1], "introduction": intro, "setup": setup_text,
            "setup_title": setup[0] if setup else ""}


def split_passages(text: str) -> list[str]:
    passages, current = [], []
    for para in text.split("\n"):
        if current and len(" ".join(current + [para]).split()) > MAX_WORDS:
            passages.append("\n".join(current))
            current = []
        current.append(para)
    if current:
        passages.append("\n".join(current))
    return passages


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    rows, passages = [], []
    for pid, first_public in PAPERS:
        pdf = PDF_DIR / f"{pid}.pdf"
        if not pdf.exists():
            pdf.write_bytes(fetch(f"https://aclanthology.org/{pid}.pdf"))
        bib = fetch(f"https://aclanthology.org/{pid}.bib").decode("utf-8")
        split = "new" if first_public > CUTOFF else "old"  # the cutoff month itself counts as known
        row = {"id": pid, "title": bib_field(bib, "title"),
               "venue": bib_field(bib, "journal") or bib_field(bib, "booktitle"),
               "year": bib_field(bib, "year"), "first_public": first_public, "split": split,
               "url": f"https://aclanthology.org/{pid}/"}
        rows.append(row)

        chosen = pick_parts(read_sections(pdf))
        assert chosen["introduction"], f"no introduction found in {pid}"
        parts = [(part, p) for part in ("abstract", "introduction", "setup")
                 for p in split_passages(chosen[part]) if p.strip()]
        for i, (part, text) in enumerate(parts):
            passages.append({"passage_id": f"{pid}#{i}", "paper_id": pid, "title": row["title"],
                             "split": split, "part": part, "text": text})
        print(f"{pid:22} {split:3} {len(parts)} passages  setup = {chosen['setup_title'][:40]!r:44} "
              f"({len(chosen['setup'].split())} words)")

    with open(ROOT / "data" / "papers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "data" / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    words = sum(len(p["text"].split()) for p in passages)
    print(f"\n{len(rows)} papers, {len(passages)} passages, {words} words -> data/passages.jsonl")


if __name__ == "__main__":
    main()
