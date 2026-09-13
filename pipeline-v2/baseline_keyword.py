"""
Keyword / definitional-pattern baseline (no LLM, no graph).

For each injection it assembles the full injected course (clean rendered
lectures plus the one injected lecture, in lecture order), derives a concept
vocabulary by surface patterns, and detects two structural-defect types by
deterministic pattern matching:

  used_before_introduction
    a vocabulary term appears (as a substring) in a lecture earlier than the
    earliest lecture in which it appears in a definitional pattern;
  orphan_concept (used_without_introduction)
    a vocabulary term appears but never appears in any definitional pattern.

Recall for an injection is a hit when the injected concept/term is flagged by
the baseline with the matching defect type. The baseline never receives the
injected concept name — it must surface it on its own.

It also writes, per injection, the assembled full course to
  injections/<id>/course.md
which the LLM-direct baseline reads.

Usage: python pipeline-v2/baseline_keyword.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"

# definitional-pattern templates; {t} is the (regex-escaped) term
DEF_PATTERNS = [
    r"\*\*{t}\*\*",                 # bold term
    r"^#{{2,6}}\s*{t}\b",          # heading term
    r"{t}\s*[—–-]\s*це\b",         # "T — це ..."
    r"{t}\s*[—–-]\s",               # "T — ..." (dash definition)
    r"{t}\s+is\s+",                 # "T is ..."
    r"{t}\s+means\b",
    r"{t}\s*:\s",                   # "T: ..."
]


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def assemble_course(root, specs_lec, inj_id, insert_lecture, manifest):
    parts = []
    for m in manifest:
        lid = m["lecture_id"]
        if lid == insert_lecture:
            p = root / "injections" / inj_id / f"{lid}.md"
        else:
            p = root / "rendered" / f"{lid}.md"
        parts.append((m["lecture_order"], lid, p.read_text(encoding="utf-8")))
    parts.sort()
    return parts  # list of (order, lecture_id, text)


def build_vocabulary(course_text):
    """Surface-derived candidate terms: bold spans, headings, Latin-script
    multi-word/Capitalized phrases. Returns a set of term strings."""
    vocab = set()
    for m in re.finditer(r"\*\*([^*\n]{3,60})\*\*", course_text):
        vocab.add(m.group(1).strip())
    for m in re.finditer(r"^#{2,6}\s*(.+)$", course_text, re.M):
        h = re.sub(r"^[\d.\s]+", "", m.group(1)).strip()
        if 3 <= len(h) <= 60:
            vocab.add(h)
    # Latin-script terms / phrases (CamelCase, multi-word capitalized, acronyms)
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9.+-]*(?:\s+[A-Z][A-Za-z0-9.+-]*){0,3})\b", course_text):
        t = m.group(1).strip()
        if 3 <= len(t) <= 40:
            vocab.add(t)
    # clean
    out = set()
    for t in vocab:
        t = t.strip(" :—–-")
        if t and not t.isdigit():
            out.add(t)
    return out


def first_appearance_order(term, parts):
    tl = term.lower()
    for order, lid, text in parts:
        if tl in text.lower():
            return order
    return None


def first_definitional_order(term, parts):
    esc = re.escape(term)
    pats = [re.compile(p.format(t=esc), re.I | re.M) for p in DEF_PATTERNS]
    for order, lid, text in parts:
        for pr in pats:
            if pr.search(text):
                return order
    return None


def detect(parts):
    text = "\n".join(t for _, _, t in parts)
    vocab = build_vocabulary(text)
    used_before, orphans = set(), set()
    for term in vocab:
        appear = first_appearance_order(term, parts)
        if appear is None:
            continue
        defo = first_definitional_order(term, parts)
        if defo is None:
            orphans.add(term)
        elif appear < defo:
            used_before.add(term)
    return used_before, orphans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    specs = json.loads((root / "injections.json").read_text(encoding="utf-8"))

    results = []
    for s in specs:
        parts = assemble_course(root, specs, s["id"], s["insert_lecture"], manifest)
        # write the full injected course for the LLM-direct baseline
        course_md = "\n\n".join(t for _, _, t in parts)
        (root / "injections" / s["id"] / "course.md").write_text(course_md, encoding="utf-8")
        if s["type"] == "placebo":
            results.append({**s, "kw_detected": None}); continue
        ub, orph = detect(parts)
        target = s["target"]
        if s["type"] == "used_before_introduction":
            hit = any(normalize(target) in normalize(t) or normalize(t) in normalize(target) for t in ub)
        else:
            hit = any(normalize(target) in normalize(t) or normalize(t) in normalize(target) for t in orph)
        results.append({**s, "kw_detected": bool(hit)})

    (root / "baseline_keyword_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    for r in results:
        if r["type"] == "placebo":
            continue
        c = by[r["type"]]
        c[0] += bool(r["kw_detected"]); c[1] += 1
    print(f"[{args.dataset}] keyword baseline recall:")
    for t, (d, n) in by.items():
        print(f"  {t:26} {d}/{n}")
    print(f"  (assembled {len(specs)} full-course files -> injections/*/course.md)")


if __name__ == "__main__":
    main()
