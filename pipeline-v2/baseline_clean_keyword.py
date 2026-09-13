"""
Run the keyword/pattern baseline on the CLEAN (non-injected) course and emit its
structural-issue detections, for a precision comparison against the graph method
on naturally occurring issues.

Two detector types (the only ones surface patterns support):
  used_before_introduction : a vocabulary term appears in a lecture earlier than
    the earliest lecture in which it appears in a definitional pattern;
  orphan_concept           : a vocabulary term appears but never in a
    definitional pattern.

Writes new-results/<ds>/keyword_clean.json: a list of
  {type, concept, first_lecture, first_unit, def_lecture}
and prints the raw false-alarm volume per type (and the graph's, for reference).

Usage: python pipeline-v2/baseline_clean_keyword.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_keyword as bk

RESULTS = HERE.parent / "new-results"


def first_unit(term, root, manifest):
    tl = term.lower()
    for m in sorted(manifest, key=lambda x: x["lecture_order"]):
        lid = m["lecture_id"]
        units = json.loads((root / "units" / f"{lid}.json").read_text(encoding="utf-8"))
        for u in units["units"]:
            if tl in u["text"].lower():
                return lid, u["unit_id"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    # assemble the clean course (rendered lectures in order)
    parts = []
    for m in sorted(manifest, key=lambda x: x["lecture_order"]):
        lid = m["lecture_id"]
        parts.append((m["lecture_order"], lid,
                      (root / "rendered" / f"{lid}.md").read_text(encoding="utf-8")))

    ub, orph = bk.detect(parts)

    dets = []
    for term in sorted(ub):
        fl, fu = first_unit(term, root, manifest)
        defo = bk.first_definitional_order(term, parts)
        dets.append({"type": "used_before_introduction", "concept": term,
                     "first_lecture": fl, "first_unit": fu, "def_order": defo})
    for term in sorted(orph):
        fl, fu = first_unit(term, root, manifest)
        dets.append({"type": "orphan_concept", "concept": term,
                     "first_lecture": fl, "first_unit": fu, "def_order": None})

    (root / "keyword_clean.json").write_text(
        json.dumps(dets, ensure_ascii=False, indent=1), encoding="utf-8")

    n_ub = sum(1 for d in dets if d["type"] == "used_before_introduction")
    n_or = sum(1 for d in dets if d["type"] == "orphan_concept")
    print(f"[{args.dataset}] KEYWORD clean detections: {len(dets)} "
          f"(used_before={n_ub}, orphan={n_or})")

    # graph reference counts (improved prereq if available)
    viol = json.loads((root / "violations.json").read_text(encoding="utf-8"))
    g_ub = len(viol["used_before_introduction"])
    g_or = len(viol["orphan_concepts"])
    g_cf = len(viol["definition_conflicts"])
    imp = root / "prereq_improved_results.json"
    print(f"[{args.dataset}] GRAPH clean detections (for reference): "
          f"used_before={g_ub}, orphan={g_or}, conflict_candidates={g_cf}, "
          f"prereq_order={viol['counts']['prerequisite_order']} (general)")


if __name__ == "__main__":
    main()
