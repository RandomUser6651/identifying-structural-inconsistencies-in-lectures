#!/usr/bin/env python3
"""Clean-course anchored prerequisite-order count (production rule): build the
graph from the dedicated-ensemble CONSENSUS edges and run the fact-anchored
detector (violation iff a prerequisite is introduced after the edge's
evidence_unit). No injections — just the clean-course report.

Usage: python score_prereq_anchored_clean.py --dataset dataset-2
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_graph as bg
from validate_prereq_anchored import build_report

ROOT = HERE.parent / "new-results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-2")
    args = ap.parse_args()
    root = ROOT / args.dataset
    order_of = {m["lecture_id"]: m["lecture_order"]
                for m in json.loads((root / "manifest.json").read_text(encoding="utf-8"))}
    units = bg.load_units(root)
    frozen = {}
    for f in sorted((root / "extraction").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        frozen[d["lecture_id"]] = d

    def consensus_edges(lid):
        p = root / "dedicated_edges" / f"{lid}.clean.consensus.json"
        return json.loads(p.read_text(encoding="utf-8"))["prerequisites"] if p.exists() else []

    clean_lecs = [{"lecture_id": l, "concepts": frozen[l].get("concepts", []),
                   "prerequisites": consensus_edges(l)} for l in frozen]
    report = build_report(clean_lecs, order_of, units, args.dataset)
    (root / "prereq_anchored.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    within = sum(1 for v in report if v.get("same_lecture"))
    same_unit = sum(1 for v in report
                    if v.get("prereq_introduced_order") == v.get("evidence_order"))
    direct = sum(1 for v in report if not v.get("transitive"))
    print(f"[{args.dataset}] anchored prerequisite_order: {len(report)} "
          f"(direct {direct}, within-lecture {within}, same-unit {same_unit} [must be 0])")
    for v in report[:12]:
        print(f"  {v['prerequisite']} -> {v['dependent']}  (prereq@{v.get('prereq_introduced_lecture')}, "
              f"needed@{v.get('evidence_lecture', v.get('evidence_unit'))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
