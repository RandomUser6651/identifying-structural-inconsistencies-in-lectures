"""
Final prerequisite-order validation: dedicated-ensemble edges + fact-anchoring
(violation iff prereq introduced after the edge's evidence_unit). Reports, per
course: clean-graph count, injection recall (leak-free set-difference) by
subtlety, placebo drift, and the same-unit floor (must be 0 by construction).

Usage: python pipeline-v2/validate_prereq_anchored.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entity_resolution as er
import build_graph as bg
import detect_violations as dv

RESULTS = HERE.parent / "new-results"


def normsub(a, b):
    a, b = er.normalize_name(a), er.normalize_name(b)
    return a == b or a in b or b in a


def build_report(lectures, order_of, units, dataset):
    occs, raw = er.collect_occurrences(lectures, order_of)
    clusters = er.cluster_occurrences(occs)
    nz = er.emit_normalized(dataset, lectures, clusters, raw)
    g = bg.build_graph(nz, units)
    return dv.detect_prerequisite_order_anchored(g, order_of)


def sigset(report):
    return {(v["prerequisite"], v["dependent"]) for v in report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    order_of = {m["lecture_id"]: m["lecture_order"]
                for m in json.loads((root / "manifest.json").read_text(encoding="utf-8"))}
    units = bg.load_units(root)
    frozen = {}
    for f in sorted((root / "extraction").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        frozen[d["lecture_id"]] = d

    def cc(lid):
        p = root / "dedicated_edges" / f"{lid}.clean.consensus.json"
        return json.loads(p.read_text(encoding="utf-8"))["prerequisites"] if p.exists() else []

    clean_lecs = [{"lecture_id": l, "concepts": frozen[l].get("concepts", []),
                   "prerequisites": cc(l)} for l in frozen]
    clean_report = build_report(clean_lecs, order_of, units, args.dataset)
    clean_sigs = sigset(clean_report)
    same_unit_clean = sum(1 for v in clean_report
                          if v["prereq_introduced_order"] == v["evidence_order"])
    print(f"=== [{args.dataset}] anchored prerequisite_order ===")
    print(f"  clean-graph violations: {len(clean_report)} "
          f"(within-lecture {sum(1 for v in clean_report if v['same_lecture'])}, "
          f"same-unit {same_unit_clean} [must be 0])")

    specs = json.loads((root / "prereq_specs.json").read_text(encoding="utf-8"))
    results = []
    for s in specs:
        lec = s["insert_lecture"]; sid = s["id"]
        inj = root / "injections" / sid / f"{lec}.dedicated.consensus.json"
        if not inj.exists():
            print(f"  MISSING injected consensus: {sid}"); continue
        inj_edges = json.loads(inj.read_text(encoding="utf-8"))["prerequisites"]
        lecs = [{"lecture_id": l, "concepts": frozen[l].get("concepts", []),
                 "prerequisites": cc(l)} for l in frozen if l != lec]
        lecs.append({"lecture_id": lec, "concepts": frozen[lec].get("concepts", []),
                     "prerequisites": inj_edges})
        new = sigset(build_report(lecs, order_of, units, args.dataset)) - clean_sigs
        detected = False
        if s["expected"]:
            for (p, d) in new:
                if normsub(s["prerequisite"], p) and normsub(s["dependent"], d):
                    detected = True; break
        results.append({**s, "detected": detected, "n_new": len(new)})

    by = defaultdict(lambda: [0, 0])
    for r in results:
        if r["type"] == "placebo":
            continue
        c = by[r["subtlety"]]; c[0] += r["detected"]; c[1] += 1
    tot = [0, 0]
    line = []
    for tier in ("blatant", "medium", "subtle"):
        d, n = by[tier]; tot[0] += d; tot[1] += n
        line.append(f"{tier}={d}/{n}")
    print(f"  injection recall: " + "  ".join(line) + f"  | TOTAL {tot[0]}/{tot[1]}")
    for r in results:
        if r["type"] != "placebo" and not r["detected"]:
            print(f"    MISS {r['id']} ({r['subtlety']}): «{r['prerequisite']}»->«{r['dependent']}»")
    placebos = [r for r in results if r["type"] == "placebo"]
    pl = sum(r["n_new"] for r in placebos)
    print(f"  placebo drift: {pl} across {len(placebos)} (≈{pl/max(1,len(placebos)):.1f}/lecture)")


if __name__ == "__main__":
    main()
