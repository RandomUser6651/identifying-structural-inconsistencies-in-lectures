"""
Minimal-injection control for prerequisite-ordering: isolate DETECTOR
correctness from extractor instability.

Instead of re-extracting the manipulated lecture (which perturbs many unrelated
prerequisite edges), this takes the FROZEN clean extraction of the manipulated
lecture and appends ONLY the injected prerequisite edge {from: A, to: C}
(A resolves globally to its real, later introduction; nothing else changes).
The graph then differs from the clean graph by exactly the injected edge and its
transitive consequences, so:

  * a real injection (A introduced later than C) must yield the (A -> C)
    ordering violation  -> recall;
  * a placebo (A' introduced earlier than C) must yield no new ordering
    violation -> a true zero-drift noise floor.

This complements score_prereq.py (end-to-end re-extraction): the gap between
the two is precisely the extractor-instability contribution.

Usage: python pipeline-v2/score_prereq_minimal.py --dataset dataset-1
"""

import argparse
import json
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


def prereq_sigs(report):
    return {f"{v['prerequisite']}→{v['dependent']}" for v in report["prerequisite_order"]}


def run(lectures, order_of, units, dataset):
    occs, raw = er.collect_occurrences(lectures, order_of)
    clusters = er.cluster_occurrences(occs)
    normalized = er.emit_normalized(dataset, lectures, clusters, raw)
    g = bg.build_graph(normalized, units)
    return dv.run_detectors(g, use_embeddings=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    units = bg.load_units(root)

    frozen = {}
    for f in sorted((root / "extraction").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        frozen[d["lecture_id"]] = d

    clean = run(list(frozen.values()), order_of, units, args.dataset)
    clean_sigs = prereq_sigs(clean)

    specs = json.loads((root / "prereq_specs.json").read_text(encoding="utf-8"))
    results = []
    for s in specs:
        lec = s["insert_lecture"]
        base = json.loads(json.dumps(frozen[lec]))  # deep copy
        base.setdefault("prerequisites", []).append({
            "from": s["prerequisite"], "to": s["dependent"],
            "reason": "injected dependency (minimal control)",
            "evidence_unit": s.get("insert_unit", ""),
        })
        lectures = [v for k, v in frozen.items() if k != lec] + [base]
        report = run(lectures, order_of, units, args.dataset)
        new = prereq_sigs(report) - clean_sigs
        detected = False
        if s["expected"]:
            for pair in new:
                p, d = pair.split("→", 1)
                if normsub(s["prerequisite"], p) and normsub(s["dependent"], d):
                    detected = True; break
        results.append({**s, "detected": detected, "n_new": len(new)})

    by = defaultdict(lambda: [0, 0])
    for r in results:
        if r["type"] == "placebo":
            continue
        c = by[r["subtlety"]]; c[0] += r["detected"]; c[1] += 1
    print(f"=== [{args.dataset}] MINIMAL-INJECTION (detector correctness, zero extractor drift) ===")
    tot = [0, 0]
    for tier in ("blatant", "medium", "subtle"):
        d, n = by[tier]; tot[0] += d; tot[1] += n
        print(f"  {tier:8} {d}/{n}")
    print(f"  TOTAL recall {tot[0]}/{tot[1]}")
    placebos = [r for r in results if r["type"] == "placebo"]
    print(f"  placebo new ordering sigs: {sum(r['n_new'] for r in placebos)} (must be 0)")
    (root / "prereq_minimal_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
