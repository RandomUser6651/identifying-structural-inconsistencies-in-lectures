"""
Score the IMPROVED prerequisite-order pipeline: dedicated, concept-grounded,
ensemble (unanimous K-of-K) prerequisite-edge extraction, under the same
leak-free frozen protocol.

Nodes/concepts come from the frozen concept extraction (unchanged); only the
EDGES are replaced by the consensus dedicated edges. Clean baseline uses the
clean consensus edges for every lecture; each injection swaps its manipulated
lecture to the injected consensus edges. New prerequisite_order signatures are
the set-difference against the clean baseline.

Reports recall (injected pair detected), placebo drift (noise floor), and the
clean-course prerequisite_order count (a precision proxy) — all directly
comparable to the single-pass general baseline in score_prereq.py.

Usage: python pipeline-v2/score_prereq_improved.py --dataset dataset-3
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
    ap.add_argument("--dataset", default="dataset-3")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    units = bg.load_units(root)
    ded = root / "dedicated_edges"

    # frozen concepts per lecture
    frozen = {}
    for f in sorted((root / "extraction").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        frozen[d["lecture_id"]] = d

    def clean_consensus(lid):
        f = ded / f"{lid}.clean.consensus.json"
        return json.loads(f.read_text(encoding="utf-8"))["prerequisites"] if f.exists() else []

    def lecture_dict(lid, edges):
        return {"lecture_id": lid, "concepts": frozen[lid].get("concepts", []),
                "prerequisites": edges}

    # clean baseline: dedicated consensus edges for every lecture
    clean_lectures = [lecture_dict(lid, clean_consensus(lid)) for lid in frozen]
    clean_report = run(clean_lectures, order_of, units, args.dataset)
    clean_sigs = prereq_sigs(clean_report)
    print(f"[{args.dataset}] IMPROVED clean prerequisite_order count = "
          f"{clean_report['counts']['prerequisite_order']}")

    specs = json.loads((root / "prereq_specs.json").read_text(encoding="utf-8"))
    results = []
    for s in specs:
        lid = s["insert_lecture"]; sid = s["id"]
        inj_f = root / "injections" / sid / f"{lid}.dedicated.consensus.json"
        if not inj_f.exists():
            print(f"  MISSING injected consensus: {sid}"); continue
        inj_edges = json.loads(inj_f.read_text(encoding="utf-8"))["prerequisites"]
        lectures = [lecture_dict(l, clean_consensus(l)) for l in frozen if l != lid]
        lectures.append(lecture_dict(lid, inj_edges))
        report = run(lectures, order_of, units, args.dataset)
        new = prereq_sigs(report) - clean_sigs
        detected = False
        if s["expected"]:
            for pair in new:
                p, d = pair.split("→", 1)
                if normsub(s["prerequisite"], p) and normsub(s["dependent"], d):
                    detected = True; break
        results.append({**s, "detected": detected, "n_new": len(new),
                        "new_sigs": sorted(new)})

    by = defaultdict(lambda: [0, 0])
    for r in results:
        if r["type"] == "placebo":
            continue
        c = by[r["subtlety"]]; c[0] += r["detected"]; c[1] += 1
    print("\n=== IMPROVED RECALL prerequisite_order ===")
    tot = [0, 0]
    for tier in ("blatant", "medium", "subtle"):
        d, n = by[tier]; tot[0] += d; tot[1] += n
        print(f"  {tier:8} {d}/{n}")
    print(f"  TOTAL {tot[0]}/{tot[1]}")
    for r in results:
        if r["type"] != "placebo" and not r["detected"]:
            print(f"    MISS {r['id']} ({r['subtlety']}): «{r['prerequisite']}»→«{r['dependent']}»")

    placebos = [r for r in results if r["type"] == "placebo"]
    pl = sum(r["n_new"] for r in placebos)
    print(f"\n=== PLACEBO drift: {pl} new ordering sigs across {len(placebos)} placebos "
          f"(≈{pl/max(1,len(placebos)):.1f}/lecture; general baseline was ≈3) ===")
    for r in placebos:
        if r["n_new"]:
            print(f"    {r['id']}: {r['new_sigs']}")

    (root / "prereq_improved_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {root / 'prereq_improved_results.json'}")


if __name__ == "__main__":
    main()
