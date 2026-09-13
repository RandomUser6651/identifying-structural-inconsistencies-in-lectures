"""
Score the prerequisite-ordering injection-recall experiment (leak-free,
frozen-extraction), mirroring score_injections.py but for the
prerequisite_order detector.

For each injection: freeze every untouched lecture's clean extraction, swap in
the re-extracted injected lecture, run ER -> graph -> detectors in-process, and
take the NEW prerequisite_order signatures as a set-difference against the clean
baseline. Recall = the expected (prerequisite -> dependent) pair appears among
the new signatures. Placebos (valid prerequisites) should add ~0.

Usage: python pipeline-v2/score_prereq.py --dataset dataset-1
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


def normsub(a: str, b: str) -> bool:
    a, b = er.normalize_name(a), er.normalize_name(b)
    return a == b or a in b or b in a


def prereq_signatures(report):
    """Set of ('prerequisite_order', '', 'PREREQ→DEPENDENT') signatures."""
    sigs = set()
    for v in report["prerequisite_order"]:
        sigs.add(("prerequisite_order", "", f"{v['prerequisite']}→{v['dependent']}"))
    return sigs


def run_pipeline(lectures, order_of, units, dataset):
    occs, raw = er.collect_occurrences(lectures, order_of)
    clusters = er.cluster_occurrences(occs)
    normalized = er.emit_normalized(dataset, lectures, clusters, raw)
    g = bg.build_graph(normalized, units)
    return dv.run_detectors(g, use_embeddings=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    ROOT = RESULTS / args.dataset
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    units = bg.load_units(ROOT)

    frozen = {}
    for f in sorted((ROOT / "extraction").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        frozen[d["lecture_id"]] = d

    clean_report = run_pipeline(list(frozen.values()), order_of, units, args.dataset)
    clean_sigs = prereq_signatures(clean_report)
    print(f"clean baseline: prerequisite_order={clean_report['counts']['prerequisite_order']}")

    specs = json.loads((ROOT / "prereq_specs.json").read_text(encoding="utf-8"))
    results = []
    for s in specs:
        lec = s["insert_lecture"]
        inj_path = ROOT / "injections" / s["id"] / f"{lec}.extraction.json"
        if not inj_path.exists():
            print(f"  MISSING re-extraction: {s['id']}"); continue
        injected = json.loads(inj_path.read_text(encoding="utf-8"))
        lectures = [v for k, v in frozen.items() if k != lec] + [injected]
        report = run_pipeline(lectures, order_of, units, args.dataset)
        new_sigs = prereq_signatures(report) - clean_sigs

        detected, matched = False, None
        if s["expected"]:
            ep, ed = s["expected"]["prerequisite"], s["expected"]["dependent"]
            for (_, _, pair) in new_sigs:
                p, d = pair.split("→", 1)
                if normsub(ep, p) and normsub(ed, d):
                    detected, matched = True, pair; break
        # did the injected DIRECT edge even appear in the graph? (diagnostic)
        edge_present = False
        for e in report["prerequisite_order"]:
            if s["expected"] and normsub(s["expected"]["prerequisite"], e["prerequisite"]) \
               and normsub(s["expected"]["dependent"], e["dependent"]) and not e.get("transitive"):
                edge_present = True; break
        results.append({**s, "detected": detected, "matched": matched,
                        "direct_edge": edge_present,
                        "n_new_prereq_order": len(new_sigs),
                        "new_sigs": sorted(pair for (_, _, pair) in new_sigs)})

    by = defaultdict(lambda: [0, 0])
    for r in results:
        if r["type"] == "placebo":
            continue
        cell = by[r["subtlety"]]
        cell[0] += r["detected"]; cell[1] += 1
    print("\n=== RECALL prerequisite_order за тонкістю ===")
    tot_d = tot_n = 0
    line = []
    for tier in ("blatant", "medium", "subtle"):
        d, n = by[tier]
        tot_d += d; tot_n += n
        line.append(f"{tier}={d}/{n}")
    print("  " + "  ".join(line) + f"   | разом {tot_d}/{tot_n}")
    # show misses with diagnostic
    for r in results:
        if r["type"] != "placebo" and not r["detected"]:
            print(f"    MISS {r['id']} ({r['subtlety']}): «{r['prerequisite']}»→«{r['dependent']}» "
                  f"direct_edge={r['direct_edge']} new={r['new_sigs'][:4]}")

    placebos = [r for r in results if r["type"] == "placebo"]
    pl = sum(r["n_new_prereq_order"] for r in placebos)
    print(f"\n=== PLACEBO (шум) ===")
    print(f"  {len(placebos)} placebo — нові prerequisite_order сигнатури: {pl} (має бути ~0)")
    for r in placebos:
        if r["n_new_prereq_order"]:
            print(f"    {r['id']} («{r['prerequisite']}»→«{r['dependent']}»): {r['new_sigs']}")

    (ROOT / "prereq_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {ROOT / 'prereq_results.json'}")


if __name__ == "__main__":
    main()
