"""
Score the injection-recall experiment (leak-free, frozen-extraction).

For each injection we assemble the full per-lecture extraction set as:
  * the frozen clean extraction for every UNtouched lecture, plus
  * the re-extracted injected lecture (injections/<id>/<lecture>.extraction.json)
then run entity resolution -> graph -> detectors in-process, and compute the
NEW violations as a signature set-difference against the clean baseline (which
is itself built from the same in-process path over the frozen extractions).
A pre-existing issue has an identical signature in both runs and cancels, so
extraction noise in untouched lectures cannot leak in.

Recall = fraction of injections whose expected (type, concept) appears among
the new violations. Placebo injections (neutral edits) should produce ~0 new
matching violations — that is the noise floor.

Usage: python pipeline-v2/score_injections.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse

import entity_resolution as er
import build_graph as bg
import detect_violations as dv

RESULTS = HERE.parent / "new-results"


def normsub(a: str, b: str) -> bool:
    a, b = er.normalize_name(a), er.normalize_name(b)
    return a == b or a in b or b in a


def signatures(report):
    """Set of (type, subtype, concept) signatures. Orphans carry their subtype
    so that injected used-without-introduction orphans are not conflated with
    isolated-singleton drift (which we do not inject and which is sensitive to
    prerequisite-edge instability across re-extractions)."""
    sigs = set()
    for v in report["used_before_introduction"]:
        sigs.add(("used_before_introduction", "", v["concept"]))
    for v in report["orphan_concepts"]:
        sigs.add(("orphan_concept", v.get("subtype", ""), v["concept"]))
    return sigs


# Signature types we actually inject (everything else is reported separately
# as drift, not as a missed/false detection).
INJECTED_SIG_TYPES = {
    ("used_before_introduction", ""),
    ("orphan_concept", "used_without_introduction"),
}


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

    # frozen clean extractions
    frozen = {}
    for f in sorted((ROOT / "extraction").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        frozen[d["lecture_id"]] = d

    clean_report = run_pipeline(list(frozen.values()), order_of, units, args.dataset)
    clean_sigs = signatures(clean_report)
    print(f"clean baseline: used_before={clean_report['counts']['used_before_introduction']} "
          f"orphans={clean_report['counts']['orphan_concepts']}")

    specs = json.loads((ROOT / "injections.json").read_text(encoding="utf-8"))
    results = []
    for s in specs:
        lec = s["insert_lecture"]
        inj_path = ROOT / "injections" / s["id"] / f"{lec}.extraction.json"
        if not inj_path.exists():
            print(f"  MISSING re-extraction: {s['id']}"); continue
        injected = json.loads(inj_path.read_text(encoding="utf-8"))
        lectures = [v for k, v in frozen.items() if k != lec] + [injected]
        report = run_pipeline(lectures, order_of, units, args.dataset)
        new_sigs = signatures(report) - clean_sigs

        # new signatures restricted to the types we actually inject
        new_inj = {(t, st, c) for (t, st, c) in new_sigs
                   if (t, st) in INJECTED_SIG_TYPES}
        detected = False
        matched = None
        if s["expected"]:
            et, ec = s["expected"]["type"], s["expected"]["concept"]
            for (t, st, c) in new_inj:
                if t == et and normsub(ec, c):
                    detected = True; matched = c; break
        results.append({**s, "detected": detected, "matched": matched,
                        "n_new_injected_types": len(new_inj),
                        "n_new_all": len(new_sigs),
                        "new_sigs": sorted(f"{t}/{st}:{c}" for t, st, c in new_inj)})

    # report
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in results:
        if r["type"] == "placebo":
            continue
        cell = by[r["type"]][r["subtlety"]]
        cell[0] += r["detected"]; cell[1] += 1
    print("\n=== RECALL за типом і тонкістю ===")
    for t in ("used_before_introduction", "orphan_concept"):
        tot_d = tot_n = 0
        line = []
        for tier in ("blatant", "medium", "subtle"):
            d, n = by[t][tier]
            tot_d += d; tot_n += n
            line.append(f"{tier}={d}/{n}")
        print(f"  {t:26} " + "  ".join(line) + f"   | разом {tot_d}/{tot_n}")

    placebos = [r for r in results if r["type"] == "placebo"]
    pl_inj = sum(r["n_new_injected_types"] for r in placebos)
    pl_all = sum(r["n_new_all"] for r in placebos)
    print(f"\n=== PLACEBO (шум) ===")
    print(f"  {len(placebos)} placebo — нові сигнатури інжектованих типів: {pl_inj} "
          f"(має бути ~0); усіх типів (вкл. singleton-drift): {pl_all}")
    for r in placebos:
        if r["n_new_injected_types"]:
            print(f"    {r['id']} ({r['target']}): {r['new_sigs']}")

    (ROOT / "injection_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {ROOT / 'injection_results.json'}")


if __name__ == "__main__":
    main()
