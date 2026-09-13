"""
Prepare definition-conflict scoring: for each conflict injection, assemble the
frozen-plus-one extraction set, run entity resolution and graph construction
in-process, locate the target concept, and emit its (now two) definitions as a
verification task. The clean baseline target has a single definition, so the
injected definition makes it a genuine candidate; whether the candidate is a
real conflict is decided by the verification workflow.

Writes new-results/<dataset>/conflict_verify_tasks.json with, per injection:
  {id, kind, concept, subtlety, is_candidate, definitions:[{lecture, text}]}

Usage: python pipeline-v2/score_conflicts_prepare.py --dataset dataset-1
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entity_resolution as er
import build_graph as bg

RESULTS = HERE.parent / "new-results"


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

    specs = json.loads((root / "conflict_specs.json").read_text(encoding="utf-8"))
    tasks = []
    for s in specs:
        lec = s["insert_lecture"]
        inj = root / "injections" / s["id"] / f"{lec}.extraction.json"
        if not inj.exists():
            print(f"  MISSING re-extraction: {s['id']}"); continue
        injected = json.loads(inj.read_text(encoding="utf-8"))
        lectures = [v for k, v in frozen.items() if k != lec] + [injected]
        occs, raw = er.collect_occurrences(lectures, order_of)
        clusters = er.cluster_occurrences(occs)
        normalized = er.emit_normalized(args.dataset, lectures, clusters, raw)
        # find the target concept cluster
        target = er.normalize_name(s["concept"])
        node = None
        for c in normalized["concepts"]:
            if er.normalize_name(c["canonical_name"]) == target:
                node = c; break
        if node is None:
            # fallback: substring
            for c in normalized["concepts"]:
                if target in er.normalize_name(c["canonical_name"]):
                    node = c; break
        defs = (node or {}).get("definitions") or []
        distinct_lec = {d["lecture"] for d in defs}
        tasks.append({
            "id": s["id"], "kind": s["kind"], "concept": s["concept"],
            "subtlety": s["subtlety"],
            "is_candidate": len(defs) >= 2 and len(distinct_lec) >= 2,
            "n_defs": len(defs),
            "definitions": [{"lecture": d["lecture"], "text": d["text"]} for d in defs],
        })

    (root / "conflict_verify_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cand = sum(1 for t in tasks if t["is_candidate"])
    n_conf = sum(1 for t in tasks if t["kind"] == "conflict")
    n_conf_cand = sum(1 for t in tasks if t["kind"] == "conflict" and t["is_candidate"])
    print(f"[{args.dataset}] {len(tasks)} tasks, {n_cand} are candidates "
          f"(conflict injections that became candidates: {n_conf_cand}/{n_conf})")
    for t in tasks:
        if not t["is_candidate"]:
            print(f"  NOT candidate: {t['id']} ({t['kind']}) {t['concept']} — n_defs={t['n_defs']}")


if __name__ == "__main__":
    main()
