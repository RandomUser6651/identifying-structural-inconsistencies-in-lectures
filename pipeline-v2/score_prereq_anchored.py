"""
Cheap validation of the fact-anchoring idea WITHOUT any new extraction.

The improved dedicated prerequisite edges already carry evidence_unit (the unit
where the dependency is actually stated). The current detector ignores it and
compares the two concepts' FIRST-INTRODUCTION positions. Here we re-score the
same edges under the fact-anchored rule:

    violation  <=>  order(prerequisite_introduced) > order(evidence_unit)

i.e. the prerequisite is introduced AFTER the point where it is needed, instead
of after the dependent concept's first mention. Reports, per course, the old vs
new prerequisite_order count on the CLEAN graph and lists which old violations
disappear (the within-lecture false positives such as Value Object/immutability).

Usage: python pipeline-v2/score_prereq_anchored.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entity_resolution as er
import build_graph as bg
import detect_violations as dv

RESULTS = HERE.parent / "new-results"


def unit_idx(u):
    m = re.search(r"#(\d+)", u or "")
    return int(m.group(1)) if m else 0


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
        f = root / "dedicated_edges" / f"{lid}.clean.consensus.json"
        return json.loads(f.read_text(encoding="utf-8"))["prerequisites"] if f.exists() else []

    # build improved clean graph to get canonical names + introduction orders
    lecs = [{"lecture_id": l, "concepts": frozen[l].get("concepts", []),
             "prerequisites": cc(l)} for l in frozen]
    occs, raw = er.collect_occurrences(lecs, order_of)
    clusters = er.cluster_occurrences(occs)
    nz = er.emit_normalized(args.dataset, lecs, clusters, raw)
    g = bg.build_graph(nz, units)
    intro = {n: (g.nodes[n].get("order_introduced") or None) for n in g.nodes}

    # canonical resolver by normalized name
    canon = {}
    for n in g.nodes:
        canon[er.normalize_name(n)] = n
        for a in (g.nodes[n].get("aliases") or []):
            canon.setdefault(er.normalize_name(a), n)

    def resolve(name):
        return canon.get(er.normalize_name(name))

    # iterate consensus edges per lecture (evidence_lecture = that lecture)
    old_v, new_v, disappear = set(), set(), []
    for lid in frozen:
        for e in cc(lid):
            a = resolve(e.get("from", "")); b = resolve(e.get("to", ""))
            if not a or not b or a == b:
                continue
            ai = intro.get(a); bi = intro.get(b)
            if not ai or not bi:
                continue
            ev = (order_of.get(lid, 0), unit_idx(e.get("evidence_unit", "")))
            old = tuple(ai) > tuple(bi)               # current rule
            new = tuple(ai) > ev                       # fact-anchored rule
            key = (a, b)
            if old:
                old_v.add(key)
            if new:
                new_v.add(key)
            if old and not new:
                disappear.append((a, b, tuple(bi), tuple(ai), ev))

    print(f"=== [{args.dataset}] prerequisite_order on clean graph: current vs fact-anchored ===")
    print(f"  current rule (prereq_intro > dependent_intro): {len(old_v)} direct-edge violations")
    print(f"  fact-anchored (prereq_intro > evidence_unit) : {len(new_v)} violations")
    print(f"  removed as within-statement false positives  : {len(disappear)}")
    for a, b, bi, ai, ev in disappear[:25]:
        print(f"    - «{a}» -> «{b}»: dep_intro={bi} prereq_intro={ai} evidence={ev}")


if __name__ == "__main__":
    main()
