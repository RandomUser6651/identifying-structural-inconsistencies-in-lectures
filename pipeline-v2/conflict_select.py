"""
Select targets for definition-conflict injections.

A target is a concept that the clean course introduces with exactly one
definition (so it is NOT already a conflict candidate), is structurally central
(high PageRank), and appears in — or can be inserted into — a lecture other than
the one that introduces it. Into that other lecture a CONTRADICTORY definition of
the concept will later be inserted, so the concept gains a second, incompatible
definition and becomes a genuine conflict.

Writes new-results/<dataset>/conflict_specs.json with, per target:
  id, concept, original_definition, original_lecture, insert_lecture, subtlety
(the contradictory sentence is authored in a later step).

Usage: python pipeline-v2/conflict_select.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"
SUBTLETY = ["blatant", "medium", "subtle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    ap.add_argument("--n", type=int, default=9)
    args = ap.parse_args()
    root = RESULTS / args.dataset
    graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    content_ids = [m["lecture_id"] for m in sorted(manifest, key=lambda x: x["lecture_order"])]

    # importance ranking
    import networkx as nx
    g = nx.readwrite.json_graph.node_link_graph(graph, directed=True, multigraph=False, edges="edges")
    try:
        pr = nx.pagerank(g, alpha=0.85)
    except Exception:
        pr = {n: 1.0 for n in g.nodes}

    cands = []
    for name, nd in nodes.items():
        defs = nd.get("definitions") or []
        if len(defs) != 1:
            continue                      # already 0 or 2+ defs -> skip
        d0 = defs[0]
        if not d0.get("text"):
            continue
        intro_lec = nd.get("lecture_introduced")
        if not intro_lec:
            continue
        # an insert lecture: a lecture where the concept already appears (used),
        # other than its introduction lecture; else any other content lecture
        appears_in = {a["lecture"] for a in (nd.get("appearances") or [])}
        other = [l for l in appears_in if l != intro_lec]
        if other:
            insert = sorted(other, key=lambda l: order_of.get(l, 0))[0]
        else:
            alt = [l for l in content_ids if l != intro_lec]
            insert = alt[len(alt) // 2] if alt else None
        if not insert:
            continue
        cands.append((pr.get(name, 0.0), name, d0["text"], intro_lec, insert))

    cands.sort(reverse=True)
    specs, idx = [], 0
    for k, (_, name, deftext, intro_lec, insert) in enumerate(cands[:args.n]):
        idx += 1
        specs.append({"id": f"cnf-inj-{idx:03d}", "kind": "conflict",
                      "concept": name, "original_definition": deftext,
                      "original_lecture": intro_lec, "insert_lecture": insert,
                      "subtlety": SUBTLETY[k % 3]})
    # placebos: 3 further concepts; a COMPATIBLE paraphrase will be injected,
    # which must NOT be flagged as a genuine conflict (false-conflict floor)
    for k, (_, name, deftext, intro_lec, insert) in enumerate(cands[args.n:args.n + 3]):
        idx += 1
        specs.append({"id": f"cnf-plc-{idx:03d}", "kind": "placebo",
                      "concept": name, "original_definition": deftext,
                      "original_lecture": intro_lec, "insert_lecture": insert,
                      "subtlety": "n/a"})

    (root / "conflict_specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.dataset}] selected {len(specs)} conflict targets:")
    for s in specs:
        print(f"  {s['id']} {s['subtlety']:8} {s['concept'][:28]:28} "
              f"def@{s['original_lecture']} -> inject@{s['insert_lecture']}")


if __name__ == "__main__":
    main()
