"""
Score the direct-model baseline for prerequisite-ordering injections against
the graph method, per course. Reads, per injection, the direct-model output at
injections/<id>/llm_direct_prereq.json with shape:
  {"id": "...", "violations": [{"dependent": "...", "prerequisite": "..."}, ...]}
A hit = the injected (prerequisite -> dependent) pair appears among the
direct-model violations (substring-tolerant match on both endpoints).

Usage: python pipeline-v2/score_prereq_baseline.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def sub(a, b):
    a, b = norm(a), norm(b)
    return a == b or a in b or b in a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    specs = json.loads((root / "prereq_specs.json").read_text(encoding="utf-8"))
    graph = {r["id"]: r for r in json.loads((root / "prereq_results.json").read_text(encoding="utf-8"))}

    rows = []
    for s in specs:
        if s["type"] == "placebo":
            continue
        sid = s["id"]
        g_hit = bool(graph.get(sid, {}).get("detected"))
        ld_path = root / "injections" / sid / "llm_direct_prereq.json"
        l_hit = None
        if ld_path.exists():
            viol = json.loads(ld_path.read_text(encoding="utf-8")).get("violations", [])
            l_hit = any(sub(s["prerequisite"], v.get("prerequisite", "")) and
                        sub(s["dependent"], v.get("dependent", "")) for v in viol)
        rows.append({"id": sid, "subtlety": s["subtlety"],
                     "prerequisite": s["prerequisite"], "dependent": s["dependent"],
                     "graph": g_hit, "llm_direct": l_hit})

    g_d = sum(1 for r in rows if r["graph"]); n = len(rows)
    l_d = sum(1 for r in rows if r["llm_direct"])
    print(f"=== [{args.dataset}] prerequisite_order: graph vs direct model ===")
    print(f"  graph={g_d}/{n}   llm_direct={l_d}/{n}")
    by = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        c = by[r["subtlety"]]
        c[0] += r["graph"]; c[1] += bool(r["llm_direct"]); c[2] += 1
    for tier in ("blatant", "medium", "subtle"):
        d = by[tier]
        print(f"    {tier:8} graph={d[0]}/{d[2]}  llm_direct={d[1]}/{d[2]}")

    (root / "prereq_baseline.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {root / 'prereq_baseline.json'}")


if __name__ == "__main__":
    main()
