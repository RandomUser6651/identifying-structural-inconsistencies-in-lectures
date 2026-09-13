"""
Combine the recall of the graph method and the two baselines on the same
injections, per type and per course.

  graph        : from injection_results.json (the leak-free frozen-extraction score)
  llm_direct   : from injections/<id>/llm_direct.json (single-prompt, no graph)
  keyword      : from baseline_keyword_results.json (pattern matching, no LLM)

Usage: python pipeline-v2/score_baselines.py --dataset dataset-1
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


def match(target, names):
    t = norm(target)
    return any(t == norm(n) or t in norm(n) or norm(n) in t for n in (names or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset

    specs = json.loads((root / "injections.json").read_text(encoding="utf-8"))
    graph = {r["id"]: r for r in json.loads((root / "injection_results.json").read_text(encoding="utf-8"))}
    kw = {r["id"]: r for r in json.loads((root / "baseline_keyword_results.json").read_text(encoding="utf-8"))}

    rows = []
    for s in specs:
        if s["type"] == "placebo":
            continue
        sid = s["id"]
        # graph
        g_hit = bool(graph.get(sid, {}).get("detected"))
        # keyword
        k_hit = bool(kw.get(sid, {}).get("kw_detected"))
        # llm-direct
        ld_path = root / "injections" / sid / "llm_direct.json"
        if ld_path.exists():
            ld = json.loads(ld_path.read_text(encoding="utf-8"))
            lst = ld.get("used_before") if s["type"] == "used_before_introduction" else ld.get("orphans")
            l_hit = match(s["target"], lst)
        else:
            l_hit = None
        rows.append({"id": sid, "type": s["type"], "subtlety": s["subtlety"],
                     "target": s["target"], "graph": g_hit,
                     "llm_direct": l_hit, "keyword": k_hit})

    def tally(rows, method, typ):
        rs = [r for r in rows if r["type"] == typ]
        d = sum(1 for r in rs if r[method])
        return d, len(rs)

    print(f"\n=== [{args.dataset}] RECALL: graph vs baselines ===")
    print(f"{'type':28} {'graph':>8} {'llm_direct':>12} {'keyword':>10}")
    for typ in ("used_before_introduction", "orphan_concept"):
        g = tally(rows, "graph", typ); l = tally(rows, "llm_direct", typ); k = tally(rows, "keyword", typ)
        print(f"{typ:28} {f'{g[0]}/{g[1]}':>8} {f'{l[0]}/{l[1]}':>12} {f'{k[0]}/{k[1]}':>10}")

    (root / "baseline_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {root / 'baseline_comparison.json'}")


if __name__ == "__main__":
    main()
