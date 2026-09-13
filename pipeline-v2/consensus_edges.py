"""
Ensemble (unanimous K-of-K, configurable) consensus of the dedicated
prerequisite-edge runs.

For each lecture text (a clean lecture or an injected lecture), read the K
independent edge-extraction runs and keep an edge (identified by the normalized
(from, to) pair) only if it appears in at least MIN_VOTES of the runs. The
production setting is unanimous (MIN_VOTES = K = 3), as reported in the article:
an edge enters the graph only if all three runs agree. This filters run-to-run
drift (a spurious edge seen in a single run is dropped) while preserving edges
that are consistently extractable. (On the current runs majority ≥2-of-3 and
unanimous 3-of-3 yield identical edge sets; unanimous is kept as the documented,
more conservative rule.)

Writes:
  dedicated_edges/<lecture>.clean.consensus.json
  injections/<id>/<lecture>.dedicated.consensus.json
each {lecture_id, prerequisites:[...], n_runs, votes_required}

Usage: python pipeline-v2/consensus_edges.py --dataset dataset-3 --min-votes 3
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"


def clean_name(s):
    """Strip a trailing '(alias...)' parenthetical the extractor may have copied
    from the vocabulary line (e.g. 'Національна рамка кваліфікацій (НРК)' ->
    'Національна рамка кваліфікацій'), so the endpoint resolves to its node."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", (s or "")).strip()


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s'-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def vote(run_files, min_votes):
    runs = []
    for f in run_files:
        if f.exists():
            try:
                runs.append(json.loads(f.read_text(encoding="utf-8")).get("prerequisites", []))
            except Exception:
                pass
    n = len(runs)
    counts = defaultdict(int)
    rep = {}
    for edges in runs:
        seen = set()
        for e in edges:
            fr, to = clean_name(e.get("from", "")), clean_name(e.get("to", ""))
            if not fr or not to:
                continue
            e = {**e, "from": fr, "to": to}
            key = (norm(fr), norm(to))
            if key in seen:
                continue          # one vote per run
            seen.add(key)
            counts[key] += 1
            rep.setdefault(key, e)
    kept = [rep[k] for k, c in counts.items() if c >= min_votes]
    return kept, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-3")
    ap.add_argument("--min-votes", type=int, default=3)  # unanimous K-of-K (production)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    root = RESULTS / args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    # clean lectures
    ded = root / "dedicated_edges"
    n_clean = 0
    for m in manifest:
        lid = m["lecture_id"]
        files = [ded / f"{lid}.clean.run{k}.json" for k in range(1, args.k + 1)]
        if not any(f.exists() for f in files):
            continue
        kept, n = vote(files, args.min_votes)
        (ded / f"{lid}.clean.consensus.json").write_text(
            json.dumps({"lecture_id": lid, "prerequisites": kept,
                        "n_runs": n, "votes_required": args.min_votes},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        n_clean += 1

    # injected lectures
    specs = json.loads((root / "prereq_specs.json").read_text(encoding="utf-8"))
    n_inj = 0
    for s in specs:
        lid = s["insert_lecture"]; sid = s["id"]
        d = root / "injections" / sid
        files = [d / f"{lid}.dedicated.run{k}.json" for k in range(1, args.k + 1)]
        if not any(f.exists() for f in files):
            print(f"  MISSING runs: {sid}"); continue
        kept, n = vote(files, args.min_votes)
        (d / f"{lid}.dedicated.consensus.json").write_text(
            json.dumps({"lecture_id": lid, "prerequisites": kept,
                        "n_runs": n, "votes_required": args.min_votes},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        n_inj += 1

    print(f"[{args.dataset}] consensus (>= {args.min_votes} of {args.k}): "
          f"{n_clean} clean lectures, {n_inj} injected lectures")


if __name__ == "__main__":
    main()
