"""
Apply curated extraction-audit decisions (cross-language merges + non-discipline
excludes) to er_overrides.json, then re-run entity resolution -> graph ->
detectors for a dataset and report the orphan delta (clean-course false-positive
surface) before vs after.

Decisions file (JSON):
  {
    "merge_groups": [["сутність", "Entity"], ["подія", "Domain Event"], ...],
    "exclude":      ["модуль", "handler", ...]
  }

Usage:
  python pipeline-v2/apply_audit.py --decisions pipeline-v2/audit_decisions.json --dataset dataset-1
  python pipeline-v2/apply_audit.py --decisions ... --dry-run    # report only, no write
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE.parent / "new-results"
OVERRIDES = HERE / "er_overrides.json"


def current_orphans(dataset: str) -> set[str]:
    p = RESULTS / dataset / "violations.json"
    if not p.exists():
        return set()
    v = json.loads(p.read_text(encoding="utf-8"))
    return {o["concept"] for o in v.get("orphan_concepts", [])}


def merge_into_overrides(decisions: dict) -> dict:
    o = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    existing = {tuple(sorted(g)) for g in o.get("merge_groups", [])}
    for g in decisions.get("merge_groups", []):
        if tuple(sorted(g)) not in existing:
            o.setdefault("merge_groups", []).append(g)
            existing.add(tuple(sorted(g)))
    excl = set(o.get("exclude", []))
    for n in decisions.get("exclude", []):
        excl.add(n)
    if excl:
        o["exclude"] = sorted(excl)
    return o


def rerun(dataset: str) -> None:
    for script in ("entity_resolution.py", "build_graph.py", "detect_violations.py"):
        subprocess.run([sys.executable, str(HERE / script), "--dataset", dataset],
                       check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--dataset", default="dataset-1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    before = current_orphans(args.dataset)

    new_overrides = merge_into_overrides(decisions)
    print(f"merge_groups now: {len(new_overrides.get('merge_groups', []))}; "
          f"exclude now: {len(new_overrides.get('exclude', []))}")
    if args.dry_run:
        print("(dry-run) not writing er_overrides.json / not re-running")
        return 0

    OVERRIDES.write_text(json.dumps(new_overrides, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    rerun(args.dataset)

    after = current_orphans(args.dataset)
    print("\n=== ORPHAN DELTA ===")
    print(f"  before: {len(before)}   after: {len(after)}")
    removed = sorted(before - after)
    added = sorted(after - before)
    print(f"  removed ({len(removed)}): {removed}")
    if added:
        print(f"  ADDED ({len(added)}) [watch for regressions]: {added}")
    print(f"  still orphan ({len(after)}): {sorted(after)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
