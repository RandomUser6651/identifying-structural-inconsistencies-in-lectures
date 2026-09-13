"""
Assemble the full injected course for each prerequisite-order injection so the
direct-model baseline can read it. Writes injections/<id>/course.md (the clean
rendered lectures in order, with the one prereq-injected lecture swapped in).

Usage: python pipeline-v2/prereq_baseline_prepare.py --dataset dataset-1
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    specs = json.loads((root / "prereq_specs.json").read_text(encoding="utf-8"))

    for s in specs:
        insert = s["insert_lecture"]
        parts = []
        for m in sorted(manifest, key=lambda x: x["lecture_order"]):
            lid = m["lecture_id"]
            if lid == insert:
                p = root / "injections" / s["id"] / f"{lid}.md"
            else:
                p = root / "rendered" / f"{lid}.md"
            parts.append(p.read_text(encoding="utf-8"))
        course_md = "\n\n".join(parts)
        (root / "injections" / s["id"] / "course.md").write_text(course_md, encoding="utf-8")
    print(f"[{args.dataset}] assembled {len(specs)} prereq course.md files")


if __name__ == "__main__":
    main()
