#!/usr/bin/env python3
"""Emit concept_vocab.txt (global concept list) for the dedicated prerequisite-
edge extractor: one line per graph node, `- canonical (alias, alias)`.

Usage: python build_concept_vocab.py --dataset dataset-2
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-2")
    args = ap.parse_args()
    root = ROOT / "new-results" / args.dataset
    g = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    lines = []
    for n in sorted(g["nodes"], key=lambda x: x["canonical_name"].lower()):
        name = n["canonical_name"]
        al = n.get("aliases") or []
        lines.append(f"- {name}" + (f" ({', '.join(al)})" if al else ""))
    (root / "concept_vocab.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{args.dataset}] wrote {len(lines)} concepts -> concept_vocab.txt")


if __name__ == "__main__":
    sys.exit(main())
