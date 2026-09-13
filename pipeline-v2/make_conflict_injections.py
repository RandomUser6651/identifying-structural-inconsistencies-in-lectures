"""
Build controlled definition-conflict injections.

A target concept that is introduced exactly once (single definition, single
lecture) is given a CONTRADICTORY second definition inserted into a different
lecture in which it does not already appear. After re-extraction the concept
acquires two definitions from two lectures, becoming a definition-conflict
candidate; the language-model verifier must then judge it a genuine conflict.

This script only selects targets and records the real definition; a separate
authoring step writes the contradictory definition, after which inject_conflicts
inserts it. Output: conflict_injections.json (specs).

Usage: python pipeline-v2/make_conflict_injections.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"
SUBTLETY = ["blatant", "medium", "subtle"]

# hand-picked targets with substantive, unambiguous real definitions, per course
TARGETS = {
    "dataset-1": ["Aggregate Root", "Anemic Domain Model", "Clean Architecture",
                  "ACID", "Context Map", "Anti-Corruption Layer"],
    "dataset-3": ["акмеологія", "академічна свобода", "бінарна лекція",
                  "Велика дидактика", "вхідний контроль", "вступна лекція"],
}


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()
    root = RESULTS / args.dataset
    g = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    ids_sorted = [m["lecture_id"] for m in sorted(manifest, key=lambda x: x["lecture_order"])]

    unit_text = {}
    for f in (root / "units").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        unit_text[d["lecture_id"]] = " ".join(u["text"].lower() for u in d["units"])

    specs = []
    for k, name in enumerate(TARGETS[args.dataset]):
        node = nodes.get(name)
        if not node or not node.get("definitions"):
            print(f"  skip {name}: missing"); continue
        intro = node["lecture_introduced"]
        real_def = node["definitions"][0]["text"]
        # insert lecture: one where the concept name does NOT already appear
        insert = None
        for lid in ids_sorted:
            if lid == intro:
                continue
            if normalize(name) not in unit_text.get(lid, ""):
                insert = lid; break
        if not insert:
            print(f"  skip {name}: no clean insert lecture"); continue
        tier = SUBTLETY[k % 3]
        specs.append({"id": f"cinj-{k+1:03d}", "type": "definition_conflict",
                      "concept": name, "real_definition": real_def,
                      "intro_lecture": intro, "insert_lecture": insert,
                      "subtlety": tier, "contradictory_definition": None})

    (root / "conflict_injections.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.dataset}] {len(specs)} conflict injections:")
    for s in specs:
        print(f"  {s['id']} {s['subtlety']:8} {s['concept'][:28]:28} "
              f"intro {s['intro_lecture'][:20]:20} -> insert {s['insert_lecture']}")


if __name__ == "__main__":
    main()
