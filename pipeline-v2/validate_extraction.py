"""
Validator for pipeline-v2 extraction JSONs (concept_extraction_ua schema).

Checks structure AND referential integrity:
  - top-level keys exactly {lecture_id, concepts, prerequisites}
  - concept fields, role enum, role/definition consistency
    (introduced -> non-empty definition, used -> empty definition)
  - first_mentioned_unit / evidence_unit exist among the lecture's CONTENT
    units (per the units JSON)
  - no duplicate concept names (case-insensitive)
  - no alias equal to its own concept name or to another concept's name
  - prerequisite endpoints present in concepts, no self-loops, no duplicate
    edges
Warnings (non-fatal):
  - concept name/alias not found as substring of the lecture text
    (Ukrainian declension can legitimately trigger this)

Exit code 0 = valid (warnings allowed), 1 = errors.

Usage:
  python validate_extraction.py <extraction.json> --units <units.json>
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extraction", type=Path)
    ap.add_argument("--units", type=Path, required=True)
    args = ap.parse_args()

    errors, warnings = [], []

    try:
        data = json.loads(args.extraction.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: cannot parse {args.extraction}: {e}")
        sys.exit(1)

    units_doc = json.loads(args.units.read_text(encoding="utf-8"))
    content_unit_ids = {u["unit_id"] for u in units_doc["units"]
                        if u["kind"] == "content"}
    full_text = "\n".join(u["text"] for u in units_doc["units"]).lower()
    headings_text = "\n".join(u["heading"] for u in units_doc["units"]).lower()
    searchable = full_text + "\n" + headings_text

    # --- top level ---------------------------------------------------------
    expected_keys = {"lecture_id", "concepts", "prerequisites"}
    if set(data.keys()) != expected_keys:
        errors.append(f"top-level keys {sorted(data.keys())} != {sorted(expected_keys)}")
    if data.get("lecture_id") != units_doc["lecture_id"]:
        errors.append(f"lecture_id {data.get('lecture_id')!r} != units file "
                      f"{units_doc['lecture_id']!r}")

    concepts = data.get("concepts", [])
    prereqs = data.get("prerequisites", [])
    if not isinstance(concepts, list) or not isinstance(prereqs, list):
        errors.append("concepts/prerequisites must be arrays")
        concepts, prereqs = [], []

    # --- concepts ----------------------------------------------------------
    names_lower = {}
    all_alias_owner = {}
    for i, c in enumerate(concepts):
        loc = f"concepts[{i}]"
        missing = {"name", "aliases", "definition", "first_mentioned_unit",
                   "role"} - set(c.keys())
        extra = set(c.keys()) - {"name", "aliases", "definition",
                                 "first_mentioned_unit", "role"}
        if missing:
            errors.append(f"{loc}: missing fields {sorted(missing)}")
            continue
        if extra:
            errors.append(f"{loc}: unexpected fields {sorted(extra)}")
        name = c["name"]
        nl = name.strip().lower()
        if not nl:
            errors.append(f"{loc}: empty name")
            continue
        if nl in names_lower:
            errors.append(f"{loc}: duplicate concept name {name!r} "
                          f"(also concepts[{names_lower[nl]}])")
        names_lower[nl] = i

        if c["role"] not in ("introduced", "used"):
            errors.append(f"{loc} ({name}): bad role {c['role']!r}")
        if c["role"] == "introduced" and not c["definition"].strip():
            errors.append(f"{loc} ({name}): introduced but empty definition")
        if c["role"] == "used" and c["definition"].strip():
            errors.append(f"{loc} ({name}): used but non-empty definition")

        if c["first_mentioned_unit"] not in content_unit_ids:
            errors.append(f"{loc} ({name}): first_mentioned_unit "
                          f"{c['first_mentioned_unit']!r} not a content unit "
                          f"of this lecture")

        if not isinstance(c["aliases"], list):
            errors.append(f"{loc} ({name}): aliases must be an array")
            continue
        for a in c["aliases"]:
            al = a.strip().lower()
            if al == nl:
                errors.append(f"{loc} ({name}): alias equals own name")
            all_alias_owner.setdefault(al, (i, name))

        if nl not in searchable:
            hit = any(a.strip().lower() in searchable for a in c["aliases"])
            if not hit:
                warnings.append(f"{loc} ({name}): name and aliases not found "
                                f"verbatim in lecture text (declension?)")

    # alias colliding with another concept's canonical name
    for al, (i, owner) in all_alias_owner.items():
        if al in names_lower and names_lower[al] != i:
            errors.append(f"alias {al!r} of concept {owner!r} equals the "
                          f"canonical name of another concept")

    # --- prerequisites -----------------------------------------------------
    seen_edges = set()
    for i, p in enumerate(prereqs):
        loc = f"prerequisites[{i}]"
        missing = {"from", "to", "reason", "evidence_unit"} - set(p.keys())
        if missing:
            errors.append(f"{loc}: missing fields {sorted(missing)}")
            continue
        f, t = p["from"].strip().lower(), p["to"].strip().lower()
        if f not in names_lower:
            errors.append(f"{loc}: from={p['from']!r} not in concepts")
        if t not in names_lower:
            errors.append(f"{loc}: to={p['to']!r} not in concepts")
        if f == t:
            errors.append(f"{loc}: self-loop on {p['from']!r}")
        if (f, t) in seen_edges:
            errors.append(f"{loc}: duplicate edge {p['from']!r} -> {p['to']!r}")
        seen_edges.add((f, t))
        if p["evidence_unit"] not in content_unit_ids:
            errors.append(f"{loc}: evidence_unit {p['evidence_unit']!r} "
                          f"not a content unit of this lecture")

    # --- report ------------------------------------------------------------
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    n_intro = sum(1 for c in concepts
                  if isinstance(c, dict) and c.get("role") == "introduced")
    print(f"--- {data.get('lecture_id')}: {len(concepts)} concepts "
          f"({n_intro} introduced), {len(prereqs)} prerequisites, "
          f"{len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
