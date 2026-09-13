"""
Entity resolution v2 (pipeline-v2, Ukrainian dataset-1).

Reads per-lecture extraction JSONs from
``new-results/<dataset>/extraction/*.json`` plus ``manifest.json`` (for the
lecture order) and merges concept occurrences that refer to the same idea
across lectures into a single ``normalized.json``.

Differences from the v1 (English) resolver:

* Unit ids ``NN#KK`` replace slide ids; ordering uses (lecture_order, unit
  index) from the manifest.
* **No blind fuzzy character-ratio merge.** On Ukrainian it is unsafe:
  ``синхронна комунікація`` vs ``асинхронна комунікація`` (ratio 98) and
  ``нормалізована схема`` vs ``денормалізована схема`` (ratio 95) are
  antonyms separated only by an а-/де- prefix. Resolution is therefore
  deterministic: exact normalized-name match + alias overlap only.
* A human-curated ``er_overrides.json`` controls the residual hard cases:
  - ``split_groups``: canonical names that must never be alias-bridged into
    one cluster (fixes over-merges such as ``домен`` being absorbed by
    ``Domain Layer`` because one lecture listed it as an alias).
  - ``merge_groups``: names force-merged into one cluster.

Output ``normalized.json`` mirrors the v1 shape so the downstream graph and
detectors port over, with ``slide`` → ``unit`` and ``slide_introduced`` →
``unit_introduced``.

Usage:
  python pipeline-v2/entity_resolution.py            # dataset-1 defaults
  python pipeline-v2/entity_resolution.py --dataset dataset-1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "new-results"
OVERRIDES_PATH = PIPELINE_DIR / "er_overrides.json"

STOPWORDS = {"the", "a", "an", "of"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop stopwords, collapse whitespace.
    ``\\w`` is Unicode-aware in Python's re, so Cyrillic is preserved."""
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"[^\w\s'-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split(" ") if t and t not in STOPWORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, x: int) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return out


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Occurrence:
    name: str
    normalized: str
    aliases: list[str]
    definition: str
    role: str
    lecture: str
    lecture_order: int
    unit: str

    @property
    def alias_keys(self) -> set[str]:
        keys = set()
        for a in self.aliases:
            k = normalize_name(a)
            if k:
                keys.add(k)
        return keys


@dataclass
class Cluster:
    canonical_name: str = ""
    occurrences: list[Occurrence] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest(dataset_dir: Path) -> dict[str, int]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {m["lecture_id"]: m["lecture_order"] for m in manifest}


def load_extractions(extraction_dir: Path) -> list[dict[str, Any]]:
    data = []
    for f in sorted(extraction_dir.glob("*.json")):
        obj = json.loads(f.read_text(encoding="utf-8"))
        if "lecture_id" in obj:
            data.append(obj)
    return data


def collect_occurrences(
    lectures: list[dict[str, Any]], order_of: dict[str, int]
) -> tuple[list[Occurrence], list[dict[str, Any]]]:
    occs: list[Occurrence] = []
    raw_prereqs: list[dict[str, Any]] = []
    for lec in lectures:
        lid = lec.get("lecture_id", "")
        lorder = order_of.get(lid, 0)
        for c in lec.get("concepts") or []:
            name = (c.get("name") or "").strip()
            if not name:
                continue
            occs.append(Occurrence(
                name=name,
                normalized=normalize_name(name),
                aliases=[a for a in (c.get("aliases") or []) if a],
                definition=(c.get("definition") or "").strip(),
                role=(c.get("role") or "used").strip(),
                lecture=lid,
                lecture_order=lorder,
                unit=(c.get("first_mentioned_unit") or "").strip(),
            ))
        for e in lec.get("prerequisites") or []:
            raw_prereqs.append({
                "from_raw": (e.get("from") or "").strip(),
                "to_raw":   (e.get("to") or "").strip(),
                "reason":   (e.get("reason") or "").strip(),
                "evidence_unit": (e.get("evidence_unit") or "").strip(),
                "evidence_lecture": lid,
            })
    return occs, raw_prereqs


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def load_overrides() -> tuple[list[set[str]], list[set[str]], set[str]]:
    if not OVERRIDES_PATH.exists():
        return [], [], set()
    o = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    splits = [{normalize_name(n) for n in g} for g in o.get("split_groups", [])]
    merges = [{normalize_name(n) for n in g} for g in o.get("merge_groups", [])]
    # exclude: normalized names dropped from clustering entirely (non-discipline
    # nodes — code identifiers, generic vocabulary — curated, domain expert in
    # the loop). Prereq endpoints that resolve only to an excluded name dangle
    # and are dropped downstream.
    exclude = {normalize_name(n) for n in o.get("exclude", [])}
    return splits, merges, exclude


def _forbidden(a: str, b: str, splits: list[set[str]]) -> bool:
    """True if normalized names a and b are distinct members of a split group."""
    if a == b:
        return False
    for g in splits:
        if a in g and b in g:
            return True
    return False


def cluster_occurrences(occs: list[Occurrence]) -> list[Cluster]:
    splits, merges, exclude = load_overrides()
    if exclude:
        occs = [o for o in occs if o.normalized not in exclude]
    n = len(occs)
    uf = UnionFind()
    for i in range(n):
        uf.add(i)

    # (1) exact normalized-name index
    by_norm: dict[str, list[int]] = defaultdict(list)
    for i, o in enumerate(occs):
        if o.normalized:
            by_norm[o.normalized].append(i)
    for group in by_norm.values():
        for j in group[1:]:
            uf.union(group[0], j)

    # Ambiguous shared sub-phrases: a normalized phrase that is a STRICT
    # token-subset of >=2 distinct occurrence names (a generic head shared by
    # sibling compounds, e.g. "моноліт" inside both "класичний моноліт" and
    # "модульний моноліт"). Bridging a merge via such an alias conflates
    # distinct concepts and was the cause of false used-before flags
    # (extractor abbreviated "класичний моноліт" to its generic head "моноліт").
    # A short form that is a sub-phrase of only ONE concept (a genuine
    # abbreviation, e.g. "CAP" -> "CAP Theorem") is NOT ambiguous and still
    # bridges normally.
    name_tokens = {nm: frozenset(nm.split()) for nm in by_norm if nm}

    def _is_ambiguous(key: str) -> bool:
        # Ambiguous iff the key is a strict head/sub-phrase of >=2 DISTINCT longer
        # concepts (genuine sibling compounds, e.g. "моноліт" inside both
        # "класичний моноліт" and "модульний моноліт"). A head of only one longer
        # concept (e.g. "Presentation" inside just "Presentation Layer") is a safe
        # abbreviation and must still bridge; a cross-language alias ("репозиторій"
        # for "Repository") is not a sub-phrase at all, so it is never blocked.
        kt = frozenset(key.split())
        if not kt:
            return False
        strict_supersets = sum(1 for t in name_tokens.values() if kt < t)
        return strict_supersets >= 2

    # (2) alias overlap, blocked when it would bridge two split-group members
    # or when the alias is an ambiguous shared sub-phrase (generic head).
    for i, o in enumerate(occs):
        for key in o.alias_keys:
            if key and key != o.normalized and key in by_norm:
                if _forbidden(o.normalized, key, splits):
                    continue
                if _is_ambiguous(key):
                    continue
                uf.union(i, by_norm[key][0])

    # (3) forced merges from overrides
    for g in merges:
        members = [i for i, o in enumerate(occs) if o.normalized in g]
        for j in members[1:]:
            uf.union(members[0], j)

    groups = uf.groups()
    clusters: list[Cluster] = []
    for members in groups.values():
        cluster_occs = [occs[i] for i in members]
        canonical = _pick_canonical_name(cluster_occs)
        canon_norm = normalize_name(canonical)
        aliases = set()
        for o in cluster_occs:
            if o.name != canonical:
                aliases.add(o.name)
            for a in o.aliases:
                if a and a != canonical:
                    aliases.add(a)
        # Drop aliases that a split rule forbids from co-occurring with the
        # canonical name (e.g. keep "домен" off Domain Layer's alias list so it
        # cannot misroute prerequisite endpoints via the global fallback).
        aliases = {a for a in aliases
                   if not _forbidden(canon_norm, normalize_name(a), splits)}
        clusters.append(Cluster(canonical_name=canonical,
                                occurrences=cluster_occs, aliases=aliases))
    clusters.sort(key=lambda c: c.canonical_name.lower())
    return clusters


def _pick_canonical_name(occs: list[Occurrence]) -> str:
    """Most frequent name; ties → introduced-role occurrences → shortest."""
    freq: dict[str, int] = defaultdict(int)
    intro_names: set[str] = set()
    for o in occs:
        freq[o.name] += 1
        if o.role == "introduced":
            intro_names.add(o.name)
    best = sorted(
        freq.items(),
        key=lambda kv: (-kv[1], kv[0] not in intro_names, len(kv[0]), kv[0]),
    )
    return best[0][0] if best else ""


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def unit_order(lecture_order: int, unit: str) -> tuple[int, int]:
    m = re.search(r"#(\d+)", unit or "")
    sub = int(m.group(1)) if m else 0
    return (lecture_order, sub)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

def emit_normalized(dataset: str, lectures: list[dict[str, Any]],
                    clusters: list[Cluster],
                    raw_prereqs: list[dict[str, Any]]) -> dict[str, Any]:
    by_raw: dict[tuple[str, str], str] = {}
    global_by_name: dict[str, str] = {}
    for c in clusters:
        for o in c.occurrences:
            by_raw[(o.lecture, o.name)] = c.canonical_name
            global_by_name[normalize_name(o.name)] = c.canonical_name
            for a in o.aliases:
                by_raw[(o.lecture, a)] = c.canonical_name
                global_by_name[normalize_name(a)] = c.canonical_name

    def resolve(lecture: str, raw: str) -> str | None:
        if (lecture, raw) in by_raw:
            return by_raw[(lecture, raw)]
        return global_by_name.get(normalize_name(raw))

    concept_payload: list[dict[str, Any]] = []
    for c in clusters:
        appearances = []
        roles = {"introduced": 0, "used": 0}
        intro: tuple[int, str, str] | None = None
        definitions = []
        for o in c.occurrences:
            appearances.append({"lecture": o.lecture,
                                "lecture_order": o.lecture_order,
                                "unit": o.unit, "role": o.role})
            roles[o.role] = roles.get(o.role, 0) + 1
            if o.role == "introduced":
                key = unit_order(o.lecture_order, o.unit)
                if intro is None or key < (intro[0], unit_order(intro[0], intro[2])[1]):
                    intro = (o.lecture_order, o.lecture, o.unit)
                if o.definition:
                    definitions.append({"text": o.definition,
                                        "lecture": o.lecture, "unit": o.unit})
        if intro:
            _, lecture_introduced, unit_introduced = intro
        else:
            lecture_introduced, unit_introduced = None, None
        concept_payload.append({
            "canonical_name": c.canonical_name,
            "aliases": sorted(c.aliases - {c.canonical_name}),
            "definitions": definitions,
            "lecture_introduced": lecture_introduced,
            "unit_introduced": unit_introduced,
            "appearances": appearances,
            "roles": roles,
        })

    pre_payload: list[dict[str, Any]] = []
    dropped = 0
    for e in raw_prereqs:
        fr = resolve(e["evidence_lecture"], e["from_raw"])
        to = resolve(e["evidence_lecture"], e["to_raw"])
        if not fr or not to or fr == to:
            dropped += 1
            continue
        pre_payload.append({
            "from": fr, "to": to, "reason": e["reason"],
            "evidence_lecture": e["evidence_lecture"],
            "evidence_unit": e["evidence_unit"],
        })

    stats = {
        "lectures": len(lectures),
        "raw_concept_occurrences": sum(len(c.occurrences) for c in clusters),
        "unique_concepts": len(clusters),
        "concepts_introduced_at_least_once":
            sum(1 for c in concept_payload if c["lecture_introduced"]),
        "concepts_never_introduced":
            sum(1 for c in concept_payload if not c["lecture_introduced"]),
        "raw_prerequisites": len(raw_prereqs),
        "kept_prerequisites": len(pre_payload),
        "dropped_prerequisites": dropped,
    }
    return {
        "dataset": dataset,
        "lectures": [{"id": lec.get("lecture_id", "")} for lec in lectures],
        "concepts": concept_payload,
        "prerequisites": pre_payload,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()

    dataset_dir = RESULTS_ROOT / args.dataset
    extraction_dir = dataset_dir / "extraction"
    order_of = load_manifest(dataset_dir)
    lectures = load_extractions(extraction_dir)
    if not lectures:
        raise SystemExit(f"no extraction JSONs in {extraction_dir}")

    occs, raw_prereqs = collect_occurrences(lectures, order_of)
    clusters = cluster_occurrences(occs)
    payload = emit_normalized(args.dataset, lectures, clusters, raw_prereqs)

    out = dataset_dir / "normalized.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    s = payload["stats"]
    print(f"[{args.dataset}] lectures={s['lectures']} "
          f"raw_occ={s['raw_concept_occurrences']} "
          f"unique={s['unique_concepts']} "
          f"never_introduced={s['concepts_never_introduced']} "
          f"prereqs={s['kept_prerequisites']}/{s['raw_prerequisites']} "
          f"(dropped {s['dropped_prerequisites']})")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
