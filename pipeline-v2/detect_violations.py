"""
Stage 4 (pipeline-v2): structural-violation detectors over the temporal
concept graph.

Four detectors + three algorithmic enhancements, ported to the unit-based
ordering of the Ukrainian pipeline:

1. ``detect_order_violations`` — transitive prerequisite-chain check. For
   every concept C, every ancestor A in the prerequisite DAG must be
   introduced no later than C is first used. Comparison uses the explicit
   numeric ``order = (lecture_order, unit_index)`` stored on the nodes, so
   it is sensitive to *within-lecture* ordering, not just lecture number.
2. ``detect_definition_conflicts`` — emits every concept with >=2 definitions
   from different lectures as a CANDIDATE. Optional semantic verification
   (``--embeddings``) keeps only pairs whose cosine similarity falls below
   the threshold (genuine conflict, not paraphrase); without it, candidates
   are passed through with ``semantic_similarity = null`` for a later
   verification stage (embeddings or subagent judgment).
3. ``detect_orphan_concepts`` — used-without-introduction + isolated singleton.
4. ``detect_cycles`` — simple cycles in the prerequisite DAG.

PageRank importance is attached to every violation; output is sorted by it.

The graph part is fully deterministic and needs no network access. Embedding
verification is the only optional online step.

Usage:
  python pipeline-v2/detect_violations.py                 # candidates only
  python pipeline-v2/detect_violations.py --embeddings    # + cosine filter
  python pipeline-v2/detect_violations.py --dataset dataset-1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import networkx as nx

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace", encoding="utf-8")
    except Exception:
        pass

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "new-results"

SIMILARITY_THRESHOLD = 0.82
EMBED_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Ordering helpers
# ---------------------------------------------------------------------------

def order_of(node_data: dict, which: str) -> tuple[int, int] | None:
    """which in {'introduced', 'first_used'}. Returns the stored numeric
    order tuple, or None if the node lacks that timestamp."""
    val = node_data.get(f"order_{which}")
    if not val:
        return None
    return (val[0], val[1])


def _unit_idx(unit: str) -> int:
    import re
    m = re.search(r"#(\d+)", unit or "")
    return int(m.group(1)) if m else 0


def earliest_used(node_data: dict, skip_forward_ref: bool = True):
    """Earliest appearance with role 'used'. Skips marked markdown
    forward-references (``[X](x.md)``) by default, so an explicit cross-lecture
    citation is not treated as use-before-introduction. Returns
    (order_tuple, appearance) or (None, None)."""
    best = None
    for a in node_data.get("appearances") or []:
        if a.get("role") != "used":
            continue
        if skip_forward_ref and a.get("forward_ref"):
            continue
        key = (a.get("lecture_order", 0), _unit_idx(a.get("unit", "")))
        if best is None or key < best[0]:
            best = (key, a)
    return best if best else (None, None)


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.readwrite.json_graph.node_link_graph(
        data, directed=True, multigraph=False, edges="edges")


# ---------------------------------------------------------------------------
# Embeddings (optional)
# ---------------------------------------------------------------------------

def compute_embeddings(texts: list[str]) -> list[list[float]] | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        print("  WARN: OPENAI_API_KEY not set; skipping embedding verification")
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+100])
            out.extend(item.embedding for item in resp.data)
        return out
    except Exception as exc:
        print(f"  WARN: embeddings failed: {exc}")
        return None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------

def compute_importance(g: nx.DiGraph) -> dict[str, float]:
    try:
        pr = nx.pagerank(g, alpha=0.85)
    except Exception:
        n = g.number_of_nodes() or 1
        pr = {node: 1.0 / n for node in g.nodes}
    if pr:
        mx = max(pr.values())
        if mx > 0:
            pr = {k: v / mx for k, v in pr.items()}
    return pr


# ---------------------------------------------------------------------------
# Detector 1: order violations (transitive)
# ---------------------------------------------------------------------------

def detect_used_before_introduction(g: nx.DiGraph) -> list[dict[str, Any]]:
    """Type A — a concept is *used* (applied/referenced) before it is itself
    introduced. One flag per concept; the fundamental, non-derived signal.
    Concepts never introduced at all are left to the orphan detector."""
    out: list[dict[str, Any]] = []
    for name, data in g.nodes(data=True):
        intro = order_of(data, "introduced")
        if not intro:
            continue
        eu_order, eu_app = earliest_used(data)
        if eu_order and eu_order < intro:
            n_before = sum(
                1 for a in (data.get("appearances") or [])
                if a.get("role") == "used" and not a.get("forward_ref")
                and (a.get("lecture_order", 0), _unit_idx(a.get("unit", ""))) < intro)
            out.append({
                "type": "used_before_introduction",
                "concept": name,
                "used_lecture": eu_app.get("lecture"),
                "used_unit": eu_app.get("unit"),
                "introduced_lecture": data.get("lecture_introduced"),
                "introduced_unit": data.get("unit_introduced"),
                "n_used_before": n_before,
                "description": (
                    f'«{name}» вживається у курсі (вперше у {eu_app.get("lecture")} / '
                    f'{eu_app.get("unit")}) ще до того, як його формально введено '
                    f'({data.get("lecture_introduced")} / {data.get("unit_introduced")}).'),
            })
    return out


def detect_prerequisite_order_anchored(g: nx.DiGraph,
                                       lecture_order_map: dict[str, int]) -> list[dict[str, Any]]:
    """Type B, fact-anchored. For each DIRECT prerequisite edge (A -> C), the
    dependency is anchored where it is actually stated (the edge's
    evidence_unit), not at C's first introduction. Violation iff A is introduced
    AFTER the unit where it is needed:

        order(A_introduced) > order(evidence_unit)

    This removes the within-statement false positives (a prerequisite explained
    in the very unit where it is used is available when needed). Transitive
    edges have no evidence anchor and are intentionally not emitted."""
    out: list[dict[str, Any]] = []
    for fr, to, data in g.edges(data=True):
        a_intro = order_of(g.nodes[fr], "introduced")
        if not a_intro:
            continue
        ev_lec = data.get("evidence_lecture")
        if ev_lec not in lecture_order_map:
            continue
        ev = (lecture_order_map[ev_lec], _unit_idx(data.get("evidence_unit", "")))
        if a_intro > ev:
            out.append({
                "type": "prerequisite_order",
                "prerequisite": fr,
                "dependent": to,
                "transitive": False,
                "evidence_lecture": ev_lec,
                "evidence_unit": data.get("evidence_unit", ""),
                "prereq_introduced_order": list(a_intro),
                "evidence_order": list(ev),
                "same_lecture": a_intro[0] == ev[0],
                "description": (
                    f'«{fr}» потрібна для «{to}» у {ev_lec}/{data.get("evidence_unit","")}, '
                    f'але вводиться лише пізніше (поз. {list(a_intro)}).'),
            })
    return out


def detect_prerequisite_order(g: nx.DiGraph,
                              use_transitive: bool = True) -> list[dict[str, Any]]:
    """Type B — a prerequisite A of concept C is INTRODUCED after C is
    introduced. Anchored on introduction times of BOTH endpoints (not first
    use), so an early name-drop of C no longer spawns derived violations for
    all of C's prerequisites. Concepts without an introduction are skipped."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def record(anc: str, dep: str, transitive: bool):
        out.append({
            "type": "prerequisite_order",
            "prerequisite": anc,
            "dependent": dep,
            "transitive": transitive,
            "prereq_introduced_lecture": g.nodes[anc].get("lecture_introduced"),
            "prereq_introduced_unit": g.nodes[anc].get("unit_introduced"),
            "dependent_introduced_lecture": g.nodes[dep].get("lecture_introduced"),
            "dependent_introduced_unit": g.nodes[dep].get("unit_introduced"),
            "description": (
                f'«{anc}» {"(транзитивна) " if transitive else ""}передумова для '
                f'«{dep}», але вводиться у {g.nodes[anc].get("lecture_introduced")}, '
                f'тоді як «{dep}» вже вводиться раніше — у '
                f'{g.nodes[dep].get("lecture_introduced")}.'),
        })

    if not use_transitive:
        for fr, to in g.edges():
            a_intro = order_of(g.nodes[fr], "introduced")
            d_intro = order_of(g.nodes[to], "introduced")
            if a_intro and d_intro and a_intro > d_intro:
                record(fr, to, transitive=False)
        return out

    for concept in g.nodes:
        d_intro = order_of(g.nodes[concept], "introduced")
        if not d_intro:
            continue
        try:
            ancestors = nx.ancestors(g, concept)
        except nx.NetworkXError:
            continue
        for anc in ancestors:
            a_intro = order_of(g.nodes[anc], "introduced")
            if not a_intro:
                continue
            if a_intro > d_intro:
                key = (anc, concept)
                if key in seen:
                    continue
                seen.add(key)
                record(anc, concept, transitive=not g.has_edge(anc, concept))
    return out


# ---------------------------------------------------------------------------
# Detector 2: definition conflicts
# ---------------------------------------------------------------------------

def detect_definition_conflicts(g: nx.DiGraph,
                                use_embeddings: bool = False) -> list[dict[str, Any]]:
    candidates: list[tuple[str, list[dict]]] = []
    for name, data in g.nodes(data=True):
        defs = data.get("definitions") or []
        if len(defs) < 2:
            continue
        if len({d.get("lecture") for d in defs}) < 2:
            continue
        candidates.append((name, defs))

    if not candidates:
        return []

    if use_embeddings:
        pair_texts: list[str] = []
        pair_info: list[tuple[str, dict, dict]] = []
        for name, defs in candidates:
            for i in range(len(defs)):
                for j in range(i + 1, len(defs)):
                    if defs[i].get("lecture") == defs[j].get("lecture"):
                        continue
                    pair_texts.append(defs[i].get("text", ""))
                    pair_texts.append(defs[j].get("text", ""))
                    pair_info.append((name, defs[i], defs[j]))
        embs = compute_embeddings(pair_texts) if pair_texts else None
        if embs and len(embs) == len(pair_texts):
            out: list[dict[str, Any]] = []
            reported: set[str] = set()
            min_sim: dict[str, float] = {}
            for pi, (name, d1, d2) in enumerate(pair_info):
                sim = cosine(embs[pi * 2], embs[pi * 2 + 1])
                min_sim[name] = min(min_sim.get(name, 1.0), sim)
            for name, defs in candidates:
                sim = min_sim.get(name, 1.0)
                if sim < SIMILARITY_THRESHOLD and name not in reported:
                    reported.add(name)
                    out.append({
                        "type": "definition_conflict",
                        "concept": name,
                        "definition_count": len(defs),
                        "definitions": defs,
                        "semantic_similarity": round(sim, 4),
                        "verified": True,
                        "description": (
                            f'«{name}» має семантично різні визначення '
                            f'(cosine = {sim:.3f} < {SIMILARITY_THRESHOLD}); '
                            f'імовірно справжній конфлікт, не парафраз.'),
                    })
            print(f"    embeddings: {len(pair_info)} pairs, {len(out)} conflicts "
                  f"(threshold={SIMILARITY_THRESHOLD})")
            return out

    # Candidate-only (no embeddings): pass through for a later verification stage
    out = []
    for name, defs in candidates:
        out.append({
            "type": "definition_conflict",
            "concept": name,
            "definition_count": len(defs),
            "definitions": defs,
            "semantic_similarity": None,
            "verified": False,
            "description": (
                f'«{name}» має {len(defs)} визначень у '
                f'{len({d.get("lecture") for d in defs})} різних лекціях; '
                f'кандидат на конфлікт (потребує семантичної верифікації).'),
        })
    return out


# ---------------------------------------------------------------------------
# Detector 3: orphans
# ---------------------------------------------------------------------------

def detect_orphan_concepts(g: nx.DiGraph) -> list[dict[str, Any]]:
    """Type C — a concept is *used* in the course but never introduced/defined
    anywhere (a genuine content gap).

    Only the ``used_without_introduction`` subtype is emitted. The former
    ``isolated_singleton`` subtype (a concept that IS introduced with a
    definition but happens to have no prerequisite edges and appears in one
    lecture) was removed: an adversarial audit of dataset-1 found that every
    instance was a properly-defined concept — antipatterns and protocols
    introduced in comparison tables (HTTP, gRPC, God Object, Identity Map,
    Onion Architecture, …) — i.e. pure false positives, not content gaps. It is
    also not part of the injected/measured defect taxonomy
    (INJECTED_SIG_TYPES), so dropping it cannot change recall — only the
    clean-course false-positive surface."""
    out: list[dict[str, Any]] = []
    for name, data in g.nodes(data=True):
        n_intro = data.get("n_introduced") or 0
        n_used = data.get("n_used") or 0
        if n_intro == 0 and n_used > 0:
            out.append({
                "type": "orphan_concept",
                "subtype": "used_without_introduction",
                "concept": name,
                "n_used": n_used,
                "first_used_lecture": data.get("lecture_first_used"),
                "first_used_unit": data.get("unit_first_used"),
                "description": (
                    f'«{name}» використовується у курсі, але ніде формально '
                    f'не вводиться (вперше вжито у {data.get("lecture_first_used")} / '
                    f'{data.get("unit_first_used")}).'),
            })
    return out


# ---------------------------------------------------------------------------
# Detector 4: cycles
# ---------------------------------------------------------------------------

def detect_cycles(g: nx.DiGraph) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cycle in nx.simple_cycles(g):
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        out.append({
            "type": "cyclic_dependency",
            "length": len(cycle),
            "cycle": list(cycle),
            "edges": edges,
            "description": "Циклічна залежність передумов: "
                           + " -> ".join(cycle + [cycle[0]]),
        })
    return out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def run_detectors(g: nx.DiGraph, use_embeddings: bool = False,
                  use_transitive: bool = True) -> dict[str, Any]:
    importance = compute_importance(g)
    used_before = detect_used_before_introduction(g)
    prereq_order = detect_prerequisite_order(g, use_transitive=use_transitive)
    defs = detect_definition_conflicts(g, use_embeddings=use_embeddings)
    orphans = detect_orphan_concepts(g)
    cycles = detect_cycles(g)

    def add_importance(vs: list[dict]) -> list[dict]:
        for v in vs:
            concept = (v.get("concept") or v.get("dependent")
                       or v.get("prerequisite")
                       or (v.get("cycle", [None])[0]) or "")
            v["importance"] = round(importance.get(concept, 0.0), 4)
        vs.sort(key=lambda v: -v.get("importance", 0))
        return vs

    used_before = add_importance(used_before)
    prereq_order = add_importance(prereq_order)
    defs = add_importance(defs)
    orphans = add_importance(orphans)
    cycles = add_importance(cycles)

    n_direct = sum(1 for v in prereq_order if not v.get("transitive"))
    n_trans = sum(1 for v in prereq_order if v.get("transitive"))

    return {
        "counts": {
            "used_before_introduction": len(used_before),
            "prerequisite_order": len(prereq_order),
            "prereq_direct": n_direct,
            "prereq_transitive": n_trans,
            "definition_conflicts": len(defs),
            "orphan_concepts": len(orphans),
            "cycles": len(cycles),
        },
        "used_before_introduction": used_before,
        "prerequisite_order": prereq_order,
        "definition_conflicts": defs,
        "orphan_concepts": orphans,
        "cycles": cycles,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    ap.add_argument("--embeddings", action="store_true",
                    help="Run cosine verification of definition conflicts via OpenAI embeddings")
    ap.add_argument("--no-transitive", action="store_true")
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / args.dataset
    g = load_graph(out_dir / "graph.json")
    print(f"[{args.dataset}] nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

    report = run_detectors(g, use_embeddings=args.embeddings,
                           use_transitive=not args.no_transitive)
    report["dataset"] = args.dataset
    report["graph_summary"] = {"nodes": g.number_of_nodes(),
                               "edges": g.number_of_edges()}
    report["params"] = {"embeddings": args.embeddings,
                        "transitive": not args.no_transitive,
                        "similarity_threshold": SIMILARITY_THRESHOLD}

    (out_dir / "violations.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    c = report["counts"]
    print(f"  used_before_intro={c['used_before_introduction']:3d}  "
          f"prereq_order={c['prerequisite_order']:3d} "
          f"(direct={c['prereq_direct']}, transitive={c['prereq_transitive']})  "
          f"defs={c['definition_conflicts']:3d}  "
          f"orphans={c['orphan_concepts']:3d}  cycles={c['cycles']:2d}")
    print(f"  -> {out_dir / 'violations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
