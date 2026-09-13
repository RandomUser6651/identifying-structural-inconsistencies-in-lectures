"""
Stage 3b (pipeline-v2): temporal concept-graph builder.

Reads ``normalized.json`` (output of entity_resolution.py) and produces a
directed prerequisite graph for one course.

Each node carries an explicit numeric ordering key so the detectors never
have to parse lecture names again:

    order = (lecture_order, unit_index)

where ``lecture_order`` is the lecture's position from the manifest and
``unit_index`` is the ``KK`` of the ``NN#KK`` unit id. ``introduced`` uses the
earliest unit with role "introduced"; ``first_used`` uses the earliest unit in
ANY role.

Outputs (under new-results/<dataset>/):
  graph.json      node-link JSON (lossless, what the detectors load)
  graph.graphml   flattened GraphML (Gephi-readable)

Usage:
  python pipeline-v2/build_graph.py            # dataset-1 default
  python pipeline-v2/build_graph.py --dataset dataset-1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

PIPELINE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = PIPELINE_DIR.parent / "new-results"

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")


def unit_index(unit: str) -> int:
    m = re.search(r"#(\d+)", unit or "")
    return int(m.group(1)) if m else 0


def load_units(out_dir: Path) -> dict[tuple[str, str], str]:
    """(lecture_id, unit_id) -> unit text, for forward-link detection."""
    lut: dict[tuple[str, str], str] = {}
    for f in sorted((out_dir / "units").glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        for u in doc["units"]:
            lut[(doc["lecture_id"], u["unit_id"])] = u["text"]
    return lut


def _content_stems(term: str) -> list[str]:
    """Inflection-tolerant stems of a term's content words (>=4 chars), e.g.
    'модульний моноліт' -> ['модульн', 'моноліт']."""
    toks = [w for w in re.findall(r"[\w'-]+", (term or "").lower()) if len(w) >= 4]
    return [w[:-2] if len(w) > 6 else w for w in toks]


def appearance_is_forward_ref(text: str, terms: list[str]) -> bool:
    """True if this unit cites the concept via a markdown link to another
    lecture, e.g. ``[Repository](repository.md)`` — an explicit *marked*
    forward/cross reference, not a use-as-known.

    Matching is inflection-tolerant: a Ukrainian link text such as
    ``[модульного моноліту](modular-monolith.md)`` is recognised for the term
    ``модульний моноліт`` (an exact substring no longer matches because of the
    case endings). A term matches a link text if its surface form is a
    substring OR all of its content-word stems appear in the link text.

    NOTE: this suppresses an appearance only when the concept name is INSIDE
    the markdown link anchor. A "see [other document]" pointer where the
    concept is named beside (outside) the link — e.g. CAP theorem at 12#07 —
    is intentionally NOT treated as a forward-ref: the instructor's criterion
    judges a forward-reference use before the definition to be a real defect
    (CAP was marked real), so the detector must keep flagging it."""
    low_terms = [t.lower() for t in terms if t]
    for m in _LINK_RE.finditer(text or ""):
        ltext = m.group(1).lower()
        for t in low_terms:
            if t in ltext:
                return True
        for t in terms:
            stems = _content_stems(t)
            if stems and all(s in ltext for s in stems):
                return True
    return False


def _flatten(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v if isinstance(v, str) else str(v)
    return json.dumps(v, ensure_ascii=False)


def build_graph(normalized: dict[str, Any], units: dict | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    g.graph["dataset"] = normalized.get("dataset", "")
    units = units or {}

    for c in normalized.get("concepts", []):
        name = c["canonical_name"]
        appearances = c.get("appearances") or []
        terms = [name] + (c.get("aliases") or [])

        # Annotate each appearance with a forward-ref flag (marked markdown
        # cross-reference to another lecture), then compute first-use from
        # NON-forward-ref appearances so a "see [X](x.md)" mention does not
        # count as use-before-introduction.
        for app in appearances:
            txt = units.get((app.get("lecture"), app.get("unit")), "")
            app["forward_ref"] = appearance_is_forward_ref(txt, terms)

        first_use = None        # earliest non-forward-ref appearance (any role)
        first_use_any = None     # earliest appearance incl. forward-ref
        first_intro = None
        for app in appearances:
            key = (app.get("lecture_order", 0), unit_index(app.get("unit", "")))
            if first_use_any is None or key < first_use_any[0]:
                first_use_any = (key, app)
            if not app.get("forward_ref"):
                if first_use is None or key < first_use[0]:
                    first_use = (key, app)
            if app.get("role") == "introduced":
                if first_intro is None or key < first_intro[0]:
                    first_intro = (key, app)
        if first_use is None:        # all appearances are forward-refs
            first_use = first_use_any

        roles = c.get("roles") or {}
        g.add_node(name,
            canonical_name=name,
            aliases=c.get("aliases") or [],
            definitions=c.get("definitions") or [],
            appearances=appearances,
            roles=roles,
            n_introduced=roles.get("introduced", 0),
            n_used=roles.get("used", 0),
            lecture_introduced=(first_intro[1].get("lecture") if first_intro else None),
            unit_introduced=(first_intro[1].get("unit") if first_intro else None),
            order_introduced=(list(first_intro[0]) if first_intro else None),
            lecture_first_used=(first_use[1].get("lecture") if first_use else None),
            unit_first_used=(first_use[1].get("unit") if first_use else None),
            order_first_used=(list(first_use[0]) if first_use else None),
        )

    for e in normalized.get("prerequisites", []):
        fr, to = e["from"], e["to"]
        if fr not in g or to not in g:
            continue
        g.add_edge(fr, to,
                   reason=e.get("reason", ""),
                   evidence_lecture=e.get("evidence_lecture", ""),
                   evidence_unit=e.get("evidence_unit", ""))
    return g


def graph_to_graphml(g: nx.DiGraph) -> nx.DiGraph:
    h = nx.DiGraph()
    h.graph.update({k: _flatten(v) for k, v in g.graph.items()})
    for n, data in g.nodes(data=True):
        h.add_node(n, **{k: _flatten(v) for k, v in data.items()})
    for u, v, data in g.edges(data=True):
        h.add_edge(u, v, **{k: _flatten(v) for k, v in data.items()})
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / args.dataset
    normalized = json.loads((out_dir / "normalized.json").read_text(encoding="utf-8"))
    units = load_units(out_dir)
    g = build_graph(normalized, units)
    print(f"[{args.dataset}] nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

    graph_json = nx.readwrite.json_graph.node_link_data(g, edges="edges")
    (out_dir / "graph.json").write_text(
        json.dumps(graph_json, indent=1, ensure_ascii=False), encoding="utf-8")
    nx.write_graphml(graph_to_graphml(g), out_dir / "graph.graphml")
    print(f"  -> {out_dir / 'graph.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
