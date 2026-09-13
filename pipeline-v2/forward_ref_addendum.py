#!/usr/bin/env python3
"""Stage the forward-reference-before-definition concepts (currently suppressed
from used-before detection) as annotation-ready cards, so the instructor can
judge them at the frozen run. See new-results/FORWARD_REF_FINDINGS.md.

These are concepts whose earliest pre-introduction appearance is a markdown
cross-link `[concept](concept.md)`; the production detector suppresses them, but
the instructor's criterion treats a forward-reference use before the definition
as a real defect (CAP confirmed real).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline-v2"))
from detect_violations import earliest_used, order_of, _unit_idx  # noqa: E402


def load_lut(dsdir):
    lut = {}
    for f in sorted((dsdir / "units").glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        lid = doc.get("lecture_id") or f.stem
        for u in doc["units"]:
            lut[(lid, u["unit_id"])] = {"heading": u.get("heading", ""), "text": u.get("text", "")}
    return lut


def main():
    ds = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "dataset-1"
    dsdir = ROOT / "new-results" / ds
    g = json.loads((dsdir / "graph.json").read_text(encoding="utf-8"))
    lut = load_lut(dsdir)
    cards = []
    for n in g["nodes"]:
        d = n
        intro = order_of(d, "introduced")
        if not intro:
            continue
        eu_skip = earliest_used(d, skip_forward_ref=True)[0]
        eu_all, eu_app = earliest_used(d, skip_forward_ref=False)
        if not (eu_all and eu_all < intro) or (eu_skip and eu_skip < intro):
            continue
        name = n["canonical_name"]
        fr_before = [a for a in d["appearances"]
                     if a.get("forward_ref")
                     and (a.get("lecture_order", 0), _unit_idx(a.get("unit", ""))) < intro]
        first_fr = sorted(fr_before, key=lambda a: (a.get("lecture_order", 0), _unit_idx(a.get("unit", ""))))[0] if fr_before else eu_app
        use_l, use_u = first_fr.get("lecture"), first_fr.get("unit")
        intro_l, intro_u = d.get("lecture_introduced"), d.get("unit_introduced")
        cards.append({
            "id": f"fwd-{len(cards):04d}",
            "claim_type": "used_before_introduction",
            "concept": name,
            "title": f"«{name}» вживається (через forward-посилання) ще до введення",
            "claim": (f"Концепт «{name}» вперше з'являється у {use_l} / {use_u} як "
                      f"forward-посилання (markdown-лінк на майбутню лекцію), а формально "
                      f"вводиться лише у {intro_l} / {intro_u}. Поточний детектор глушить такі "
                      f"forward-посилання; за критерієм викладача це дефект (як CAP)."),
            "evidence": [
                {"label": f"Forward-посилання на «{name}»", "lecture": use_l, "unit": use_u,
                 "text": (lut.get((use_l, use_u), {}).get("text", "") or "")[:500]},
                {"label": f"Де «{name}» формально вводиться", "lecture": intro_l, "unit": intro_u,
                 "text": (lut.get((intro_l, intro_u), {}).get("text", "") or "")[:500]},
            ],
        })
    out = dsdir / "forward_ref_addendum.json"
    out.write_text(json.dumps({"dataset": ds, "note": "forward-ref-before-definition; pending instructor annotation at the frozen run",
                               "n_cards": len(cards), "items": cards}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{ds}] wrote {len(cards)} forward-ref addendum cards -> {out}")
    for c in cards:
        print("  ", c["concept"])


if __name__ == "__main__":
    sys.exit(main())
