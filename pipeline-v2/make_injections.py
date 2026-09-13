"""
Generate controlled, graded-subtlety injections for the leak-free
injection-recall experiment. Dataset-general: use-before-introduction targets
are auto-selected from the graph, orphan terms come from a per-dataset list of
terms verified absent from the course.

  used_before_introduction
    Insert a genuine USE (no markdown link, so not a forward-ref) of a
    LATE-introduced concept into an EARLIER lecture where it does not already
    appear. The concept is then used before its own introduction. Targets that
    are ALREADY used-before-introduction in the clean course are excluded.

  orphan_concept
    Insert a USE of a NOVEL technical term that is absent from the whole course.

Three subtlety tiers (blatant / medium / subtle) control how explicit the
inserted mention is. A blatant use-before sentence must never define the
concept, or the extractor would record it as introduced.

For each injection an injected copy of one rendered lecture is written to
  injections/<id>/<lecture_stem>.md
and the spec (target, insert location, expected signature) is recorded in
  injections.json

Usage: python pipeline-v2/make_injections.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"

SUBTLETY = ["blatant", "medium", "subtle"]

# Per-dataset configuration.
CONFIG = {
    "dataset-1": {
        "early_orders": [2, 3, 4, 5, 6],
        "late_min": 7,
        "orphan_terms": ["Two-Phase Commit", "Sharding", "Circuit Breaker",
                         "Service Mesh", "Bulkhead", "Leader Election",
                         "Vector Clock", "Write-Ahead Log", "Read-Through Cache"],
    },
    "dataset-3": {
        "early_orders": [2, 3, 4],
        "late_min": 5,
        "orphan_terms": ["гейміфікація", "перевернутий клас", "мікронавчання",
                         "конективізм", "скафолдинг", "таксономія Блума",
                         "адаптивне навчання", "сторітелінг", "фасилітація"],
    },
    "dataset-2": {
        "early_orders": [2, 3, 4, 5, 6],
        "late_min": 9,
        "orphan_terms": ["скінченний автомат", "формальна граматика", "ентропія",
                         "марковський ланцюг", "матроїд", "мережевий потік",
                         "тюрингова машина", "динамічне програмування",
                         "лямбда-числення"],
    },
}


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s'-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def used_before_sentence(concept, tier):
    if tier == "blatant":
        return (f"Тут ми безпосередньо спираємося на {concept}: подальший виклад "
                f"передбачає, що читач уже вільно користується цим механізмом.")
    if tier == "medium":
        return (f"На цьому етапі зручно застосувати {concept}, щоб коректно "
                f"розв'язати описану проблему.")
    return f"(механізм по суті відповідає тому, як працює {concept})"


def orphan_sentence(term, tier):
    if tier == "blatant":
        return (f"Для надійності тут застосовується {term}: він забезпечує "
                f"коректний результат у цьому випадку, тому ми спираємося на "
                f"{term} нижче.")
    if tier == "medium":
        return f"Це тут забезпечує {term}."
    return f"(підхід нагадує {term})"


def inject_into_rendered(rendered, unit_id, sentence):
    lines = rendered.split("\n")
    out, i, injected = [], 0, False
    header_re = re.compile(r"^## Unit (\S+) —")
    while i < len(lines):
        m = header_re.match(lines[i])
        if m and m.group(1) == unit_id:
            out.append(lines[i]); i += 1
            block = []
            while i < len(lines) and not header_re.match(lines[i]):
                block.append(lines[i]); i += 1
            while block and block[-1].strip() == "":
                block.pop()
            block += ["", sentence, ""]
            out += block; injected = True
            continue
        out.append(lines[i]); i += 1
    if not injected:
        raise ValueError(f"unit {unit_id} not found")
    return "\n".join(out)


def mid_unit(units_doc):
    content = [u for u in units_doc["units"] if u["kind"] == "content"]
    return content[len(content) // 2]["unit_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset-1")
    ap.add_argument("--n-used-before", type=int, default=9)
    ap.add_argument("--n-orphan", type=int, default=9)
    args = ap.parse_args()

    root = RESULTS / args.dataset
    cfg = CONFIG[args.dataset]
    inj_dir = root / "injections"
    inj_dir.mkdir(exist_ok=True)
    for old in inj_dir.glob("*/*.md"):
        old.unlink()

    graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    id_by_order = {m["lecture_order"]: m["lecture_id"] for m in manifest}
    viol = json.loads((root / "violations.json").read_text(encoding="utf-8"))
    already_ub = {normalize(v["concept"]) for v in viol["used_before_introduction"]}

    # unit text per lecture for the "concept absent in insert lecture" check
    unit_text = {}
    for f in (root / "units").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        unit_text[d["lecture_id"]] = " ".join(u["text"].lower() for u in d["units"])

    early_ids = [id_by_order[o] for o in cfg["early_orders"] if o in id_by_order]

    # auto-select use-before targets
    ub_targets = []
    for name, node in sorted(nodes.items()):
        oi = node.get("order_introduced")
        if not oi or oi[0] < cfg["late_min"] or not node.get("definitions"):
            continue
        if normalize(name) in already_ub:
            continue
        # earliest valid insert lecture where the concept does not already appear
        ins = None
        for lid in early_ids:
            if order_of[lid] >= oi[0]:
                continue
            if normalize(name) in unit_text.get(lid, ""):
                continue
            ins = lid; break
        if ins:
            ub_targets.append((name, ins))
        if len(ub_targets) >= args.n_used_before:
            break

    specs, idx = [], 0
    for k, (concept, lec) in enumerate(ub_targets):
        tier = SUBTLETY[k % 3]
        idx += 1
        specs.append({"id": f"inj-{idx:03d}", "type": "used_before_introduction",
                      "target": concept, "insert_lecture": lec, "subtlety": tier,
                      "sentence": used_before_sentence(concept, tier),
                      "expected": {"type": "used_before_introduction", "concept": concept}})

    orphan_terms = cfg["orphan_terms"][:args.n_orphan]
    # spread orphan inserts across all lectures
    all_ids = [id_by_order[o] for o in sorted(id_by_order)]
    for k, term in enumerate(orphan_terms):
        lec = all_ids[k % len(all_ids)]
        tier = SUBTLETY[k % 3]
        idx += 1
        specs.append({"id": f"inj-{idx:03d}", "type": "orphan_concept",
                      "target": term, "insert_lecture": lec, "subtlety": tier,
                      "sentence": orphan_sentence(term, tier),
                      "expected": {"type": "orphan_concept", "concept": term}})

    # placebos: already-introduced AND used concepts, recap mention
    placebos = [n for n, nd in nodes.items()
                if (nd.get("roles") or {}).get("introduced", 0) > 0
                and (nd.get("roles") or {}).get("used", 0) > 0]
    placebos = sorted(placebos)[:3]
    for k, concept in enumerate(placebos):
        # insert into a lecture after its introduction
        oi = nodes[concept].get("order_introduced") or [0]
        later = [lid for lid in all_ids if order_of[lid] > oi[0]]
        lec = later[len(later)//2] if later else all_ids[-1]
        idx += 1
        specs.append({"id": f"inj-{idx:03d}", "type": "placebo",
                      "target": concept, "insert_lecture": lec, "subtlety": "n/a",
                      "sentence": f"Як зазначалося раніше, {concept} лишається "
                                  f"важливим і в цьому контексті.",
                      "expected": None})

    for s in specs:
        lec = s["insert_lecture"]
        rendered = (root / "rendered" / f"{lec}.md").read_text(encoding="utf-8")
        units_doc = json.loads((root / "units" / f"{lec}.json").read_text(encoding="utf-8"))
        unit_id = mid_unit(units_doc)
        s["insert_unit"] = unit_id
        injected = inject_into_rendered(rendered, unit_id, s["sentence"])
        d = inj_dir / s["id"]; d.mkdir(exist_ok=True)
        (d / f"{lec}.md").write_text(injected, encoding="utf-8")

    (root / "injections.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ub = sum(1 for s in specs if s["type"] == "used_before_introduction")
    n_or = sum(1 for s in specs if s["type"] == "orphan_concept")
    n_pl = sum(1 for s in specs if s["type"] == "placebo")
    print(f"[{args.dataset}] {len(specs)} injections: {n_ub} used_before, "
          f"{n_or} orphan, {n_pl} placebo")
    for s in specs:
        if s["type"] != "placebo":
            print(f"  {s['id']} {s['type'][:12]:12} {s['subtlety']:8} "
                  f"{s['target'][:30]:30} -> {s['insert_lecture']}")


if __name__ == "__main__":
    main()
