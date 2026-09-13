"""
Generate controlled prerequisite-ordering injections (the 4th detector type),
graded by subtlety, for the leak-free frozen-extraction recall experiment.

A prerequisite-ordering violation is: a prerequisite A of a concept C is
INTRODUCED later than C is introduced. To inject one cleanly under the
frozen-extraction protocol we manipulate exactly ONE lecture — the EARLY
lecture where the dependent C is introduced — and add a sentence that:

  * states explicitly that C depends on / requires A (so the extractor emits a
    prerequisite edge A -> C, with both endpoints in `concepts` as the prompt
    demands), and
  * names A as required prior knowledge WITHOUT defining it (so A stays role
    "used" here and its real, later introduction in a frozen lecture is what
    fixes its introduction order).

After re-extraction of that one lecture: A is introduced late (frozen lecture),
C is introduced early (manipulated lecture), edge A -> C exists, therefore
A_intro > C_intro and the prerequisite_order detector flags (A, C).

Leak-free target selection:
  * C introduced in an early lecture, with a definition (genuinely introduced).
  * A introduced in a late lecture (order >= late_min), with a definition.
  * order_introduced(A) > order_introduced(C).
  * (A, C) is NOT already an edge and NOT already a prerequisite_order
    violation in the clean course.
  * A's surface name does not already occur in C's lecture (the mention is new).

Placebos inject a VALID prerequisite (a genuinely earlier-introduced A'),
which must NOT produce a violation — the negative control / noise floor.

Writes:
  injections/<id>/<lecture_stem>.md   (one injected lecture per injection)
  prereq_specs.json                   (targets, insert location, expected pair)

Usage: python pipeline-v2/make_prereq_injections.py --dataset dataset-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"

SUBTLETY = ["blatant", "medium", "subtle"]

CONFIG = {
    "dataset-1": {"early_orders": [2, 3, 4, 5, 6], "late_min": 8},
    "dataset-3": {"early_orders": [2, 3, 4], "late_min": 5},
}


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s'-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prereq_sentence(dep, prereq, tier):
    """dep = C (introduced here, early); prereq = A (introduced later)."""
    if tier == "blatant":
        return (f"Перш ніж переходити до «{dep}», необхідно вже володіти поняттям "
                f"«{prereq}»: саме «{prereq}» — це те знання, без якого «{dep}» "
                f"зрозуміти неможливо, тож воно є обов'язковою передумовою цього "
                f"матеріалу.")
    if tier == "medium":
        return (f"Щоб коректно застосувати «{dep}», читач має спершу опанувати "
                f"«{prereq}».")
    return f"Виклад «{dep}» передбачає попереднє знайомство з «{prereq}»."


def placebo_sentence(dep, prereq):
    return (f"Виклад «{dep}» спирається на «{prereq}», розглянуту в попередніх "
            f"лекціях.")


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
    ap.add_argument("--n", type=int, default=9)
    args = ap.parse_args()

    root = RESULTS / args.dataset
    cfg = CONFIG[args.dataset]
    inj_dir = root / "injections"
    inj_dir.mkdir(exist_ok=True)
    for old in inj_dir.glob("po-inj-*/*.md"):
        old.unlink()

    graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    order_of = {m["lecture_id"]: m["lecture_order"] for m in manifest}
    id_by_order = {m["lecture_order"]: m["lecture_id"] for m in manifest}
    viol = json.loads((root / "violations.json").read_text(encoding="utf-8"))
    existing_po = {(normalize(v["prerequisite"]), normalize(v["dependent"]))
                   for v in viol["prerequisite_order"]}

    unit_text = {}
    for f in (root / "units").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        unit_text[d["lecture_id"]] = " ".join(u["text"].lower() for u in d["units"])

    def intro_order(name):
        oi = nodes[name].get("order_introduced")
        return oi[0] if oi else None

    early_orders = set(cfg["early_orders"])
    # candidate dependents C: introduced early, with definition
    deps = sorted(name for name, n in nodes.items()
                  if n.get("definitions") and intro_order(name) in early_orders)
    # candidate prerequisites A: introduced late, with definition
    late = sorted((name for name, n in nodes.items()
                   if n.get("definitions") and (intro_order(name) or 0) >= cfg["late_min"]),
                  key=lambda nm: intro_order(nm))

    # build targets, one distinct dependent lecture preferred, distinct concepts
    targets = []
    used_deps, used_preqs, used_lecs = set(), set(), set()
    for dep in deps:
        c_lec = nodes[dep].get("lecture_introduced")
        c_ord = intro_order(dep)
        if dep in used_deps:
            continue
        # prefer spreading across lectures, but allow reuse if needed later
        for prereq in late:
            if prereq in used_preqs or prereq == dep:
                continue
            a_ord = intro_order(prereq)
            if a_ord is None or a_ord <= c_ord:
                continue
            if (prereq, dep) in edges or (dep, prereq) in edges:
                continue
            if (normalize(prereq), normalize(dep)) in existing_po:
                continue
            if normalize(prereq) in unit_text.get(c_lec, ""):
                continue
            # spread: skip a 2nd target in a lecture we already used until we have variety
            if c_lec in used_lecs and len(used_lecs) < len(early_orders):
                continue
            targets.append((prereq, dep, c_lec))
            used_deps.add(dep); used_preqs.add(prereq); used_lecs.add(c_lec)
            break
        if len(targets) >= args.n:
            break

    specs, idx = [], 0
    for k, (prereq, dep, c_lec) in enumerate(targets):
        tier = SUBTLETY[k % 3]
        idx += 1
        specs.append({
            "id": f"po-inj-{idx:03d}", "type": "prerequisite_order",
            "prerequisite": prereq, "dependent": dep,
            "insert_lecture": c_lec, "subtlety": tier,
            "sentence": prereq_sentence(dep, prereq, tier),
            "expected": {"type": "prerequisite_order",
                         "prerequisite": prereq, "dependent": dep},
        })

    # placebos: valid prerequisite (A' introduced strictly earlier than C')
    earliest = sorted((name for name, n in nodes.items()
                       if n.get("definitions") and intro_order(name) is not None),
                      key=lambda nm: intro_order(nm))
    placebos = []
    used_pl_dep = set()
    for dep in deps:
        if dep in used_deps or dep in used_pl_dep:
            continue
        c_lec = nodes[dep].get("lecture_introduced")
        c_ord = intro_order(dep)
        for prereq in earliest:
            a_ord = intro_order(prereq)
            if a_ord is None or a_ord >= c_ord or prereq == dep:
                continue
            if (normalize(prereq), normalize(dep)) in existing_po:
                continue
            if normalize(prereq) in unit_text.get(c_lec, ""):
                continue
            placebos.append((prereq, dep, c_lec))
            used_pl_dep.add(dep)
            break
        if len(placebos) >= 3:
            break
    for prereq, dep, c_lec in placebos:
        idx += 1
        specs.append({
            "id": f"po-inj-{idx:03d}", "type": "placebo",
            "prerequisite": prereq, "dependent": dep,
            "insert_lecture": c_lec, "subtlety": "n/a",
            "sentence": placebo_sentence(dep, prereq),
            "expected": None,
        })

    # render injected lectures
    for s in specs:
        lec = s["insert_lecture"]
        rendered = (root / "rendered" / f"{lec}.md").read_text(encoding="utf-8")
        units_doc = json.loads((root / "units" / f"{lec}.json").read_text(encoding="utf-8"))
        unit_id = mid_unit(units_doc)
        s["insert_unit"] = unit_id
        injected = inject_into_rendered(rendered, unit_id, s["sentence"])
        d = inj_dir / s["id"]; d.mkdir(exist_ok=True)
        (d / f"{lec}.md").write_text(injected, encoding="utf-8")

    (root / "prereq_specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_po = sum(1 for s in specs if s["type"] == "prerequisite_order")
    n_pl = sum(1 for s in specs if s["type"] == "placebo")
    print(f"[{args.dataset}] {len(specs)} injections: {n_po} prerequisite_order, "
          f"{n_pl} placebo")
    for s in specs:
        tag = s["subtlety"] if s["type"] != "placebo" else "PLACEBO"
        print(f"  {s['id']} {tag:8} «{s['prerequisite']}» (поз.{intro_order(s['prerequisite'])}) "
              f"-> передумова для «{s['dependent']}» (поз.{intro_order(s['dependent'])}) "
              f"@ {s['insert_lecture']}")


if __name__ == "__main__":
    main()
