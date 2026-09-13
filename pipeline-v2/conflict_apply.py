"""
Merge authored injection sentences into conflict_specs.json and write the
injected lectures (injections/<id>/<insert_lecture>.md).

Reads the authored sentences from a JSON file with records
{dataset, id, sentence} (the output of the author workflow).

Usage:
  python pipeline-v2/conflict_apply.py --sentences <sentences.json>
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent.parent / "new-results"


def inject(rendered, unit_id, sentence):
    lines = rendered.split("\n")
    out, i, done = [], 0, False
    hdr = re.compile(r"^## Unit (\S+) —")
    while i < len(lines):
        m = hdr.match(lines[i])
        if m and m.group(1) == unit_id:
            out.append(lines[i]); i += 1
            block = []
            while i < len(lines) and not hdr.match(lines[i]):
                block.append(lines[i]); i += 1
            while block and block[-1].strip() == "":
                block.pop()
            block += ["", sentence, ""]
            out += block; done = True; continue
        out.append(lines[i]); i += 1
    if not done:
        raise ValueError(f"unit {unit_id} not found")
    return "\n".join(out)


def mid_unit(units_doc):
    content = [u for u in units_doc["units"] if u["kind"] == "content"]
    return content[len(content) // 2]["unit_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    args = ap.parse_args()
    sents = json.loads(Path(args.sentences).read_text(encoding="utf-8"))
    by = {}
    for s in sents:
        by.setdefault(s["dataset"], {})[s["id"]] = s["sentence"]

    for ds, idmap in by.items():
        root = RESULTS / ds
        specs = json.loads((root / "conflict_specs.json").read_text(encoding="utf-8"))
        inj_dir = root / "injections"
        for s in specs:
            sent = idmap.get(s["id"])
            if not sent:
                print(f"  WARN no sentence for {ds}/{s['id']}"); continue
            s["sentence"] = sent
            lec = s["insert_lecture"]
            rendered = (root / "rendered" / f"{lec}.md").read_text(encoding="utf-8")
            units_doc = json.loads((root / "units" / f"{lec}.json").read_text(encoding="utf-8"))
            unit_id = mid_unit(units_doc)
            s["insert_unit"] = unit_id
            d = inj_dir / s["id"]; d.mkdir(parents=True, exist_ok=True)
            (d / f"{lec}.md").write_text(inject(rendered, unit_id, sent), encoding="utf-8")
        (root / "conflict_specs.json").write_text(
            json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{ds}] applied {len(specs)} conflict injections")


if __name__ == "__main__":
    main()
