"""
Stage 1 (pipeline v2): deterministic segmentation of Ukrainian markdown
lectures (dataset-1, software-architecture course) into ordered text units.

A unit is an h2 (`## `) section of the lecture markdown. Code fences are
respected: a `## ` line inside a ``` fence does not start a new unit.

Per lecture:
  new-results/dataset-1/units/<lecture_id>.json    structured units
  new-results/dataset-1/rendered/<lecture_id>.md   canonical markdown for
                                                   the extractor agents
Course-level:
  new-results/dataset-1/manifest.json              lecture-order manifest

Usage:
  python segment_lectures.py            # default dataset-1 paths
  python segment_lectures.py --src ... --out ...
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT / "datasets-ua" / "dataset-1" / "lecture_notes"
DEFAULT_OUT = PROJECT / "new-results" / "dataset-1"

TOC_HEADINGS = {"зміст", "зміст лекції", "план", "план лекції"}
SOURCE_HEADINGS = {"джерела", "література", "посилання", "додаткові матеріали",
                   "список джерел", "список літератури"}

FENCE_RE = re.compile(r"^\s*(```|~~~)")
H1_RE = re.compile(r"^#\s+(.*)$")
H2_RE = re.compile(r"^##\s+(.*)$")


def classify_heading(heading: str) -> str:
    h = heading.strip().lower().rstrip(":")
    if h in TOC_HEADINGS:
        return "toc"
    if h in SOURCE_HEADINGS:
        return "sources"
    return "content"


def segment_file(path: Path) -> dict:
    stem = path.stem                      # e.g. "02-value-objects" or "Konspekt_Lek3"
    m = re.search(r"(\d+)", stem)
    if not m:
        return None                        # not a numbered lecture (e.g. an aggregate file)
    lecture_order = int(m.group(1))
    lecture_num = f"{lecture_order:02d}"

    lines = path.read_text(encoding="utf-8").splitlines()

    title = None
    units = []
    current = None          # dict with heading/kind/lines
    in_fence = False

    def close_current():
        nonlocal current
        if current is None:
            return
        text = "\n".join(current["lines"]).strip()
        # strip leading/trailing horizontal rules
        text = re.sub(r"^(---+\s*\n?)+", "", text)
        text = re.sub(r"(\n?---+\s*)+$", "", text)
        text = text.strip()
        units.append({
            "heading": current["heading"],
            "kind": current["kind"],
            "text": text,
        })
        current = None

    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current is not None:
                current["lines"].append(line)
            continue
        if not in_fence:
            m1 = H1_RE.match(line)
            if m1 and not line.startswith("##"):
                if title is None:
                    title = m1.group(1).strip()
                    continue
            m2 = H2_RE.match(line)
            if m2 and not line.startswith("###"):
                close_current()
                heading = m2.group(1).strip()
                current = {
                    "heading": heading,
                    "kind": classify_heading(heading),
                    "lines": [],
                }
                continue
        if current is not None:
            current["lines"].append(line)
        # text before the first ## (other than the title) is dropped only if
        # blank; otherwise keep it as a preamble unit
        elif line.strip():
            current = {"heading": "(преамбула)", "kind": "content", "lines": [line]}

    close_current()

    if in_fence:
        print(f"  WARNING: unclosed code fence in {path.name}", file=sys.stderr)

    # assign ids in document order
    for i, u in enumerate(units, start=1):
        u["unit_id"] = f"{lecture_num}#{i:02d}"
        u["order"] = i

    return {
        "lecture_id": stem,
        "lecture_order": lecture_order,
        "title": title or stem,
        "source_file": path.name,
        "units": units,
    }


def render_for_extraction(lec: dict) -> str:
    """Canonical markdown the extraction agents read (content units only)."""
    out = [f"# Лекція {lec['lecture_order']}: {lec['title']}",
           f"_Source: {lec['source_file']}_", ""]
    for u in lec["units"]:
        if u["kind"] != "content":
            continue
        out.append(f"## Unit {u['unit_id']} — {u['heading']}")
        out.append("")
        out.append(u["text"])
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    units_dir = args.out / "units"
    rendered_dir = args.out / "rendered"
    units_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for path in sorted(args.src.glob("*.md")):
        lec = segment_file(path)
        if lec is None:
            print(f"  skip (no lecture number): {path.name}")
            continue
        (units_dir / f"{lec['lecture_id']}.json").write_text(
            json.dumps(lec, ensure_ascii=False, indent=1), encoding="utf-8")
        rendered = render_for_extraction(lec)
        (rendered_dir / f"{lec['lecture_id']}.md").write_text(
            rendered, encoding="utf-8")

        content_units = [u for u in lec["units"] if u["kind"] == "content"]
        manifest.append({
            "lecture_id": lec["lecture_id"],
            "lecture_order": lec["lecture_order"],
            "title": lec["title"],
            "n_units": len(lec["units"]),
            "n_content_units": len(content_units),
            "content_chars": sum(len(u["text"]) for u in content_units),
        })
        print(f"{lec['lecture_id']}: {len(lec['units'])} units "
              f"({len(content_units)} content, "
              f"{manifest[-1]['content_chars']:,} chars)")

    manifest.sort(key=lambda r: r["lecture_order"])
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nManifest: {args.out / 'manifest.json'} ({len(manifest)} lectures)")


if __name__ == "__main__":
    main()
