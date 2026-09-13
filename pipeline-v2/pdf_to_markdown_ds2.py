#!/usr/bin/env python3
"""dataset-2 (discrete math, PDF theory notes) -> structured markdown lectures.

Selects a coherent subset (Module 1: set theory through combinatorics, Тема1.1–1.7),
extracts text with pypdf, and segments each lecture into ## units at reliable
math-text boundaries (numbered subsections like "1.3.1." and definition/theorem/
example markers). Content-preserving: no text is dropped or summarised; only
PDF artefacts (page numbers, end-of-line hyphenation, repeated spaces) are cleaned.

Output: new-results/dataset-2/lecture_notes/NN_temaX_Y.md  (NN = lecture order)
"""
import re, sys
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets-ua" / "dataset-2" / "Theory"
OUT = ROOT / "new-results" / "dataset-2" / "lecture_notes"

# Full course in order: Module 1 (set theory / combinatorics) + Module 2 (graphs)
SELECTED = [
    ("Тема1.1", "Основні положення теорії множин"),
    ("Тема1.2", "Відповідності та відношення"),
    ("Тема1.3", "Відношення еквівалентності"),
    ("Тема1.4", "Відношення порядку"),
    ("Тема1.5", "Функції та їхні властивості"),
    ("Тема1.6", "Вступ у комбінаторику"),
    ("Тема1.7", "Базові комбінаторні алгоритми"),
    ("Тема2.1", "Основні положення теорії графів"),
    ("Тема2.2", "Способи задавання й властивості графів"),
    ("Тема2.3", "Відношення та відображення на графах"),
    ("Тема2.4", "Числа графа"),
    ("Тема2.5", "Дерева та їхні властивості, ліс, цикли"),
    ("Тема2.6", "Обхід графів"),
    ("Тема2.8", "Розфарбування графа"),
    ("Тема2.9", "Основні алгоритми розфарбування графів"),
    ("Тема2.10", "Шляхи та цикли Ейлера. Плоскі та планарні графи"),
]

NUM_HEAD = re.compile(r"^\d+\.\d+(?:\.\d+)?\.?\s+[А-ЯІЇЄҐA-Z]")
KW_HEAD = re.compile(r"^(Визначення|Означення|Теорема|Властивост|Приклад|Наслідок|"
                     r"Лема|Аксіом|Твердження|Алгоритм|Спосіб|Способи|Основні визначення|"
                     r"Основні поняття)\b", re.IGNORECASE)
PAGENUM = re.compile(r"^\s*\d{1,3}\s*$")
FIG = re.compile(r"^\s*Рис\.\s*\d")


def clean_text(t: str) -> list[str]:
    # de-hyphenate words broken across line ends:  "матема-\nтика" -> "математика"
    t = re.sub(r"([а-яіїєґА-ЯІЇЄҐ])-\s*\n\s*([а-яіїєґ])", r"\1\2", t)
    out = []
    for raw in t.split("\n"):
        s = re.sub(r"[ \t]+", " ", raw).strip()
        if not s or PAGENUM.match(s):
            continue
        out.append(s)
    return out


def is_heading(s: str) -> bool:
    if len(s) > 90:
        return False
    return bool(NUM_HEAD.match(s) or KW_HEAD.match(s))


def to_markdown(title_full: str, lines: list[str]) -> str:
    md = [f"# {title_full}", ""]
    # drop a leading repetition of the title line if present
    body = lines[1:] if lines and title_full.split('.')[-1].strip()[:15] in lines[0] else lines
    buf, have_section = [], False

    def flush():
        if buf:
            md.append(" ".join(buf).strip())
            md.append("")
            buf.clear()

    for s in body:
        if FIG.match(s):           # keep figure captions inline as text
            buf.append(s); continue
        if is_heading(s):
            flush()
            md.append(f"## {s}")
            md.append("")
            have_section = True
        else:
            buf.append(s)
    flush()
    return "\n".join(md)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (stem, title) in enumerate(SELECTED, start=1):
        pdf = SRC / f"{stem}.pdf"
        text = "".join(p.extract_text() for p in PdfReader(str(pdf)).pages)
        lines = clean_text(text)
        title_full = f"{stem.replace('Тема', 'Тема ')}. {title}"
        md = to_markdown(title_full, lines)
        n_units = md.count("\n## ")
        fn = OUT / f"{i:02d}_{stem.replace('.', '_')}.md"
        fn.write_text(md, encoding="utf-8")
        print(f"{fn.name}: {len(md):,} chars, {n_units} ## units")


if __name__ == "__main__":
    sys.exit(main())
