# Identifying structural inconsistencies in lecture courses

The method represents a lecture course as a temporal knowledge graph and identifies four
types of structural inconsistency: use before introduction, prerequisite-order violation,
definition conflict, and orphan concept.

## Contents

- `pipeline-v2/` contains the pipeline: segmentation of lectures into units, validation of
  the extraction output, consensus of the dedicated prerequisite-edge runs, entity
  resolution, graph construction with PageRank, the detectors, controlled injections and
  their scoring, the keyword baseline, and the scoring of the baselines.
- `pipeline-v2/prompts/` contains the prompts of the extraction stage, in which a language
  model extracts concepts and prerequisites from each lecture.
- `pipeline-v2/schemas/` contains the JSON schema that the extraction output is validated
  against.
- `statistics/` recomputes the confidence intervals and exact tests reported in the article
  from the counts in `statistics/counts.json`.

Course materials, pipeline outputs, and annotation results are not included in this version
of the repository.

## Requirements

- Python 3.12
- networkx 3.6
- pypdf 6.14, only to convert PDF lecture notes to Markdown (`pdf_to_markdown_ds2.py`)
- openai and python-dotenv, only for the optional embedding filter of definition conflicts
  (`detect_violations.py --embeddings`, which reads the `OPENAI_API_KEY` environment variable)

The statistics script needs only the Python standard library.

## Usage

The pipeline scripts read course materials from `datasets-ua/` and write their outputs to
`new-results/`, both at the repository root. Each script describes its arguments at the top
of the file.

To recompute the statistics reported in the article:

```
python statistics/compute_statistics.py
```

## License

MIT, see [LICENSE](LICENSE).
