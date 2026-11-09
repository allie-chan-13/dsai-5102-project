# DSAI4205 drug reviews

| Path | What it is |
| --- | --- |
| `drug-review/` | Dataset, later methods, and tracked scores in `drug-review/results/` |
| `project-code.ipynb` | Course notebook (EDA, baseline, and the five original models) |
| `project-report.docx`, `project-slides.pdf` | Course writeup. Kept locally and gitignored |

Model checkpoints (`*.pt`, `*.bin`, `*.ckpt`), `.venv/`, and Python caches are gitignored. The metrics CSV and the comparison chart under `drug-review/results/` are kept.

Run the later methods from `drug-review/`:

```bash
cd drug-review
uv sync
uv run python -m drug_review.pipeline
```
