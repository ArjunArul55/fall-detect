# Documentation map

Use this directory for material prepared for review and presentation. The raw
SisFall dataset, processed model inputs, and model result artifacts stay under
their existing top-level paths so the training and evaluation scripts continue
to work without changing their data contracts. The large raw dataset is
excluded from the VS Code Explorer view and is not tracked in Git.

## Start here

- [Presentation guide](PRESENTATION_GUIDE.md) — recommended talk sequence,
  demo order, and key points to explain.
- [Project status report](reports/PROJECT_STATUS_REPORT.md) — detailed
  artifact and metric inventory.
- [End-to-end project report](reports/project_report/PROJECT_REPORT.md) —
  report with generated figures.
- [Technical appendix](reports/technical_appendix/REVIEW2_REPORT.md) —
  supporting tables, plots, and traceable source artifacts.
- Severity K=2/K=3 diagnostic outputs are written to
  `reports/severity_comparison/`.

## Presentation figures

- `presentation/figures/severity/` — severity-clustering and severity-wise
  evaluation plots.
- `presentation/figures/personalization/` — FiLM personalization results and
  architecture plots. This is a research prototype separate from the primary
  CNN-LSTM detector.

## Rebuild reports and figures

Run these commands from the repository root after the relevant data and result
artifacts have been generated:

```powershell
.\.venv\Scripts\python.exe scripts\generate_project_status_report.py
.\.venv\Scripts\python.exe scripts\generate_project_report.py
.\.venv\Scripts\python.exe scripts\generate_requested_plots.py
.\.venv\Scripts\python.exe scripts\plot_film_personalization_results.py
```

The report and figure scripts write into this `docs/` tree. Training and
evaluation outputs continue to use the existing `data/processed/` and
`results/` paths.
