# SkillForge ICLR 2027 paper draft

This directory is a self-contained, anonymous, Overleaf-compatible paper project. It uses the official ICLR 2027 style files downloaded from the conference author-guideline page.

## Open in Overleaf

1. In Overleaf, choose **New Project → Upload Project**.
2. Upload `skillforge_iclr2027_overleaf.zip`.
3. Set the main document to `main.tex` if Overleaf does not select it automatically.
4. Use pdfLaTeX. No shell escape or external figure files are required.

## Local build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Structure

- `main.tex`: title, abstract, statements, bibliography, and section assembly.
- `sections/`: complete main paper and appendices.
- `figures/`: TikZ/PGFPlots source; figures remain editable in Overleaf.
- `tables/`: primary result tables.
- `references.bib`: literature used in the draft.
- `LITERATURE_AUDIT.md`: closest-work matrix, primary links, and explicit novelty boundaries.
- `SUBMISSION_CHECKLIST.md`: anonymization and missing-evidence checklist.
- `iclr2027_conference.*`, `natbib.sty`, `fancyhdr.sty`: unmodified official style dependencies.

## Submission status

This is a research-complete **initial draft**, not a submission-ready claim set. Before submission, the authors should independently verify every citation and number, replace the anonymous code placeholder, add multi-seed and multi-benchmark evidence listed in Appendix I, confirm the AI-use statement, and re-check the 9-page main-text limit against the current official guidelines. Keep `\iclrfinalcopy` commented for review.
