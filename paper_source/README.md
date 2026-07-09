# Paper Source Bundle

This directory is a shareable LaTeX bundle for coauthors.

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_draft.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

The bundle includes the main manuscript, supplement, bibliography, appendix
input, methods/theory input, and figure PDFs used by the manuscript.  The full
Python implementation and experiment artifacts remain in the repository root.
