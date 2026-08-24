# Overleaf synchronization

The GitHub `main` branch is the complete research repository. The live Overleaf working tree is replaced by a manuscript-only projection assembled from `.overleaf-export`, so analysis code, data, and caches do not appear in the project files. Overleaf prohibits force pushes, so historical commits from before this split remain in its Git history.

## Included material

- `theSpaceOfConcerns.tex`
- `theSpaceOfConcerns.bib`
- `theSpaceOfConcerns.bbl` when present
- `fun_article.cls`
- PDF files under `figures/`

Change `.overleaf-export` when another manuscript dependency is required. Do not add scripts or data merely to make them accessible through Overleaf.

## Preview the projection

```bash
tools/push_overleaf.sh --dry-run
```

## Publish it

```bash
tools/push_overleaf.sh --push
```

Publishing checks out the current Overleaf `main`, replaces its files with the projection, creates a commit on top of the remote head, and pushes it as a fast-forward update. This is compatible with Overleaf's prohibition on force pushes.

This is intentionally a one-way workflow. Treat GitHub `main` as the source of truth. Retrieve and reconcile any edits made only in Overleaf before publishing again, or they will be replaced.
