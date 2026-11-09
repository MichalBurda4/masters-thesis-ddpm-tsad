# Overleaf setup (AGH WI thesis)

This folder is an **AGH WI** diploma thesis skeleton (`agh-wi` class), not the old `report`-class draft.

## Upload

1. Create a new Overleaf project → **Upload Project** (zip this folder) **or** sync from Git.
2. Ensure the project root contains at least:
   - `main.tex`
   - `agh-wi.cls`
   - `bibliography.bib`
   - `tex/` (chapter includes)
   - `img/logo_podstawowe.png` (optional branding asset; title-page AGH mark is drawn by TikZ in the class — if missing, copy from the Downloads AGH template `img/` folder)
3. Set **Main document** to `main.tex`.

## Compiler settings

| Setting | Value |
|--------|--------|
| Compiler | **pdfLaTeX** |
| Bibliography | **Biber** (Menu → Settings → Compiler → Bibliography: Biber) |
| TeX Live | default Overleaf image is fine |

Compile sequence Overleaf runs for you: `pdflatex` → `biber` → `pdflatex` ×2.

## Class options

In `main.tex`:

```latex
\documentclass[english]{agh-wi}           % screen / English body
% \documentclass[english,print]{agh-wi}   % print margins
% \documentclass[english,ds]{agh-wi}      % Data Science programme label
% \documentclass[english,umisi]{agh-wi}   % UMISI programme label
```

Default programme line without `ds`/`umisi` is **Informatyka**. Confirm with the dean's office which label belongs on your title page for *Artificial Intelligence and Data Analysis*.

The English title page shows **Faculty of Computer Science**, the Polish diploma title (`\titlePL`), and the English title (`\titleEN`) underneath.

**Local-only full draft:** the filled thesis text lives in `../thesis_local/` (gitignored). Keep expanding chapters there; commit only the public skeleton under `thesis/` until you decide to publish the full manuscript.

## Intentionally omitted (simpler Overleaf)

- **minted** (needs `-shell-escape` + Pygments)
- **nomencl** (needs `makeindex`)

You can re-add them later like the official AGH demo if needed. `listings` and `algorithm2e` are loaded; uncomment list-of-algorithms / list-of-listings in `main.tex` when you use them.

## Local compile (optional)

```bash
pdflatex main
biber main
pdflatex main
pdflatex main
```

## Obsolete files

The previous `chapters/` + `report`-class `main.tex` layout was replaced by `tex/` + `agh-wi`. Do not mix both structures in one Overleaf project.
