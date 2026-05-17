# The Spectral Gap-Statement

**When the Negative Subspace of Attention Transport is a Well-Posed Invariant**

Evgeny Vyaltsev ([ORCID 0009-0004-3712-6798](https://orcid.org/0009-0004-3712-6798)) and Daniil Vyaltsev

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20257482.svg)](https://doi.org/10.5281/zenodo.20257482)

---

This repository accompanies the manuscript *The Spectral Gap-Statement:
When the Negative Subspace of Attention Transport is a Well-Posed
Invariant*. It contains the paper source, the compiled PDF, and the
numerical scripts that reproduce every figure and quantitative claim.

## What the paper is about

Attention can be viewed as a convex transport process whose free-energy
Hessian governs an instability of mass transport: a negative eigenvalue
produces an exponentially growing mode. The number of such modes,
`dim E_-`, is used as a routing signal — but a count of negative
eigenvalues is only meaningful if the spectrum separates the negative
modes from the rest. This paper shows that `dim E_-` is a topological
invariant — threshold-free via the Riesz spectral projector, and stable
under perturbations of operator norm below half the spectral gap, by the
Davis–Kahan theorem — exactly when the free-energy Hessian possesses a
gap, `Delta(H) > 0`. The three empirical operating "windows" of the
framework are shown to be three coordinate sections of this single
spectral condition. The geometry is verified end to end: a synthetic
consolidation experiment, a direct geodesic / Jacobi-equation
verification of the curvature formula, and an empirical study on a
trained Llama-3.2-1B.

## Repository contents

```
paper2.pdf              Compiled manuscript (13 pages)
paper2.tex              Manuscript source
section6_body.tex       The Gap-Statement body (\input by paper2.tex)
refs.bib                Bibliography (12 published references)
figures/                Figures used in the paper
  section6_diagnostic.png    Fig. 1 — synthetic 4-panel verification
  geodesic_diagnostic.png    Fig. 2 — geodesic / Jacobi verification
  layer_evolution.png        Fig. 3 — empirical layer-evolution study
scripts/                Numerical scripts (see mapping below)
```

## Script-to-claim mapping

Each script is self-contained and reproduces a specific result.

| Script | Paper section | Reproduces |
|---|---|---|
| `section6_consolidation.py` | §3.7, Fig. 1 | The four-panel synthetic verification: Riesz threshold-independence, Davis–Kahan stability, energy-window caveat, count = unstable-mode count. |
| `geodesic_test.py` | §3.8, Fig. 2 | The Jacobi–Maupertuis curvature formula vs a direct finite-difference reference (median rel. error 3.7e-4), and geodesic-bundle defocusing vs the Jacobi deviation equation. |
| `inertial_test.py` | §3.6 | The inertial-flow integration: count of exponentially unstable modes equals `dim E_-`, at the characteristic-polynomial rates. |
| `gap_test.py` | §3.2–3.5 | The Riesz projector, Davis–Kahan/Weyl stability, and the energy-window caveat on the curvature implication. |
| `general_test2.py` | §3.4 | The universality / window analysis: the confinement sweep and the two-sided temperature window. |
| `sym_check.py` | §3.4 | Control experiment: the high-temperature behaviour for symmetric vs generic queries; demonstrates that an eigenvalue count without a spectral gap is ill-posed. |

The empirical Llama-3.2-1B study (§3.9) was run separately; its capture
and analysis code, raw captures, and per-layer profiles are released as a
distinct artefact (the Phase 2c gap-validation package).

## Requirements

```
python >= 3.10
numpy
matplotlib
```

Install with `pip install numpy matplotlib`. The scripts are pure
NumPy/Matplotlib and run on CPU; no GPU is required.

## Reproducing the figures

```
cd scripts
python section6_consolidation.py    # -> section6_diagnostic.png
python geodesic_test.py             # -> geodesic_diagnostic.png
python inertial_test.py             # -> inertial_diagnostic.png
```

Each script prints its quantitative results to stdout and writes its
figure to the working directory.

## Building the paper

```
pdflatex paper2.tex
bibtex   paper2
pdflatex paper2.tex
pdflatex paper2.tex
```

Requires a standard LaTeX distribution (`amsmath`, `amssymb`, `amsthm`,
`graphicx`, `booktabs`, `hyperref`). Place the figures from `figures/`
alongside `paper2.tex`, or adjust the `\includegraphics` paths.

## Scope and limitations

Stated plainly, as in the manuscript:

- The geodesic verification (§3.8) is on a synthetic two-dimensional
  free-energy landscape; it tests the geometric mechanism, not its
  occurrence in a trained model.
- In two dimensions the sectional curvature is a scalar; the
  higher-dimensional operator case is not verified here.
- The empirical study (§3.9) is a single model (Llama-3.2-1B) on a single
  corpus (WikiText-2); conclusions are stated with that scope.

## Citation

If you use this work, please cite it via its archived DOI:

> Vyaltsev, E. and Vyaltsev, D. (2026). *The Spectral Gap-Statement:
> When the Negative Subspace of Attention Transport is a Well-Posed
> Invariant.* Zenodo. https://doi.org/10.5281/zenodo.20257482

BibTeX:

```bibtex
@misc{vyaltsev2026spectralgap,
  author       = {Vyaltsev, Evgeny and Vyaltsev, Daniil},
  title        = {The Spectral Gap-Statement: When the Negative
                  Subspace of Attention Transport is a
                  Well-Posed Invariant},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20257482},
  url          = {https://doi.org/10.5281/zenodo.20257482}
}
```

A `CITATION.cff` file is also provided; GitHub renders it under
"Cite this repository".

## License

The code in `scripts/` is released under the MIT License (see `LICENSE`).
The manuscript text and figures are released under
Creative Commons Attribution 4.0 (CC BY 4.0).
