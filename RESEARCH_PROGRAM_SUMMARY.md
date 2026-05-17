# Research Program Summary — Spectral Geometry of Attention

**Status: closed.** This document records the full arc of the research
program behind *The Spectral Gap-Statement*, including the falsification
results that closed its central hypothesis. It is written so that anyone
arriving at this repository sees the complete trajectory — the surviving
result and the refuted ones — rather than only the published positive.

---

## The hypothesis

The program began with a strong hypothesis: that the spectral geometry of
attention — specifically the free-energy Hessian `H = -beta * Cov_p(k)` and
its dominant negative eigenmode — is a *core computational mechanism* of
the transformer, not merely a description of it. The hope was a "new
principle of computation": attention as controlled anisotropic transport
instability.

## The arc — five stages of scrutiny

The hypothesis was tested in five stages. Each stage strengthened the
criterion rather than weakening it; this is the discipline the program
held to throughout.

**Stage 1 — Operator formalization.**
The Gap-Statement was made rigorous: the negative-eigenvalue count
`dim E_-` is a well-posed topological invariant (threshold-free via the
Riesz projector, perturbation-stable via Davis–Kahan) exactly inside the
gap-regime `Delta(H) > 0`. This is a correct contribution-by-synthesis of
classical results. **It survived — because it is correct mathematics.**

**Stage 2 — Empirical study on a trained model.**
On Llama-3.2-1B, the gap-regime was found to hold for ~96% of attention
heads, with an explosive transition at the first layer and a typical head
carrying one dominant mode. A real, measurable observation — **on one
model.**

**Stage 3 — Cross-model stress test.**
The U-shaped depth profile of spectral concentration was tested on five
architectures (GPT-2 small/medium, BERT, Pythia-1.4B, Llama-3.2-1B). It
**did not reproduce**: five models gave five different depth patterns. The
U-profile is Llama-specific, not an architectural invariant. (INS-33)

**Stage 4 — Random-control test of the surviving "structure".**
The 3-metric coupling (`chi`, `S_lambda`, `R`, Spearman |rho| ~ 0.7–0.97
across all models) was tested against a random-spectrum control. Random
spectra produced the same low-dimensional coupling. The coupling is an
**algebraic identity** — three monotone functions of spectral peakedness —
not a property of trained transformers.

**Stage 5 — Matched-control causal intervention.**
The decisive test. The dominant negative mode of `Cov_p(k)` was ablated
under a matched-magnitude control (suppress a random bulk mode of equal
perturbation norm). Pre-registered criterion: null → close the direction.
Result: **null.** The treatment-minus-control NLL differential was
+2.7e-4, 95% CI bracketing zero, Wilcoxon p = 0.26. Suppressing the
dominant mode harms next-token prediction no more than suppressing a
random mode of equal size. (INS-34, `SPECTRAL_ABLATION_REPORT.md`)

## What survived

- **The Gap-Statement** as an operator-theoretic formalization: the
  well-posedness of `dim E_-` in the gap-regime. Published, v1.1, DOI
  10.5281/zenodo.20257482. This is correct and stands.
- **The empirical observation** that trained attention heads on
  Llama-3.2-1B are spike-dominated and typically in the gap-regime —
  stated honestly as a single-model observation, not a law.
- **The causal-control methodology**: a matched-magnitude intervention
  framework for testing spectral causality without scale-asymmetry
  artifacts.

## What was refuted

- A universal U-shaped depth profile (Stage 3).
- `chi`-type spectral-summary invariants (refuted earlier; the candidate
  invariant `I` and the constant-`chi` hypothesis both failed).
- The 3-metric coupling as a non-trivial structure — it is algebraic
  (Stage 4).
- **The central hypothesis**: that the dominant spectral mode is a
  privileged computational object. The causal test (Stage 5) places this
  below the threshold of detectability. The transport geometry is real as
  a *description* but causally inert as a *mechanism* — a decorative
  reparametrization, in the precise sense the experiment was designed to
  detect.

## What this program is

Not a discovery of a new computational principle — that hypothesis was
tested and did not survive. What it is: a complete, honest falsification
cycle. A beautiful mechanistic narrative was formalized, stress-tested
across models, controlled against random baselines, and subjected to a
matched causal intervention. It failed at the causal stage — which is
exactly the stage at which a genuinely mechanistic claim must survive if
it is real.

The value of the program is methodological: it is a worked example of how
spectral-mechanistic narratives in deep learning can be made rigorous and
then honestly falsified, and of the controls (matched-magnitude
intervention, random-spectrum baselines) required to avoid the false
positives such narratives are prone to.

## A note on what was deliberately not pursued

After the Stage 5 null, the Bayesian weight on "spectral geometry is a
core computational mechanism" is low. Remaining untested corners (the gap
itself versus the mode, bulk spread, other architectures, long-context
retrieval heads) exist, but pursuing them as attempts to revive the
central hypothesis would be poor discipline — the causal stage is
precisely where the hypothesis should have survived if it were real. The
program is therefore closed as a hypothesis about computational mechanism.

The honest residual question — *why* trained attention produces
spike-dominated Hessians — is largely answered by an elementary fact:
`H = -beta Cov_p(k)`, and the softmax weights `p` of a trained model are
concentrated, so the covariance under a concentrated measure is
spike-dominated almost by construction. This belongs to the study of
optimization-induced anisotropy, not to a theory of transport computation.
