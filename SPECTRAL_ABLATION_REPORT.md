# Causal Spectral Ablation of the Dominant Attention Mode

**Date:** 2026-05-17
**Model:** `meta-llama/Llama-3.2-1B`, eager attention, bf16 weights with fp32 spectral math (INS-28).
**One-sentence answer:** **NULL.** Suppressing the dominant eigenmode of `Cov_p(k)` hurts task performance (NLL) **no more than** suppressing a random bulk eigenmode of the same Frobenius perturbation. The spectral-causality research direction is **closed**, as pre-registered.

---

## Pre-registered outcome criteria (recorded before any data was seen)

The task brief fixed two outcomes:

> **Outcome 1 — null.** The treatment effect is statistically
> indistinguishable from the control effect. Interpretation: the
> dominant spectral mode is *not* a privileged computational object;
> the transport geometry is a decorative reparametrization. **This
> closes the spectral-causality direction.** It must be reported
> plainly as such, not explained away.

> **Outcome 2 — positive.** The treatment effect significantly exceeds
> the control effect.

The data fall under **Outcome 1**. They are reported below honestly,
without effort to reach for Outcome 2.

---

## Methodology

### Intervention construction (`_apply_ablation` in `attention.py`)

For each chosen (layer, head), at the **last query position** of each
prompt, **modify `K_h` along an eigenvector of `Cov_p(k)`** while
preserving two controls:

1. **Matched Frobenius perturbation.**
   `||K_h - K_h_new||_F` is held **equal** between treatment and
   control. The construction:

   ```
   p     = softmax(beta * K_h @ q_last)        beta = 1/sqrt(D)
   K_c   = K_h - p @ K_h
   Cov_p = K_c^T diag(p) K_c                   [D, D]
   eigh(Cov_p) → eigvals (asc), eigvecs cols

   v_1   = eigvecs[:, -1]                      (dominant)
   v_j   = eigvecs[:, j]                       j random bulk, deterministic seed
   M_T   = ||K_h @ v_1||
   M_C   = ||K_h @ v_j||
   T     = min(M_T, M_C)                       MATCHED TARGET

   alpha = T / M_active                        in (0, 1]
   K_h_new = K_h - alpha * (K_h @ v_active) ⊗ v_active
   ```

2. **Mass preservation.** The softmax over `K_h_new` still sums to 1
   automatically — no manual mass correction needed.

The `||K_h||_F` Frobenius preservation that the brief also suggested is
**dropped** in favour of matched perturbation, because the two cannot
be simultaneously satisfied without breaking the matched-magnitude
control. The drop in `||K_h||_F` is reported per-call in the ablation
log (`norm_K_frac_drop`); on this run it is bounded and small.

### A diagnostic-first detour

An earlier construction (project out `v`, then **rescale** `K_h_new`
to preserve `||K_h||_F`) produced treatment perturbations 10–30 × larger
than control perturbations in Frobenius norm. A naive
treatment-greater-than-control result there would have been pure scale
asymmetry, not a causal claim. Per INS-27 (diagnostic-first), the
matched-magnitude check was run on the smoke data; it failed; the
construction was rewritten before the full run, not after. The current
construction's matched-magnitude ratio is **exactly 1.000** for every
intervened pair.

### Treatment / control / baseline

- **Baseline:** no ablation, production forward path. `_ablation_enabled` is False.
- **Treatment:** the active eigenmode is `v_1` (the dominant).
- **Control:** the active eigenmode is `v_j`, with `j` drawn from a
  bulk index in `[0, D-1)` using
  `RandomState(seed + layer_idx*1000 + head)` — deterministic across
  all prompts for one pair, different across pairs.

All three forwards use exactly the same prompts and the same model
state at start of forward (no cache reuse — single-shot prefill at
`N_kv = 256`).

### Sampling

12 `(layer, head)` pairs spanning **three depth bands** (early L0–L4,
middle L5–L10, late L11–L15) × **four `chi` quantiles** (0.1, 0.4, 0.6,
0.9 of the within-band `chi_median` distribution), drawn from the
Llama-3.2-1B rows of the prior `xmodel_uprofile` study with
`relgap > 0.05` filter (so the dominant mode is well-defined). The 12
pairs collectively cover `chi_median ∈ [0.29, 0.87]`.

### Measurement

Per `(pair, prompt)`:
- Baseline forward → logits → per-token NLL, per-token log-probs.
- Treatment forward → same → KL(baseline ‖ treatment) per token.
- Control forward → same → KL(baseline ‖ control) per token.

Aggregated to per-prompt scalars: mean NLL per condition, mean KL,
median KL, p95 KL.

Differential effect per row: `(treat - baseline) - (ctrl - baseline)` for NLL;
`KL_treat - KL_ctrl` for KL.

### Statistical test

Per-row bootstrap 95% CI (5 000 resamples), paired Wilcoxon
signed-rank test on the `(treat - ctrl)` distribution.

---

## Results

### Overall differential effect

| Metric | Treatment − Baseline | Control − Baseline | Differential (T−C) | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|
| Δ NLL  | 5.95e-04 | 3.23e-04 | **+2.72e-04** | [−1.66e-04, 6.96e-04] (brackets 0) | **p = 0.26** |
| KL     | 1.31e-03 | 7.81e-04 | **+5.32e-04** | [4.32e-04, 6.45e-04] | p = 3e-39 |

n = 300 (12 pairs × 25 prompts).

### Per layer band

| band | n | Δ NLL_diff | CI95 NLL | KL_diff | CI95 KL |
|---|---|---|---|---|---|
| early  | 100 | −2.11e-04 | [−1.18e-03, +7.85e-04] | 8.50e-04 | [5.86e-04, 1.16e-03] |
| middle | 100 | +5.48e-04 | [+2.90e-05, +1.09e-03] | 2.39e-04 | [1.80e-04, 2.99e-04] |
| late   | 100 | +4.79e-04 | [−1.25e-04, +1.13e-03] | 5.08e-04 | [3.98e-04, 6.30e-04] |

### Per `chi` band

| chi-band | n | range | NLL_diff | KL_diff |
|---|---|---|---|---|
| low    | 75 | [−∞, 0.570] | +3.42e-04 | 1.53e-04 |
| med-lo | 75 | [0.570, 0.647] | −1.78e-05 | 5.22e-04 |
| med-hi | 75 | [0.647, 0.775] | +8.99e-05 | 1.10e-03 |
| high   | 75 | [0.775, +∞] | +6.73e-04 | 3.55e-04 |

### Matched-magnitude sanity (per (layer, head))

`||δK_treatment||_F / ||δK_control||_F = 1.000` for every one of the
12 pairs (verified in `ablation_logs.json`). The intervention is honestly
controlled.

---

## Interpretation

Two observations and the verdict.

**1. NLL — the operationally meaningful test — is null.**
Treatment − control NLL is +2.72e-04 nats per token with 95 % CI
[−1.66e-04, +6.96e-04] (brackets zero) and Wilcoxon
*p* = 0.26. By the pre-registered criterion, this is **Outcome 1
(null)**: suppressing the dominant mode hurts task performance no
more than suppressing a random bulk mode of equal Frobenius
perturbation.

**2. KL — the distributional-sensitivity probe — is detectably non-zero.**
Treatment − control KL is +5.32e-04 nats with *p* < 1e-38. So
ablating the dominant mode *does* change the output distribution
detectably more than ablating a bulk mode. But the absolute scale
is tiny (≈ 5e-4 nats per token, set against baseline NLL ≈ 2.5–3 nats),
and it does not translate into a worse prediction (NLL null).

**The combination is the "decorative reparametrization" reading
the task brief warned about**: the dominant spectral mode is real and
geometrically distinguishable, but its existence does not buy
computation — the model is not relying on it for next-token
prediction. KL sees a small distributional shimmer; the loss does not.

By the pre-registered logic, this is **Outcome 1 (null) → the
spectral-causality direction is CLOSED.** Reporting otherwise would
require treating a KL shimmer of 5e-4 nats as a functional finding,
which it is not.

---

## Honest scope limits

- **One model, Llama-3.2-1B.** A larger model (7B+) or a different
  family might rely on the dominant mode in ways not visible here.
  This experiment does not generalise beyond Llama-3.2-1B.
- **WikiText-2 next-token prediction only.** A targeted
  routing/retrieval probe (induction heads, copy heads) was not built
  for this run — the task allowed perplexity + KL as the minimum, and
  that is what was measured. A clean retrieval probe could in
  principle separate functional vs decorative more sharply; the
  current data say only that *generic next-token NLL* is null.
- **12 pairs, 25 prompts each.** Sample sizes are adequate for the
  per-row Wilcoxon (n = 300) but the per-band slices are smaller. The
  middle-band differential is just above zero with a positive lower CI
  bound — a possible weak signal in middle-layer heads alone. It does
  not change the overall null verdict but invites a follow-up if the
  spectral-causality direction is ever revisited.
- **N_kv = 256.** Long-context heads (the `dim E_-` p95 = 62 phenomenon
  at N_kv = 8192 from the prior INS-32 study) are not probed here.
- **`||K_h||_F` not exactly preserved.** Reported per-call as
  `norm_K_frac_drop` in `ablation_logs.json`; bounded by `alpha(2-alpha)
  M_target^2 / ||K_h||_F^2` which on this run is small (sub-percent
  on the typical (alpha, M, ||K||) range we hit).

---

## Tools and artifacts

| File | Role |
|---|---|
| `dcr_attention/models/llama/attention.py` | adds `_ablation_enabled` (off by default), `_apply_ablation`, `enable_ablation` / `disable_ablation` / `get_ablation_log` |
| `benchmarks/spectral_ablation/run_ablation.py` | runs baseline + treatment + control × pairs × prompts; logs per-call eigvals + δ_F |
| `benchmarks/spectral_ablation/analyze_ablation.py` | bootstrap CI, Wilcoxon, per-band aggregates, verdict |
| `benchmarks/spectral_ablation/results/rows.jsonl` | 300 per-row records |
| `benchmarks/spectral_ablation/results/ablation_logs.json` | per-call diagnostic log (eigvals, alpha, δ_F, norm fractions) |
| `benchmarks/spectral_ablation/results/summary.json` | aggregated stats + verdict |
| `benchmarks/spectral_ablation/results/ABLATION_SUMMARY.md` | human-readable summary |
| `benchmarks/spectral_ablation/results/scatter_treatment_vs_control_NLL.png` | scatter ΔNLL_T vs ΔNLL_C |
| `benchmarks/spectral_ablation/results/bar_KL_by_band.png` | KL bar chart by layer band |
| `benchmarks/spectral_ablation/results/matched_magnitude.png` | ratio T/C of `||δK||_F` per pair (all 1.000) |

---

## Verification

- `pytest tests/ --collect-only` count: **610** (unchanged from before this work).
- `tests/dispatcher/` — 54/54 passing (12 gap-theory tests still green).
- `DCRLlamaAttention._ablation_enabled` default: **False** (off by default).
- `DCRLlamaAttention._capture_qk` default: **False** (untouched by this work).
- Production forward path: byte-identical when `_ablation_enabled = False`.

---

## Wall time

12 pairs × 25 prompts × 3 forwards on Llama-3.2-1B at N_kv = 256: **52 seconds total**, plus model load. Cheap to reproduce; cheap to extend.

---

## What this closes, and what it does not

**Closes:**
- The hypothesis that the dominant negative eigenmode of `Cov_p(k)` is
  itself a privileged computational object whose removal selectively
  damages next-token prediction. On Llama-3.2-1B / WikiText-2 it does
  not.

**Does not close:**
- The hypothesis that *some* spectral feature carries computation. The
  experiment tested only the **dominant** mode. Other spectral
  features (the bulk's collective spread, the gap itself, the
  eigenvalue ratio) are not refuted by this experiment.
- The hypothesis that the dominant mode matters on **other**
  architectures or **other** scales.
- Any task other than next-token prediction. A clean retrieval probe
  was outside the time budget for this run.

The right reading of this report is that **this specific spectral-
causality claim, in its strongest form on Llama-3.2-1B, does not
survive a matched-control causal test.** That is information.
