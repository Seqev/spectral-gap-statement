# Spectral ablation — analysis summary

## Overall differential effect (treatment − control)

- mean Δ NLL (treatment − baseline): 5.9505e-04
- mean Δ NLL (control − baseline):   3.2322e-04
- **mean differential NLL**: 2.7184e-04  95% CI -1.6621e-04 .. 6.9573e-04  Wilcoxon p=2.558e-01
- mean KL treatment: 1.3137e-03
- mean KL control:   7.8141e-04
- **mean differential KL**: 5.3232e-04  95% CI 4.3182e-04 .. 6.4462e-04  Wilcoxon p=2.941e-39


## Per layer band

| band | n | mean ΔNLL_diff | CI95 NLL | mean KL_diff | CI95 KL |
|---|---|---|---|---|---|
| early | 100 | -2.114e-04 | [-1.185e-03, 7.850e-04] | 8.497e-04 | [5.856e-04, 1.161e-03] |
| late | 100 | 4.785e-04 | [-1.252e-04, 1.131e-03] | 5.079e-04 | [3.975e-04, 6.299e-04] |
| middle | 100 | 5.484e-04 | [2.900e-05, 1.092e-03] | 2.393e-04 | [1.797e-04, 2.993e-04] |

## Per chi band

| chi-band | n | range | mean ΔNLL_diff | mean KL_diff |
|---|---|---|---|---|
| low | 75 | [-inf, 0.570] | 3.424e-04 | 1.533e-04 |
| med-lo | 75 | [0.570, 0.647] | -1.776e-05 | 5.224e-04 |
| med-hi | 75 | [0.647, 0.775] | 8.986e-05 | 1.099e-03 |
| high | 75 | [0.775, inf] | 6.729e-04 | 3.546e-04 |

## Matched-magnitude sanity

| L H | ||δK_T||_F | ||δK_C||_F | ratio T/C |
|---|---|---|---|
| L0H27 | 2.013e+01 | 2.013e+01 | 1.000 |
| L4H23 | 3.395e+01 | 3.395e+01 | 1.000 |
| L1H17 | 4.028e+01 | 4.028e+01 | 1.000 |
| L3H3 | 3.016e+01 | 3.016e+01 | 1.000 |
| L9H24 | 5.458e+01 | 5.458e+01 | 1.000 |
| L10H22 | 3.149e+01 | 3.149e+01 | 1.000 |
| L9H5 | 1.529e+01 | 1.529e+01 | 1.000 |
| L10H29 | 2.674e+01 | 2.674e+01 | 1.000 |
| L13H22 | 9.318e+00 | 9.318e+00 | 1.000 |
| L14H14 | 3.551e+01 | 3.551e+01 | 1.000 |
| L15H26 | 3.913e+01 | 3.913e+01 | 1.000 |
| L12H11 | 2.643e+01 | 2.643e+01 | 1.000 |

## VERDICT: **NULL**

Pre-registered NULL on the operationally meaningful test (NLL). Treatment-minus-control NLL: mean=2.718e-04, 95% CI [-1.662e-04, 6.957e-04] brackets zero, Wilcoxon p=2.558e-01. KL-divergence is detectably different (diff=5.323e-04, p=2.941e-39) but absolute scale is tiny — the dominant mode shifts the output distribution measurably without harming task performance, i.e. it is causally inert for next-token prediction. **Spectral-causality direction CLOSED.**
