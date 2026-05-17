# -*- coding: utf-8 -*-
"""Universality test, v2 (fixed Delta_logit measurement)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import qr, eigvalsh

def reg_simplex(m):
    V = np.eye(m) - np.ones((m, m)) / m
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    C = U[:, :m-1] * S[:m-1]
    return C / np.linalg.norm(C[0] - C[1])

def make_keys(m, d, sk, nc, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    Bq, _ = qr(rng.standard_normal((d, d)))
    cents = scale * reg_simplex(m) @ Bq[:, :m-1].T
    keys, lab = [], []
    for a in range(m):
        keys.append(cents[a] + sk * rng.standard_normal((nc, d))); lab += [a]*nc
    return np.vstack(keys), np.array(lab), cents

def cov_neff_dlog(x, K, lab, m, beta):
    logit = K @ x
    z = beta * logit; z -= z.max()
    p = np.exp(z); p /= p.sum()
    Kc = K - p @ K
    ev = np.sort(np.clip(eigvalsh((Kc * p[:, None]).T @ Kc), 0, None))[::-1]
    P = np.array([p[lab == a].sum() for a in range(m)])
    return ev, 1.0 / np.sum(P**2), np.std(logit)

def queries(cents, m, rng, nr=45):
    qs = []
    for k in range(1, m+1):
        for _ in range(3):
            qs.append(cents[rng.choice(m, k, replace=False)].mean(0))
    for _ in range(nr):
        c = 10**rng.uniform(-1, 1); qs.append(rng.dirichlet(np.full(m, c)) @ cents)
    return np.array(qs)

def safe_r(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return np.corrcoef(a, b)[0, 1]

d, nc, sk, tol = 48, 12, 0.010, 1e-3
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# ---- E1 : confinement sweep ------------------------------------------
print("="*70)
print("E1  -  CONFINEMENT SWEEP   (beta=1) :  Hess V = gamma*I - beta*Cov")
print("="*70)
print(f"{'gamma':>7} | {'slope':>7} | {'intercept':>10} | {'r':>8} | {'mean dimE':>10}")
print("-"*70)
beta = 1.0
gammas = [0.0, 0.02, 0.04, 0.06, 0.10, 0.20]
cols = plt.cm.plasma(np.linspace(0, 0.85, len(gammas)))
for gamma, col in zip(gammas, cols):
    NE, DE = [], []
    for m in range(3, 10):
        for seed in range(4):
            K, lab, cents = make_keys(m, d, sk, nc, seed)
            rng = np.random.default_rng(100 + seed)
            for x in queries(cents, m, rng):
                ev, ne, _ = cov_neff_dlog(x, K, lab, m, beta)
                DE.append(int(np.sum(beta*ev > max(gamma, tol)))); NE.append(ne)
    NE, DE = np.array(NE), np.array(DE)
    s, i = np.polyfit(NE, DE, 1)
    print(f"{gamma:>7.2f} | {s:>7.3f} | {i:>+10.3f} | {safe_r(NE,DE):>8.4f} |"
          f" {DE.mean():>10.2f}")
    ax[0].scatter(NE, DE+np.random.uniform(-.12,.12,len(DE)), s=6, alpha=.3,
                  color=col, label=f"gamma={gamma:g}")
xx = np.linspace(1, 9, 50)
ax[0].plot(xx, xx-1, 'k--', lw=2, label="N_eff - 1")
ax[0].set_xlabel(r"$N_{eff}$"); ax[0].set_ylabel(r"$\dim E_-$")
ax[0].set_title("A.  Clean law survives only gamma < gamma_c (attention edge)")
ax[0].legend(fontsize=7, loc="upper left"); ax[0].grid(alpha=.3)

# ---- E2 : two-sided window, collapse under beta*Delta_logit ----------
print()
print("="*70)
print("E2  -  TWO-SIDED WINDOW  and  beta*Delta_logit COLLAPSE")
print("="*70)
betas = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
for scale, mk, cc in [(1.0,'o','C0'), (2.0,'s','C3')]:
    # mean logit spread over the population
    dl = []
    for m in range(3, 10):
        K, lab, cents = make_keys(m, d, sk, nc, 0, scale)
        rng = np.random.default_rng(7)
        for x in queries(cents, m, rng):
            dl.append(cov_neff_dlog(x, K, lab, m, 1.0)[2])
    Dlog = np.mean(dl)
    slopes, ints, spreads = [], [], []
    for beta in betas:
        NE, DE, sp = [], [], []
        for m in range(3, 10):
            for seed in range(4):
                K, lab, cents = make_keys(m, d, sk, nc, seed, scale)
                rng = np.random.default_rng(100+seed)
                nem = []
                for x in queries(cents, m, rng):
                    ev, ne, _ = cov_neff_dlog(x, K, lab, m, beta)
                    DE.append(int(np.sum(beta*ev > tol))); NE.append(ne); nem.append(ne)
                sp.append(np.ptp(nem))
        NE, DE = np.array(NE), np.array(DE)
        s, i = np.polyfit(NE, DE, 1)
        slopes.append(s); ints.append(i); spreads.append(np.mean(sp))
    print(f"\n  key-scale={scale}  mean Delta_logit = {Dlog:.4f}")
    print(f"  {'beta':>6} {'b*Dlog':>9} {'slope':>8} {'intcpt':>9} {'Neff-spread':>12}")
    for b, s, i, sp in zip(betas, slopes, ints, spreads):
        print(f"  {b:>6.1f} {b*Dlog:>9.3f} {s:>8.3f} {i:>+9.3f} {sp:>12.2f}")
    ax[1].plot(betas*Dlog, slopes, mk+'-', color=cc, label=f"scale={scale:g}")
ax[1].axhline(1, ls='--', color='k', lw=1.2)
ax[1].set_xscale('log'); ax[1].set_xlabel(r"$\beta \cdot \Delta_{logit}$")
ax[1].set_ylabel("fitted slope")
ax[1].set_title("B.  Two scales collapse under beta*Delta_logit")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("general_diagnostic.png", dpi=130)
print("\nfigure saved")
