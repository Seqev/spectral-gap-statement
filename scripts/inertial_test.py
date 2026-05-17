# -*- coding: utf-8 -*-
"""G2: closing the loop. dim E_- (negative Hessian eigenvalues of the
attention potential) == number of exponentially unstable modes of the
inertial transport flow  x'' + gamma x' = -grad V(x), at the rates
predicted by Corollary 5.4:  z+ = 1/2(-gamma + sqrt(gamma^2 + 4|mu|))."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
d, m, nc, sk, beta = 24, 5, 8, 0.008, 1.0
cent = rng.standard_normal((m, d))
cent /= np.linalg.norm(cent, axis=1, keepdims=True); cent *= 1.1
K = np.repeat(cent, nc, 0) + sk*rng.standard_normal((m*nc, d))
K -= K.mean(0)                       # center -> x*=0 is an exact equilibrium
n = m*nc

def gradV(X):                        # X:(N,d) -> -sum_j p_j k_j
    lo = beta*(X @ K.T); lo -= lo.max(1, keepdims=True)
    p = np.exp(lo); p /= p.sum(1, keepdims=True)
    return -(p @ K)

H = -beta*(K.T @ K)/n                # Hess V(0) = -beta Cov_uniform(k)
ev, evec = np.linalg.eigh(H)         # ascending: most negative first
mu = ev[:m-1]                        # m-1 inter-cluster (largest |mu|)
print("="*68)
print(f"d={d}, m={m} clusters  ->  predicted dim E_- = m-1 = {m-1}")
print(f"  inter-cluster Hessian eigenvalues mu : {np.round(mu,4)}")
print(f"  within-cluster eigenvalues (sample)  : {np.round(ev[m-1:m+2],6)}")

zplus = lambda g, mm: 0.5*(-g + np.sqrt(g*g + 4*np.abs(mm)))

def integrate(gamma, N=600, T=15.0, dt=0.01, seed=7):
    r = np.random.default_rng(seed)
    X = 1e-3*r.standard_normal((N, d)); Vel = np.zeros((N, d))
    ts = [0.0]; S = [np.std((X-X.mean(0)) @ evec, 0)]
    for i in range(int(T/dt)):
        def acc(Xx, Vv): return Vv, -gamma*Vv - gradV(Xx)
        k1x,k1v = acc(X, Vel)
        k2x,k2v = acc(X+dt/2*k1x, Vel+dt/2*k1v)
        k3x,k3v = acc(X+dt/2*k2x, Vel+dt/2*k2v)
        k4x,k4v = acc(X+dt*k3x,   Vel+dt*k3v)
        X   = X   + dt/6*(k1x+2*k2x+2*k3x+k4x)
        Vel = Vel + dt/6*(k1v+2*k2v+2*k3v+k4v)
        ts.append((i+1)*dt); S.append(np.std((X-X.mean(0)) @ evec, 0))
    return np.array(ts), np.array(S)          # S:(steps,d) std along each evec

def fit_rate(ts, sig):
    w = (sig > 3e-3) & (sig < 0.05)            # linear, post-transient window
    if w.sum() < 10: return np.nan
    return np.polyfit(ts[w], np.log(sig[w]), 1)[0]

# ---- G2a : the count, at gamma = 0.5 ---------------------------------
print()
print("G2a  COUNT + RATES   (gamma = 0.5)")
ts, S = integrate(0.5)
print(f"  {'mode':>22} {'measured rate':>14} {'Cor.5.4  z+':>13}")
for k in range(m-1):
    rm = fit_rate(ts, S[:, k])
    print(f"  unstable evec {k+1:<2d} mu={mu[k]:+.4f} {rm:>14.4f}"
          f" {zplus(0.5,mu[k]):>13.4f}")
for k in [m-1, m]:
    rm = fit_rate(ts, S[:, k])
    print(f"  within-cl evec {k+1:<2d} mu={ev[k]:+.5f}"
          f" {('%.4f'%rm) if not np.isnan(rm) else '   ~0 (flat)':>14}"
          f" {zplus(0.5,ev[k]):>13.5f}")
grew = np.sum(S[len(ts)//2] / S[0] > 5.0)
print(f"  directions amplified >5x at mid-run: {grew}   "
      f"(predicted dim E_- = {m-1})")

# ---- G2b : damping sweep vs Corollary 5.4 ----------------------------
print()
print("G2b  DOMINANT RATE vs DAMPING   (Corollary 5.4 curve)")
gammas = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
meas = []
for g in gammas:
    tg, Sg = integrate(g)
    rr = fit_rate(tg, Sg[:, 0])
    meas.append(rr)
    print(f"  gamma={g:>4.2f}  measured z+={rr:.4f}   "
          f"Cor.5.4 z+={zplus(g,mu[0]):.4f}")
meas = np.array(meas)
print(f"  gamma=0 endpoint: measured {meas[0]:.4f}  vs  "
      f"sqrt(|mu_max|)={np.sqrt(abs(mu[0])):.4f}  (= the JM-geodesic rate)")

# ---- figure ----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cols = plt.cm.viridis(np.linspace(0, .8, m-1))
for k in range(m-1):
    ax[0].semilogy(ts, S[:, k], color=cols[k], lw=2,
                   label=f'unstable mode {k+1}')
    t0 = 4.0
    i0 = np.argmin(np.abs(ts-t0))
    ax[0].semilogy(ts, S[i0,k]*np.exp(zplus(0.5,mu[k])*(ts-t0)), '--',
                   color=cols[k], lw=1, alpha=.7)
for k in [m-1, m, m+1]:
    ax[0].semilogy(ts, S[:, k], color='lightgray', lw=1.2)
ax[0].set_ylim(5e-4, 5)
ax[0].set_xlabel('transport time $t$'); ax[0].set_ylabel('cloud spread along eigenmode')
ax[0].set_title(f"A.  {m-1} modes amplify (= dim $E_-$); within-cluster flat\n"
                "dashed = Corollary 5.4 rate")
ax[0].legend(fontsize=7, loc='lower right'); ax[0].grid(alpha=.3)
gg = np.linspace(0, 4, 100)
ax[1].plot(gg, zplus(gg, mu[0]), '-', color='C3', lw=2,
           label=r'Cor. 5.4  $z_+=\frac{1}{2}(-\gamma+\sqrt{\gamma^2+4|\mu|})$')
ax[1].plot(gammas, meas, 'o', ms=8, color='C0', label='measured (nonlinear flow)')
ax[1].scatter([0],[np.sqrt(abs(mu[0]))], marker='*', s=200, color='gold',
              edgecolor='k', zorder=5, label=r'$\sqrt{|\mu|}$ = JM-geodesic rate')
ax[1].set_xlabel(r'damping $\gamma$'); ax[1].set_ylabel('dominant instability rate')
ax[1].set_title("B.  Transport instability rate obeys the\ncharacteristic polynomial")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("inertial_diagnostic.png", dpi=130)
print("\nfigure saved")
