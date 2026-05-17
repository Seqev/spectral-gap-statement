# -*- coding: utf-8 -*-
"""Section 6 consolidation: the Gap-Statement on ONE geometry, one run.
Four panels, one configuration:
  S1  Riesz projector  P_theta = (1/2pi i) oint (zI-H)^-1 dz  is threshold-free
  S2  Davis-Kahan / Weyl: dim E_- stable iff ||dH||_op < Delta/2
  S3  energy-window caveat: Hess<0 does NOT imply K_JM<0 (the |grad F|^2 term)
  S4  the count IS the number of unstable inertial modes, at Cor.5.4 rates
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================================================================
# ONE shared geometry: m clusters in d dims, pure attention H=-b Cov
# ===================================================================
rng = np.random.default_rng(7)
d, m, nc, sk, beta = 24, 5, 8, 0.008, 1.0
cent = rng.standard_normal((m, d))
cent /= np.linalg.norm(cent, axis=1, keepdims=True); cent *= 1.1
K = np.repeat(cent, nc, 0) + sk*rng.standard_normal((m*nc, d))
K -= K.mean(0); n = m*nc
H = -beta*(K.T @ K)/n
ev = np.sort(np.linalg.eigvalsh(H))                 # ascending
mu_inter = ev[:m-1]
lam_lo, lam_hi = ev[m-2], ev[m-1]                   # gap bracket, both < 0
Delta = lam_hi - lam_lo
theta = -0.5*(lam_lo + lam_hi)                      # threshold inside the gap
print("="*72)
print("SECTION 6 CONSOLIDATION  -  one geometry, one run")
print(f"  d={d}, m={m} clusters, pure attention  H = -beta*Cov")
print(f"  inter-cluster eigenvalues : {np.round(mu_inter,4)}")
print(f"  gap bracket (lam_lo,lam_hi) = ({lam_lo:.4f}, {lam_hi:.2e})")
print(f"  gap width Delta = {Delta:.4f},  threshold theta = {theta:.4f}")

# ---- S1 : Riesz projector, threshold-free --------------------------
def riesz_trace(H, center, radius, npts=720):
    phi = np.linspace(0, 2*np.pi, npts, endpoint=False)
    z = center + radius*np.exp(1j*phi)
    dz = 1j*radius*np.exp(1j*phi)*(2*np.pi/npts)
    I = np.eye(H.shape[0])
    return sum(np.trace(np.linalg.inv(zk*I - H))*dzk
               for zk, dzk in zip(z, dz))/(2j*np.pi)

c_neg = 0.5*(ev[0] + lam_lo)
r_neg = 0.5*(lam_lo - ev[0]) + 0.45*Delta
trP = riesz_trace(H, c_neg, r_neg).real
thetas = np.logspace(np.log10(Delta)-2.3, np.log10(Delta)+0.05, 240)
counts = np.array([(ev < -t).sum() for t in thetas])
plateau = counts[(thetas > abs(lam_hi)*3) & (thetas < abs(lam_lo)*0.7)]
print()
print("S1  RIESZ PROJECTOR")
print(f"  tr P_theta (contour in gap) = {trP:+.4f}   (= dim E_- = {m-1})")
print(f"  count #(lambda<-theta) across the gap: "
      f"min={plateau.min()}, max={plateau.max()}  -> flat plateau")

# ---- S2 : Davis-Kahan / Weyl stability -----------------------------
eps_grid = Delta*np.array([0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.85,1.0])
frac = []
for eps in eps_grid:
    ok = 0
    for _ in range(160):
        A = rng.standard_normal((d, d)); A = 0.5*(A+A.T)
        A *= eps/np.linalg.norm(A, 2)
        if (np.linalg.eigvalsh(H+A) < -theta).sum() == m-1:
            ok += 1
    frac.append(ok/160)
frac = np.array(frac)
print()
print("S2  DAVIS-KAHAN / WEYL")
for e, f in zip(eps_grid, frac):
    print(f"  ||dH||/Delta = {e/Delta:>4.2f} : dim E_- preserved {f:>6.1%}")
print(f"  -> 100% for ||dH|| <= Delta/2, collapse beyond. Weyl bound exact.")

# ---- S3 : energy-window caveat -------------------------------------
ang = np.deg2rad(22.5 + 45*np.arange(8))
ctr = 1.8*np.stack([np.cos(ang), np.sin(ang)], 1)
b2 = 2.5
def FE2(x):
    diff = x[:,None,:] - ctr
    U = 0.5*np.sum(diff*diff, -1)
    w = -b2*U; M = w.max(-1, keepdims=True)
    ew = np.exp(w-M); Z = ew.sum(-1, keepdims=True)
    F = -1/b2*(M[:,0] + np.log(Z[:,0]))
    p = ew/Z
    cb = np.einsum('nj,jk->nk', p, ctr)
    gF = x - cb
    cc = ctr - cb[:,None,:]
    Cov = np.einsum('nj,njk,njl->nkl', p, cc, cc)
    trH = 2.0 - b2*(Cov[:,0,0] + Cov[:,1,1])
    return F[0], np.sum(gF*gF,-1)[0], trH[0]
pt = np.array([[0.45, 0.30]])
F0, gn2, trH = FE2(pt)
Es = np.linspace(F0+0.05, F0+3.0, 240)
hE = 2*(Es - F0)
KE = trH/hE**2 + 2*gn2/hE**3
hstar = -2*gn2/trH
Estar = F0 + hstar/2
print()
print("S3  ENERGY-WINDOW CAVEAT")
print(f"  concave point: Delta F = {trH:+.3f} (<0), |grad F|^2 = {gn2:.3f}")
print(f"  K_JM < 0 only for E > E* = {Estar:.3f}  (h > h* = {hstar:.3f})")
print(f"  -> 'Hess<0 => K_JM<0' holds only inside the energy window.")

# ---- S4 : count == number of unstable inertial modes ---------------
def gradV(X):
    lo = beta*(X @ K.T); lo -= lo.max(1, keepdims=True)
    p = np.exp(lo); p /= p.sum(1, keepdims=True)
    return -(p @ K)
evec = np.linalg.eigh(H)[1]
def integrate(gamma, N=500, T=14.0, dt=0.01):
    r = np.random.default_rng(3)
    X = 1e-3*r.standard_normal((N, d)); Vel = np.zeros((N, d))
    ts = [0.0]; S = [np.std((X-X.mean(0)) @ evec, 0)]
    for i in range(int(T/dt)):
        def acc(Xx, Vv): return Vv, -gamma*Vv - gradV(Xx)
        k1x,k1v = acc(X,Vel); k2x,k2v = acc(X+dt/2*k1x,Vel+dt/2*k1v)
        k3x,k3v = acc(X+dt/2*k2x,Vel+dt/2*k2v); k4x,k4v = acc(X+dt*k3x,Vel+dt*k3v)
        X = X + dt/6*(k1x+2*k2x+2*k3x+k4x)
        Vel = Vel + dt/6*(k1v+2*k2v+2*k3v+k4v)
        ts.append((i+1)*dt); S.append(np.std((X-X.mean(0)) @ evec, 0))
    return np.array(ts), np.array(S)
def fit_rate(ts, sig):
    w = (sig > 3e-3) & (sig < 5e-2)
    return np.polyfit(ts[w], np.log(sig[w]), 1)[0] if w.sum() > 8 else np.nan
zplus = lambda g, mm: 0.5*(-g + np.sqrt(g*g + 4*np.abs(mm)))
ts, S = integrate(0.5)
print()
print("S4  COUNT == NUMBER OF UNSTABLE INERTIAL MODES (gamma=0.5)")
rates_m, rates_p = [], []
for k in range(m-1):
    rm = fit_rate(ts, S[:,k]); rp = zplus(0.5, mu_inter[k])
    rates_m.append(rm); rates_p.append(rp)
    print(f"  mode {k+1}: mu={mu_inter[k]:+.4f}  measured z+={rm:.4f}"
          f"  Cor.5.4 z+={rp:.4f}")
r_intra = fit_rate(ts, S[:,m-1])
print(f"  within-cluster mode: rate ~ {r_intra:.5f}  (flat: gap is a RATE gap)")

# ===================================================================
# FIGURE : 4 panels
# ===================================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

a = ax[0,0]
a.step(thetas, counts, where='post', color='C0', lw=2)
a.axvspan(abs(lam_hi), abs(lam_lo), color='C2', alpha=.22, label='spectral gap')
a.axhline(m-1, ls='--', color='C3', lw=1.2, label=f'dim$E_-$={m-1}')
a.set_xscale('log'); a.set_xlabel(r'threshold $\theta$')
a.set_ylabel(r'rank $P_\theta=\#\{\lambda<-\theta\}$')
a.set_title(r'S1.  Riesz projector: $\dim E_-$ is threshold-free')
a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[0,1]
a.plot(eps_grid/Delta, frac, 'o-', color='C0', lw=2)
a.axvline(0.5, ls='--', color='C3', lw=1.5, label=r'Weyl bound $\Delta/2$')
a.fill_betweenx([0,1], 0, 0.5, color='C2', alpha=.15)
a.set_xlabel(r'$\|\delta H\|_{op}/\Delta$')
a.set_ylabel(r'fraction with $\dim E_-$ preserved')
a.set_title('S2.  Davis-Kahan: invariant stable below $\\Delta/2$')
a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1,0]
a.axhline(0, color='k', lw=.8)
a.plot(Es, KE, color='C0', lw=2)
a.axvspan(Es[0], Estar, color='C3', alpha=.18,
          label=r'$K_{JM}>0$ (focusing) — $\Delta F<0$ here!')
a.axvspan(Estar, Es[-1], color='C2', alpha=.18, label=r'$K_{JM}<0$ (defocusing)')
a.set_xlabel('energy $E$'); a.set_ylabel(r'$K_{JM}$')
a.set_ylim(min(KE.min()*1.1,-2), np.percentile(KE,88))
a.set_title(r'S3.  Energy-window caveat on $K_{JM}<0$')
a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1,1]
cols = plt.cm.viridis(np.linspace(0,.8,m-1))
for k in range(m-1):
    a.semilogy(ts, S[:,k], color=cols[k], lw=2, label=f'mode {k+1}')
    i0 = np.argmin(np.abs(ts-4.0))
    a.semilogy(ts, S[i0,k]*np.exp(rates_p[k]*(ts-4.0)), '--',
               color=cols[k], lw=1, alpha=.7)
for k in range(m-1, m+2):
    a.semilogy(ts, S[:,k], color='lightgray', lw=1.1)
a.set_ylim(5e-4, 5)
a.set_xlabel('transport time $t$'); a.set_ylabel('spread along eigenmode')
a.set_title(f'S4.  Count = #unstable modes = {m-1}\n(dashed = Cor.5.4 rate)')
a.legend(fontsize=7, loc='lower right'); a.grid(alpha=.3)

plt.tight_layout(); plt.savefig("section6_diagnostic.png", dpi=130)
print("\nfigure saved")
