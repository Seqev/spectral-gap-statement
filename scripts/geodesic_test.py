# -*- coding: utf-8 -*-
"""G1 (v3): K_JM<0  =>  exponential geodesic defocusing.  Ring of free-energy
centers -> wide K_JM<0 interior; geodesic traverses its diameter along the
x-symmetry-axis, staying inside K_JM<0 throughout."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ang = np.deg2rad(22.5 + 45*np.arange(8))
R, beta, E = 1.8, 2.5, 1.5
centers = R*np.stack([np.cos(ang), np.sin(ang)], 1)

def FE(x):
    diff = x[..., None, :] - centers
    U = 0.5*np.sum(diff*diff, -1)
    w = -beta*U; M = w.max(-1, keepdims=True)
    ew = np.exp(w-M); Z = ew.sum(-1, keepdims=True)
    F = -1.0/beta*(M[..., 0] + np.log(Z[..., 0]))
    p = ew/Z
    cbar = np.einsum('...j,jk->...k', p, centers)
    gF = x - cbar
    cc = centers - cbar[..., None, :]
    Cov = np.einsum('...j,...jk,...jl->...kl', p, cc, cc)
    trH = 2.0 - beta*(Cov[..., 0, 0] + Cov[..., 1, 1])
    return F, gF, trH, np.sum(gF*gF, -1)

def Kfield(x):
    F, gF, trH, gn2 = FE(x)
    h = 2.0*(E-F)
    return trH/h**2 + 2.0*gn2/h**3, h, F, gF

# ---- (1) formula vs direct curvature on {F<E} ------------------------
g = np.linspace(-2.4, 2.4, 280)
X, Y = np.meshgrid(g, g)
Kf, h, F, _ = Kfield(np.stack([X, Y], -1))
mask = h > 0.15
logh = np.where(mask, np.log(np.where(mask, h, 1.0)), np.nan)
dy, dx = np.gradient(logh, g, g)
dyy, _ = np.gradient(dy, g, g); _, dxx = np.gradient(dx, g, g)
Kd = -(dxx+dyy)/(2.0*h)
good = mask.copy()
for sh in (2,-2):
    good &= np.roll(mask,sh,0) & np.roll(mask,sh,1)
rel = np.abs(Kf[good]-Kd[good])/(np.abs(Kd[good])+1e-9)
print("="*68)
print("G1(1)  K_JM formula  vs  direct -Lap(log h)/(2h)")
print(f"  median rel.error={np.median(rel):.2e}  90th pct={np.percentile(rel,90):.2e}")

# ---- (2) geodesic across the K_JM<0 interior -------------------------
xs = np.linspace(-1.7, 1.7, 500)
Kx = np.array([Kfield(np.array([x,0.0]))[0] for x in xs])
neg = np.where(Kx < 0)[0]
xL, xR = xs[neg[0]], xs[neg[-1]]
print(f"\nG1(2)  K_JM<0 segment on x-axis: x in [{xL:.2f},{xR:.2f}]")

def rhs(st):
    x, v = st[:2], st[2:4]
    K, h, F, gF = Kfield(x)
    gp = -gF/h
    dv = -2.0*v*(gp@v) + (v@v)*gp
    return np.array([v[0], v[1], dv[0], dv[1]])

def rk4(st, ds):
    k1=rhs(st); k2=rhs(st+ds/2*k1); k3=rhs(st+ds/2*k2); k4=rhs(st+ds*k3)
    return st + ds/6*(k1+2*k2+2*k3+k4)

def uspeed(x, vdir):
    _, h, _, _ = Kfield(x)
    return vdir/np.linalg.norm(vdir)*np.exp(-0.5*np.log(h))

ds, eps = 0.004, 1e-3
x0 = np.array([xL+0.02, 0.0]); vdir = np.array([1.0, 0.0])
stc = np.concatenate([x0, uspeed(x0, vdir)])
xn0 = x0 + eps*np.array([0.0, 1.0])
stn = np.concatenate([xn0, uspeed(xn0, vdir)])

Cx, Nx, Vc, Kp, S = [stc[:2].copy()], [stn[:2].copy()], [stc[2:4].copy()], [], [0.0]
for _ in range(4000):
    K, h, _, _ = Kfield(stc[:2])
    if K >= 0 or h < 0.2:
        break
    Kp.append(K)
    stc, stn = rk4(stc, ds), rk4(stn, ds)
    Cx.append(stc[:2].copy()); Nx.append(stn[:2].copy()); Vc.append(stc[2:4].copy())
    S.append(S[-1]+ds)
Cx, Nx, Vc = np.array(Cx), np.array(Nx), np.array(Vc)
Kp = np.array(Kp + [Kfield(stc[:2])[0]])
s = np.array(S)

vhat = Vc/np.linalg.norm(Vc, axis=1, keepdims=True)
nhat = np.stack([-vhat[:,1], vhat[:,0]], 1)
phi = 0.5*np.log(np.array([Kfield(p)[1] for p in Cx]))
eta = np.exp(phi)*np.sum((Nx-Cx)*nhat, 1)             # signed JM separation

# Jacobi eqn  J'' + K_JM J = 0 , ICs from eta's first two points
J, Jp = eta[0], (eta[1]-eta[0])/ds
Jarr = [J]
for i in range(len(s)-1):
    Km, Kn = Kp[i], Kp[i+1]; Kmid = 0.5*(Km+Kn)
    def f(y, k): return np.array([y[1], -k*y[0]])
    y = np.array([J, Jp])
    k1=f(y,Km); k2=f(y+ds/2*k1,Kmid); k3=f(y+ds/2*k2,Kmid); k4=f(y+ds*k3,Kn)
    y = y + ds/6*(k1+2*k2+2*k3+k4)
    J, Jp = y; Jarr.append(J)
Jarr = np.array(Jarr)
cum = np.concatenate([[0], np.cumsum(np.sqrt(np.abs(0.5*(Kp[:-1]+Kp[1:])))*ds)])
wkb = eta[0]*(np.abs(Kp[0])/np.abs(Kp))**0.25*np.exp(cum)

print(f"  arc length s in [0,{s[-1]:.2f}],  K_JM along path in "
      f"[{Kp.min():.3f},{Kp.max():.3f}]  (all<0)")
print(f"  {'s':>6} {'eta measured':>14} {'Jacobi eqn':>13} {'WKB':>12}")
for k in range(0, len(s), max(1,len(s)//8)):
    print(f"  {s[k]:>6.2f} {eta[k]:>14.4e} {Jarr[k]:>13.4e} {wkb[k]:>12.4e}")
print(f"  amplification: measured {eta[-1]/eta[0]:.1f}x  Jacobi {Jarr[-1]/Jarr[0]:.1f}x")
print(f"  measured vs Jacobi: max rel.deviation = "
      f"{np.max(np.abs(eta-Jarr)/np.abs(eta)):.2%}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
Kpl = np.where(good, Kf, np.nan); vm = np.nanmax(np.abs(Kpl))
cf = ax[0].contourf(X, Y, Kpl, levels=np.linspace(-vm, vm, 25), cmap='RdBu_r')
ax[0].contour(X, Y, Kpl, levels=[0], colors='k', linewidths=1)
plt.colorbar(cf, ax=ax[0], label=r'$K_{JM}$')
ax[0].plot(Cx[:,0], Cx[:,1], 'k-', lw=3, label='JM geodesic')
ax[0].scatter(centers[:,0], centers[:,1], c='yellow', edgecolor='k', s=70,
              zorder=5, label='free-energy centers')
ax[0].set_title("A.  Geodesic across the $K_{JM}<0$ interior")
ax[0].legend(fontsize=8, loc='upper right'); ax[0].set_aspect('equal')
ax[1].semilogy(s, np.abs(eta), 'o', ms=3, color='C0', label='measured bundle')
ax[1].semilogy(s, np.abs(Jarr), '-', color='C3', lw=2, label="Jacobi $J''+K_{JM}J=0$")
ax[1].semilogy(s, wkb, ':', color='C2', lw=2, label=r'WKB $|K|^{-1/4}e^{\int\!\sqrt{|K|}}$')
ax[1].set_xlabel('JM arc length $s$'); ax[1].set_ylabel('perpendicular separation')
ax[1].set_title("B.  Negative curvature -> exponential defocusing")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("geodesic_diagnostic.png", dpi=130)
print("\nfigure saved")
