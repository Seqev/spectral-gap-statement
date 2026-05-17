# -*- coding: utf-8 -*-
"""Is the high-beta breakdown a real failure of the pointwise law,
or a genericity/sampling effect? Probe SYMMETRIC queries (exact k-cluster
barycenters) where, by simplex symmetry, attention stays balanced at any beta."""
import numpy as np
from numpy.linalg import qr, eigvalsh

def reg_simplex(m):
    V = np.eye(m)-np.ones((m,m))/m
    U,S,_ = np.linalg.svd(V, full_matrices=False)
    C = U[:,:m-1]*S[:m-1]
    return C/np.linalg.norm(C[0]-C[1])

def make_keys(m,d,sk,nc,seed):
    rng=np.random.default_rng(seed)
    Bq,_=qr(rng.standard_normal((d,d)))
    cents=reg_simplex(m)@Bq[:,:m-1].T
    keys,lab=[],[]
    for a in range(m):
        keys.append(cents[a]+sk*rng.standard_normal((nc,d))); lab+=[a]*nc
    return np.vstack(keys),np.array(lab),cents

def probe(x,K,lab,m,beta):
    z=beta*(K@x); z-=z.max(); p=np.exp(z); p/=p.sum()
    Kc=K-p@K
    ev=np.sort(np.clip(eigvalsh((Kc*p[:,None]).T@Kc),0,None))[::-1]
    P=np.array([p[lab==a].sum() for a in range(m)])
    return 1.0/np.sum(P**2), ev

d,nc,sk,tol=48,12,0.010,1e-3
m=7
K,lab,cents=make_keys(m,d,sk,nc,0)
print("="*64)
print(f"SYMMETRIC-QUERY CHECK  (m={m}, exact k-cluster barycenters)")
print("="*64)
for beta in [1.0,4.0,16.0,64.0,256.0]:
    print(f"\n  beta={beta:g}")
    print(f"  {'k':>3} {'N_eff':>8} {'dim E_-':>9} {'curv.mag (b*lam_max)':>22}")
    for k in range(2,m+1):
        x=cents[:k].mean(0)
        ne,ev=probe(x,K,lab,m,beta)
        dimE=int(np.sum(beta*ev>tol))
        print(f"  {k:>3} {ne:>8.3f} {dimE:>9d} {beta*ev[0]:>22.4f}")
print()
print("  If dim E_- = k-1 holds at ALL beta for symmetric queries,")
print("  the high-beta fit breakdown is a genericity effect, not a law failure.")
print("  Curvature magnitude b*lam_max shows whether curvature vanishes or diverges.")
