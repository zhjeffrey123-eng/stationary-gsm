"""原 EFIE 上的预条件 COCG。只用于法向入射的复对称算子。"""
import time
import numpy as np
from .physics import apply


def refine(a,Y0,tolerance=1e-4,maximum_iterations=512,
           levels=(.1,.01,.001,.0001),fixed=(0,4,16,64,256,512)):
    """每个端口第一次经真实残差检查达标时保存，不用末次电流替代早期快照。

    双线性型 d^T Z d 可能抵消，失效端口单独记录，不能当作成功。
    斜入射算子并非同相位复对称，应使用 physics.phase_solve 的 FFT 求解器。
    """
    if np.linalg.norm(a.p['beta'])>1e-12:
        raise ValueError('COCG 仅用于法向入射复对称系统')
    start = time.perf_counter()
    B,Y = a.p['B'][a.ids],np.asarray(Y0,complex).copy()
    Az = lambda x:apply(a.plus,a.ids,x)
    Ap = lambda x:apply(a.plus,a.ids,x,inverse=True)
    norms = np.linalg.norm(B,axis=0)
    r = B-Az(Y)
    s = Ap(r)
    gamma = np.sum(r*s,axis=0)
    direction = s.copy()
    np_,nl = B.shape[1],len(levels)
    snapshots = [np.full_like(Y,np.nan) for _ in levels]
    hit = np.zeros((np_,nl),bool)
    hit_iter = np.full((np_,nl),np.nan)
    hit_res = np.full((np_,nl),np.nan)
    level_times = np.full(nl,np.nan)
    initial = np.linalg.norm(r,axis=0)/np.maximum(norms,np.finfo(float).tiny)
    for j,level in enumerate(levels):
        ix = initial<=level
        snapshots[j][:,ix] = Y[:,ix]
        hit[ix,j],hit_iter[ix,j],hit_res[ix,j] = True,0,initial[ix]
        if np.all(ix):
            level_times[j] = 0.
    converged,broken = initial<=tolerance,np.zeros(np_,bool)
    iterations = np.zeros(np_,int)
    direction[:,converged] = 0
    fixed_snapshots = {0:Y.copy()}
    fixed_seconds = {0:0.}
    history = [[float(np.linalg.norm(r)/np.linalg.norm(B)),float(np.max(initial))]]
    last = 0
    for it in range(1,maximum_iterations+1):
        if np.all(converged|broken):
            break
        q = Az(direction)
        den = np.sum(direction*q,axis=0)
        pivot_scale = np.sum(abs(direction)*abs(q),axis=0)
        newly_broken = (~converged)&(~broken)&(abs(den)<=1e-14*np.maximum(pivot_scale,np.finfo(float).tiny))
        iterations[newly_broken] = it
        broken |= newly_broken
        valid = (~converged)&(~broken)&(abs(den)>np.finfo(float).tiny)
        alpha = np.zeros(np_,complex)
        alpha[valid] = gamma[valid]/den[valid]
        Y += direction*alpha
        r -= q*alpha
        if it % 16==0:
            r = B-Az(Y)
        rel = np.linalg.norm(r,axis=0)/np.maximum(norms,np.finfo(float).tiny)
        pending = np.any((~hit)&(rel[:,None]<=np.asarray(levels)),axis=1)
        newly = np.zeros(np_,bool)
        if np.any(pending):
            cols = np.flatnonzero(pending)
            exact = np.linalg.norm(B[:,pending]-Az(Y[:,pending]),axis=0)/norms[pending]
            newly[cols] = (~converged[cols])&(exact<=tolerance)
            for j,level in enumerate(levels):
                local = (~hit[cols,j])&(exact<=level)
                ix = cols[local]
                snapshots[j][:,ix] = Y[:,ix]
                hit[ix,j],hit_iter[ix,j],hit_res[ix,j] = True,it,exact[local]
                if np.all(hit[:,j]) and np.isnan(level_times[j]):
                    level_times[j] = time.perf_counter()-start
        iterations[newly] = it
        converged |= newly
        s = Ap(r)
        new_gamma = np.sum(r*s,axis=0)
        valid = (~converged)&(~broken)&(abs(gamma)>np.finfo(float).tiny)
        beta = np.zeros(np_,complex)
        beta[valid] = new_gamma[valid]/gamma[valid]
        direction = s+direction*beta
        direction[:,converged|broken] = 0
        gamma,last = new_gamma,it
        history.append([float(np.linalg.norm(r)/np.linalg.norm(B)),float(np.max(rel))])
        if it in fixed:
            fixed_snapshots[it] = Y.copy()
            fixed_seconds[it] = time.perf_counter()-start
    rel = np.linalg.norm(B-Az(Y),axis=0)/np.maximum(norms,np.finfo(float).tiny)
    iterations[(rel>tolerance)&(~broken)] = last
    for k in fixed:
        if k not in fixed_snapshots:
            fixed_snapshots[k] = Y.copy()
            fixed_seconds[k] = time.perf_counter()-start
    return Y,{'iterations':iterations,'converged':rel<=tolerance,'broken':broken,'residuals':rel,
        'history':np.asarray(history),'level_hit':hit,'level_iterations':hit_iter,'level_residuals':hit_res,
        'level_seconds':level_times,'seconds':time.perf_counter()-start,'fixed':fixed_snapshots,
        'fixed_seconds':fixed_seconds,'last_iteration':last},snapshots
