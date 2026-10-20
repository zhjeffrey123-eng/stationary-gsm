"""COCG with a reciprocal-source safeguard for normal incidence."""
import time
import numpy as np
from . import correction, physics
from .physics import apply

def reciprocal_basis(orders):
    P = 2 * len(orders)
    U = np.zeros((P, P), complex)
    for oi, order in enumerate(orders):
        rev = int(np.flatnonzero(np.all(orders == -order, axis=1))[0])
        for pol in [0, 1]:
            i, j = (2 * oi + pol, 2 * rev + pol)
            if oi == rev:
                U[i, i] = 1
            elif oi < rev:
                U[i, i] = U[j, i] = 1 / np.sqrt(2)
                U[i, j] = -1j / np.sqrt(2)
                U[j, j] = 1j / np.sqrt(2)
    if np.linalg.norm(U.conj().T @ U - np.eye(P)) > 1e-12:
        raise ArithmeticError('Nonunitaryreciprocalbasis')
    return U

def _refine_observed(a, Y0, tolerance=0.0001, maximum_iterations=512, levels=(0.1, 0.01, 0.001, 0.0001), fixed=(0, 4, 16, 64, 256, 512), *, observe):
    """每个端口第一次经真实残差检查达标时保存，不用末次电流替代早期快照。

    双线性型 d^T Z d 可能抵消，失效端口单独记录，不能当作成功。
    斜入射算子并非同相位复对称，应使用 physics.phase_solve 的 FFT 求解器。
    """
    if np.linalg.norm(a.p['beta']) > 1e-12:
        raise ValueError('COCG 仅用于法向入射复对称系统')
    start = time.perf_counter()
    B, Y = (a.p['B'][a.ids], np.asarray(Y0, complex).copy())
    Az = lambda x: apply(a.plus, a.ids, x)
    Ap = lambda x: apply(a.plus, a.ids, x, inverse=True)
    norms = np.linalg.norm(B, axis=0)
    r = B - Az(Y)
    s = Ap(r)
    gamma = np.sum(r * s, axis=0)
    direction = s.copy()
    np_, nl = (B.shape[1], len(levels))
    snapshots = [np.full_like(Y, np.nan) for _ in levels]
    hit = np.zeros((np_, nl), bool)
    hit_iter = np.full((np_, nl), np.nan)
    hit_res = np.full((np_, nl), np.nan)
    level_times = np.full(nl, np.nan)
    initial = np.linalg.norm(r, axis=0) / np.maximum(norms, np.finfo(float).tiny)
    for j, level in enumerate(levels):
        ix = initial <= level
        snapshots[j][:, ix] = Y[:, ix]
        hit[ix, j], hit_iter[ix, j], hit_res[ix, j] = (True, 0, initial[ix])
        if np.all(ix):
            level_times[j] = 0.0
    converged, broken = (initial <= tolerance, np.zeros(np_, bool))
    iterations = np.zeros(np_, int)
    direction[:, converged] = 0
    fixed_snapshots = {0: Y.copy()}
    fixed_seconds = {0: 0.0}
    history = [[float(np.linalg.norm(r) / np.linalg.norm(B)), float(np.max(initial))]]
    last = 0
    observe(Y, 0)
    for it in range(1, maximum_iterations + 1):
        if np.all(converged | broken):
            break
        q = Az(direction)
        den = np.sum(direction * q, axis=0)
        pivot_scale = np.sum(abs(direction) * abs(q), axis=0)
        newly_broken = ~converged & ~broken & (abs(den) <= 1e-14 * np.maximum(pivot_scale, np.finfo(float).tiny))
        iterations[newly_broken] = it
        broken |= newly_broken
        valid = ~converged & ~broken & (abs(den) > np.finfo(float).tiny)
        alpha = np.zeros(np_, complex)
        alpha[valid] = gamma[valid] / den[valid]
        Y += direction * alpha
        observe(Y, it)
        r -= q * alpha
        if it % 16 == 0:
            r = B - Az(Y)
        rel = np.linalg.norm(r, axis=0) / np.maximum(norms, np.finfo(float).tiny)
        pending = np.any(~hit & (rel[:, None] <= np.asarray(levels)), axis=1)
        newly = np.zeros(np_, bool)
        if np.any(pending):
            cols = np.flatnonzero(pending)
            exact = np.linalg.norm(B[:, pending] - Az(Y[:, pending]), axis=0) / norms[pending]
            newly[cols] = ~converged[cols] & (exact <= tolerance)
            for j, level in enumerate(levels):
                local = ~hit[cols, j] & (exact <= level)
                ix = cols[local]
                snapshots[j][:, ix] = Y[:, ix]
                hit[ix, j], hit_iter[ix, j], hit_res[ix, j] = (True, it, exact[local])
                if np.all(hit[:, j]) and np.isnan(level_times[j]):
                    level_times[j] = time.perf_counter() - start
        iterations[newly] = it
        converged |= newly
        s = Ap(r)
        new_gamma = np.sum(r * s, axis=0)
        valid = ~converged & ~broken & (abs(gamma) > np.finfo(float).tiny)
        beta = np.zeros(np_, complex)
        beta[valid] = new_gamma[valid] / gamma[valid]
        direction = s + direction * beta
        direction[:, converged | broken] = 0
        gamma, last = (new_gamma, it)
        history.append([float(np.linalg.norm(r) / np.linalg.norm(B)), float(np.max(rel))])
        if it in fixed:
            fixed_snapshots[it] = Y.copy()
            fixed_seconds[it] = time.perf_counter() - start
    rel = np.linalg.norm(B - Az(Y), axis=0) / np.maximum(norms, np.finfo(float).tiny)
    iterations[(rel > tolerance) & ~broken] = last
    for k in fixed:
        if k not in fixed_snapshots:
            fixed_snapshots[k] = Y.copy()
            fixed_seconds[k] = time.perf_counter() - start
    return (Y, {'iterations': iterations, 'converged': rel <= tolerance, 'broken': broken, 'residuals': rel, 'history': np.asarray(history), 'level_hit': hit, 'level_iterations': hit_iter, 'level_residuals': hit_res, 'level_seconds': level_times, 'seconds': time.perf_counter() - start, 'fixed': fixed_snapshots, 'fixed_seconds': fixed_seconds, 'last_iteration': last}, snapshots)


class _AllOriginalPortsAttained(Exception):
    pass


def refine(a, initial):
    """Return first verified 0.1-residual currents in the original port basis.

    The primary stage preserves correction.refine's original COCG snapshots.
    Only unattained columns use COCG in the unitary reciprocal cos/sin basis.
    Both stages have a 512-iteration limit. No model inference occurs here.
    """
    initial = np.asarray(initial, complex)
    B = a.p['B'][a.ids]
    if initial.shape != B.shape or not np.isfinite(initial).all():
        raise ValueError('Initial currents must match the active rows and input columns.')
    norms = np.linalg.norm(B, axis=0)
    if np.any(norms == 0):
        raise ValueError('A zero excitation column cannot use this residual criterion.')
    _, primary, snapshots = correction.refine(
        a, initial, tolerance=.01, maximum_iterations=512, levels=(.1, .01), fixed=())
    Y = snapshots[0].copy()
    hit = primary['level_hit'][:, 0].copy()
    stage = np.where(hit, 'primary_COCG', 'unattained').astype('<U32')
    first = primary['level_iterations'][:, 0].copy()
    guard_last = 0
    if not np.all(hit):
        U = reciprocal_basis(a.p['orders'])
        ports = dict(a.p)
        ports['B'] = a.p['B'] @ U
        transformed = physics.Sheet(a.plus, a.minus, ports, ports, a.ids)

        def observe(current, iteration):
            nonlocal guard_last
            guard_last = int(iteration)
            original = current @ U.conj().T
            needed = np.flatnonzero(~hit)
            if not len(needed):
                raise _AllOriginalPortsAttained()
            residual = np.linalg.norm(
                B[:, needed] - apply(a.plus, a.ids, original[:, needed]), axis=0) / norms[needed]
            attained = needed[residual <= .1]
            Y[:, attained] = original[:, attained]
            hit[attained] = True
            stage[attained] = 'reciprocal_basis_COCG'
            first[attained] = iteration
            if np.all(hit):
                raise _AllOriginalPortsAttained()

        try:
            _, guard, _ = _refine_observed(
                transformed, initial @ U, tolerance=.01, maximum_iterations=512,
                levels=(.1, .01), fixed=(), observe=observe)
            guard_last = int(guard['last_iteration'])
        except _AllOriginalPortsAttained:
            pass
    if not np.all(hit):
        raise RuntimeError('Some original input columns did not attain the verified residual level.')
    residual = np.linalg.norm(B - apply(a.plus, a.ids, Y), axis=0) / norms
    if not np.isfinite(residual).all() or np.max(residual) > .1 * (1 + 1e-10):
        raise ArithmeticError('Final original-port residual verification failed.')
    return Y, {'source_stage': stage, 'first_hit_iteration_in_stage': first,
               'true_original_port_residuals': residual,
               'primary_broken_ports': primary['broken'],
               'primary_last_iteration': int(primary['last_iteration']),
               'guard_last_iteration': guard_last, 'tau': .1}
