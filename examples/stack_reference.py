"""Recompute normal-incidence reference responses of a frozen PEC stack.

Run from the repository root with Python's module option, for example:
python -m examples.stack_reference --geometry data/fss/geometry.json \
    --frequencies 7.0 --N 24 --M 1 --polarization x --out local_reference
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from stationary_gsm import physics as ph, stack


def load_geometry(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    masks = np.asarray(data['masks'], dtype=bool)
    gaps = np.asarray(data['gaps_m'], dtype=float)
    if masks.ndim != 3 or masks.shape[1:] != (8, 8) or len(gaps) != len(masks)-1:
        raise ValueError('Geometry needs masks[layer,y,x] with 8x8 cells and one gap between layers.')
    if np.any(gaps <= 0) or not np.all(np.isfinite(gaps)):
        raise ValueError('Air gaps must be finite and positive.')
    if not np.isclose(float(data['period_m']), ph.PERIOD, rtol=0, atol=1e-14):
        raise ValueError('The implemented EFIE uses a 0.02 m period.')
    return masks, gaps


def fundamental_coupled(masks, gaps, frequency, N, Q):
    """Solve the four fundamental inputs directly, without an interface truncation.

    The direct solve retains the same rooftop basis and Green spectrum as the
    single-layer model. The normal-incidence matrix is complex symmetric.
    """
    pt = ph.ports(frequency, N, 1, 0, 0)
    if np.count_nonzero(np.linalg.eigvalsh(pt['G']) > 1e-12) != 2:
        raise ValueError('This four-port output requires only the fundamental order to propagate.')
    kernel = ph.kernel(frequency, N, Q, 0, 0)
    z = np.r_[0., np.cumsum(gaps)]
    ids = [ph.active(mask, N) for mask in masks]
    offsets = np.r_[0, np.cumsum([len(active) for active in ids])]
    kz = pt['kz'][pt['normal'][0]]
    Z = np.zeros((offsets[-1], offsets[-1]), complex)
    B = np.zeros((offsets[-1], 4), complex)
    C = np.zeros((4, offsets[-1]), complex)
    for layer, active in enumerate(ids):
        region = slice(offsets[layer], offsets[layer+1])
        Z[region, region] = ph.block(kernel, active, active)
        incident = pt['B'][active][:, pt['normal']]
        readout = pt['C'][pt['normal']][:, active]
        front = np.exp(1j*kz*z[layer])
        rear = np.exp(1j*kz*(z[-1]-z[layer]))
        B[region, :2], B[region, 2:] = incident*front, incident*rear
        C[:2, region], C[2:, region] = readout*front, readout*rear
    for left in range(len(ids)):
        for right in range(left+1, len(ids)):
            interaction = ph.kernel(frequency, N, Q, 0, 0, float(z[right]-z[left]))
            value = ph.block(interaction, ids[left], ids[right])
            l = slice(offsets[left], offsets[left+1])
            r = slice(offsets[right], offsets[right+1])
            Z[l, r], Z[r, l] = value, value.T
    X = solve(Z, B, check_finite=False)
    residual = float(np.linalg.norm(Z@X-B)/np.linalg.norm(B))
    direct = np.exp(1j*kz*z[-1])*np.eye(2)
    zeros = np.zeros((2, 2))
    S = np.block([[zeros, direct], [direct, zeros]])+C@X
    defect = float(np.linalg.norm(S.conj().T@S-np.eye(4), 2))
    return S, residual, defect


def cascade_reference(masks, gaps, frequency, N, Q, M):
    sheets = [ph.solve(mask, frequency, N, Q, M, 0, 0)[0] for mask in masks]
    ports = ph.ports(frequency, N, M, 0, 0)
    S, rcond = stack.cascade(sheets, ports['kz'], gaps)
    P, q = len(ports['kz']), ports['normal']
    fundamental = np.r_[q, P+q]
    return S[np.ix_(fundamental, fundamental)], rcond


def response(S, polarization):
    incident = {'x': np.array([1., 0.]), 'y': np.array([0., 1.]),
                '45': np.array([1., 1.])/np.sqrt(2)}[polarization]
    transmitted = S[2:, :2]@incident
    reflected = S[:2, :2]@incident
    T = float(np.vdot(transmitted, transmitted).real)
    R = float(np.vdot(reflected, reflected).real)
    s3 = float(2*np.imag(transmitted[0]*transmitted[1].conj()))
    d = np.sqrt(max(0., T*T-s3*s3))
    axial = float(10*np.log10(max((T+d)/max(T-d, 1e-16), 1.)))
    return {'T_total': T, 'R_total': R,
            'T_copolarized': float(abs(incident@transmitted)**2),
            'AR_dB': axial, 'circularity': s3/max(T, 1e-15),
            'insertion_loss_dB': float(-10*np.log10(max(T, 1e-300)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--frequencies', type=float, nargs='+', required=True)
    parser.add_argument('--N', type=int, default=24)
    parser.add_argument('--Q', type=int)
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--polarization', choices=['x', 'y', '45'], default='x')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    Q = 8*args.N if args.Q is None else args.Q
    if args.N < 8 or args.N % 8 or Q < args.M or args.M < 0 or args.threads < 1:
        raise ValueError('Invalid grid, spectral cutoff, modal cutoff or thread count.')
    if any(f <= 0 or f >= ph.C0/ph.PERIOD/1e9 for f in args.frequencies):
        raise ValueError('Frequency must be positive and below the first diffraction cutoff.')
    masks, gaps = load_geometry(args.geometry)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, matrices, timings = [], {}, []
    with threadpool_limits(limits=args.threads):
        for index, frequency in enumerate(args.frequencies):
            start = time.perf_counter()
            coupled, residual, defect = fundamental_coupled(masks, gaps, frequency, args.N, Q)
            coupled_seconds = time.perf_counter()-start
            start = time.perf_counter()
            cascade, rcond = cascade_reference(masks, gaps, frequency, args.N, Q, args.M)
            cascade_seconds = time.perf_counter()-start
            interface = float(np.linalg.norm(cascade-coupled)/np.linalg.norm(coupled))
            for method, value in [('coupled', coupled), ('reference_cascade', cascade)]:
                rows.append({'frequency_GHz': frequency, 'method': method, **response(value, args.polarization)})
                matrices[f'f{index}_{method}'] = value
            timings.append({'frequency_GHz': frequency, 'coupled_residual': residual,
                            'coupled_power_defect': defect, 'cascade_feedback_rcond': rcond,
                            'relative_interface_error': interface,
                            'coupled_seconds': coupled_seconds, 'cascade_seconds': cascade_seconds})
    with (args.out/'response.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out/'fundamental_GSM.npz', **matrices)
    result = {'N': args.N, 'Q': Q, 'M': args.M, 'polarization': args.polarization,
              'threads': args.threads, 'diagnostics': timings}
    (args.out/'diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
