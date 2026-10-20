"""Recompute full 36x36 reference GSMs and grid/spectral changes.

All 18 input columns are solved at the requested signed Bloch wave vector.
Both-side blocks follow the zero-thickness free-space PEC relation T=I+R.
The calculation uses FFT matrix-free GMRES and verifies true EFIE residuals.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from stationary_gsm import physics as ph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--pattern', type=int, default=701)
    parser.add_argument('--N', type=int, nargs='+', default=[80, 96])
    parser.add_argument('--Q-ratios', type=int, nargs='+', default=[6, 8])
    parser.add_argument('--frequency', type=float, default=7.)
    parser.add_argument('--theta', type=float, default=40.)
    parser.add_argument('--phi', type=float, default=30.)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.geometry.read_text(encoding='utf-8'))
    if not np.isclose(float(source['period_m']), ph.PERIOD, rtol=0, atol=1e-14):
        raise ValueError('The implemented EFIE uses a 0.02 m period.')
    mask = np.asarray(next(row['mask'] for row in source['patterns'] if row['pattern_id'] == args.pattern), bool)
    if any(N < 8 or N % 8 for N in args.N) or any(q < 1 for q in args.Q_ratios):
        raise ValueError('N must be a positive multiple of 8; Q ratios must be positive.')
    if args.frequency <= 0 or args.tolerance <= 0 or args.threads < 1:
        raise ValueError('Frequency, tolerance and thread count must be positive.')
    args.out.mkdir(parents=True, exist_ok=False)
    matrices, diagnostics = {}, []
    with threadpool_limits(limits=args.threads):
        for N in sorted(set(args.N)):
            for ratio in sorted(set(args.Q_ratios)):
                Q = ratio*N
                started = time.perf_counter()
                sheet = ph.sheet(mask, args.frequency, N, Q, 1, args.theta, args.phi)
                B = sheet.p['B'][sheet.ids]
                X, check = ph.phase_solve(sheet.plus, sheet.ids, B, method='fft',
                                          tol=args.tolerance, restart=100, max_cycles=120)
                S = ph.assemble_gsm(sheet.p['C'][:, sheet.ids]@X)
                assert S.shape == (36, 36) and np.all(np.isfinite(S))
                matrices[f'N{N}_Q{Q}'] = S
                diagnostics.append({'N': N, 'Q': Q, 'unknowns': len(sheet.ids),
                                    'GSM_shape': [36, 36], 'input_columns_solved': 18,
                                    'relative_residual': check['relative_residual'],
                                    'maximum_column_residual': check['maximum_column_residual'],
                                    'iterations_per_column': check['iterations'],
                                    'power_defect': ph.power_defect(S, sheet.p),
                                    'seconds': time.perf_counter()-started})
    changes = []
    grids, ratios = sorted(set(args.N)), sorted(set(args.Q_ratios))
    for ratio in ratios:
        for coarse, fine in zip(grids[:-1], grids[1:]):
            a, b = matrices[f'N{coarse}_Q{coarse*ratio}'], matrices[f'N{fine}_Q{fine*ratio}']
            changes.append({'kind': 'grid', 'N_coarse': coarse, 'N_fine': fine,
                            'Q_coarse': coarse*ratio, 'Q_fine': fine*ratio,
                            'relative_full_GSM_change': float(np.linalg.norm(a-b)/np.linalg.norm(b))})
    for N in grids:
        for coarse, fine in zip(ratios[:-1], ratios[1:]):
            a, b = matrices[f'N{N}_Q{N*coarse}'], matrices[f'N{N}_Q{N*fine}']
            changes.append({'kind': 'spectrum', 'N_coarse': N, 'N_fine': N,
                            'Q_coarse': N*coarse, 'Q_fine': N*fine,
                            'relative_full_GSM_change': float(np.linalg.norm(a-b)/np.linalg.norm(b))})
    np.savez_compressed(args.out/'full_GSM.npz', **matrices)
    with (args.out/'resolution_changes.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['kind', 'N_coarse', 'N_fine', 'Q_coarse', 'Q_fine', 'relative_full_GSM_change']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)
    report = {'pattern_id': args.pattern, 'frequency_GHz': args.frequency,
              'theta_degrees': args.theta, 'phi_degrees': args.phi, 'M': 1,
              'solver': 'FFT matrix-free GMRES with periodic inverse preconditioner',
              'threads': args.threads, 'diagnostics': diagnostics}
    (args.out/'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
