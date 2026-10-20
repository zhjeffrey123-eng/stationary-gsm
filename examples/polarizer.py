"""Five-layer PEC polarizer: numeric replay and user-supplied currents."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from stationary_gsm import physics as ph, extraction, source_guard

SCALE = 1.0
PERIOD_M = .020
ROT = np.array([[1., -1.], [1., 1.]])/np.sqrt(2)
CIRC = np.array([[1., 1.], [1j, -1j]])/np.sqrt(2)
METHODS = ("projection", "reciprocal", "stationary")
PACKAGE = Path(__file__).resolve().parents[1]

def translated_ports(f, N, M, shift):
    p = ph.ports(f * SCALE, N, M, 0, 0)
    q = 2 * np.pi * p['orders'] / ph.PERIOD
    phase = np.repeat(np.exp(1j * (q @ shift)), 2)
    p['B'] = p['B'] * phase[None, :]
    p['C'] = phase.conj()[:, None] * p['C']
    p['origin_shift_ref_m'] = np.array(shift)
    return p

def rotated_fundamental(S):
    W = np.kron(np.eye(2), ROT)
    return W @ S @ W.T

def rotated_gsm(S):
    W = np.kron(np.eye(len(S) // 2), ROT)
    return W @ S @ W.T

def metrics(S):
    out = {}
    for pol, p in [('x', 0), ('y', 1)]:
        e = S[2:, p]
        r = S[:2, p]
        T = float(np.vdot(e, e).real)
        R = float(np.vdot(r, r).real)
        axes = np.linalg.svd(np.c_[e.real, -e.imag], compute_uv=False)
        ar = float('nan') if T == 0 else float('inf') if axes[1] == 0 else float(20 * np.log10(axes[0] / axes[1]))
        s3 = float(2 * np.imag(e[0] * e[1].conj()))
        phase = float(np.angle(e[1] * e[0].conj(), deg=True)) if abs(e[0]) > 0 and abs(e[1]) > 0 else float('nan')
        out[pol] = {'T': T, 'R': R, 'AR_dB': ar, 'IL_dB': float(-10 * np.log10(T)) if T > 0 else float('inf'), 'circularity': s3 / T if T > 0 else float('nan'), 'phase_Ey_minus_Ex_deg': phase, 'P_x_plus_i_y': float(abs((e[0] - 1j * e[1]) / np.sqrt(2)) ** 2), 'P_x_minus_i_y': float(abs((e[0] + 1j * e[1]) / np.sqrt(2)) ** 2), 'power_balance': T + R - 1}
    out['T_min'] = min(out['x']['T'], out['y']['T'])
    out['AR_defined'] = {pol: bool(not np.isnan(out[pol]['AR_dB'])) for pol in ['x', 'y']}
    out['AR_max_defined'] = bool(all(out['AR_defined'].values()))
    out['AR_max_dB'] = max(out['x']['AR_dB'], out['y']['AR_dB']) if out['AR_max_defined'] else float('nan')
    out['opposite_circularity'] = bool(out['x']['circularity'] * out['y']['circularity'] < 0)
    return out

def join(A, B, D):
    P = len(D)
    a, b, c, d = (A[:P, :P], A[:P, P:] * D[None, :], D[:, None] * A[P:, :P], D[:, None] * A[P:, P:] * D[None, :])
    e, ff, g, h = (B[:P, :P], B[:P, P:], B[P:, :P], B[P:, P:])
    x = solve(np.eye(P) - e @ d, np.c_[e @ c, ff], check_finite=False)
    x1, x2 = (x[:, :P], x[:, P:])
    return np.block([[a + b @ x1, b @ x2], [g @ (c + d @ x1), h + g @ d @ x2]])

def shifted_kernel(f, N, Q, gap, shift):
    """Full Green sum with actual lateral source/test-origin difference before aliasing."""
    if np.linalg.norm(shift) < 1e-15:
        return ph.kernel(f * SCALE, N, Q, 0, 0, gap / SCALE)
    k0 = 2 * np.pi * f * SCALE * 1000000000.0 / ph.C0
    d = ph.PERIOD / N
    W = np.zeros((4, N * N), complex)
    mm = np.arange(-Q, Q + 1)
    for first in range(-Q, Q + 1, 32):
        m, n = np.meshgrid(mm, np.arange(first, min(first + 32, Q + 1)), indexing='ij')
        m, n = (m.ravel(order='F'), n.ravel(order='F'))
        kx, ky = (2 * np.pi * m / ph.PERIOD, 2 * np.pi * n / ph.PERIOD)
        kz = ph.outgoing_kz(k0, kx, ky)
        sx = np.sinc(kx * d / (2 * np.pi)) ** 2 * np.sinc(ky * d / (2 * np.pi))
        sy = np.sinc(kx * d / (2 * np.pi)) * np.sinc(ky * d / (2 * np.pi)) ** 2
        prop = np.exp(1j * kz * gap / SCALE + 1j * (kx * shift[0] + ky * shift[1]))
        xx = (k0 * k0 - kx * kx) / (2 * k0 * kz) * prop
        yy = (k0 * k0 - ky * ky) / (2 * k0 * kz) * prop
        xy = -kx * ky / (2 * k0 * kz) * prop
        alias = m % N + N * (n % N)
        for j, value in enumerate([xx * sx * sx, xy * sx * sy * np.exp(1j * d * (-kx + ky) / 2), xy * sx * sy * np.exp(1j * d * (kx - ky) / 2), yy * sy * sy]):
            W[j] += np.bincount(alias, weights=value.real, minlength=N * N) + 1j * np.bincount(alias, weights=value.imag, minlength=N * N)
    W = np.stack([v.reshape((N, N), order='F') for v in W])
    return {'W': W, 'K': np.fft.ifft2(W, axes=(1, 2)), 'phase': np.ones((N, N), complex), 'N': N, 'Q': Q, 'beta': np.zeros(2), 'k0': k0, 'gap': gap / SCALE}

def coupled(masks, shifts, gaps, f, N, Q=None):
    start = time.perf_counter()
    Q = 8 * N if Q is None else Q
    z = np.r_[0, np.cumsum(gaps)]
    h = z[-1]
    pt = ph.ports(f * SCALE, N, 0, 0, 0)
    g = ph.kernel(f * SCALE, N, Q, 0, 0)
    ids = [ph.active(m, N) for m in masks]
    off = np.r_[0, np.cumsum([len(i) for i in ids])]
    Z = np.zeros((off[-1], off[-1]), complex)
    B = np.zeros((off[-1], 4), complex)
    C = np.zeros((4, off[-1]), complex)
    k0 = 2 * np.pi * f * 1000000000.0 / ph.C0
    for j in range(len(masks)):
        sl = slice(off[j], off[j + 1])
        Z[sl, sl] = ph.block(g, ids[j], ids[j])
        bl, cl = (pt['B'][ids[j]], pt['C'][:, ids[j]])
        front, rear = (np.exp(1j * k0 * z[j]), np.exp(1j * k0 * (h - z[j])))
        B[sl, :2] = bl * front
        B[sl, 2:] = bl * rear
        C[:2, sl] = cl * front
        C[2:, sl] = cl * rear
    for j in range(len(masks)):
        for l in range(j + 1, len(masks)):
            kg = shifted_kernel(f, N, Q, float(z[l] - z[j]), shifts[j] - shifts[l])
            val = ph.block(kg, ids[j], ids[l])
            Z[off[j]:off[j + 1], off[l]:off[l + 1]] = val
            Z[off[l]:off[l + 1], off[j]:off[j + 1]] = val.T
    W = np.kron(np.eye(2), ROT)
    B = B @ W.T
    C = W @ C
    assembled = time.perf_counter()
    X = solve(Z, B, check_finite=False)
    solved = time.perf_counter()
    D = np.exp(1j * k0 * h) * np.eye(2)
    O = np.zeros((2, 2))
    S = np.block([[O, D], [D, O]]) + C @ X
    Suv = W.T @ S @ W
    res = np.linalg.norm(Z @ X - B, axis=0) / np.linalg.norm(B, axis=0)
    return (S, {'N': N, 'Q': Q, 'unknowns': int(off[-1]), 'max_true_column_residual': float(max(res)), 'power_defect': float(np.linalg.norm(S.conj().T @ S - np.eye(4), 2)), 'principal_crosspol_max': float(np.max(abs(Suv[np.ix_([0, 2], [1, 3])]))), 'assembly_seconds': assembled - start, 'solve_seconds': solved - assembled, 'total_seconds': time.perf_counter() - start}, Suv)

def cp_jones(S):
    return CIRC.conj().T @ S[2:, :2]

def cp_gsm(S):
    W = np.kron(np.eye(2), CIRC)
    return W.conj().T @ S @ W

def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value)):
        return 'NaN(undefined)' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify_checksums():
    """Verify the distribution files listed in CODE_SHA256SUMS.txt."""
    manifest = PACKAGE/'CODE_SHA256SUMS.txt'
    count = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = (PACKAGE/name).resolve()
        if not path.is_relative_to(PACKAGE) or not path.is_file():
            raise ValueError(f'Invalid checksum path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ArithmeticError(f'Checksum mismatch: {name}')
        count += 1
    return count


def geometry(path):
    global SCALE, PERIOD_M, ROT
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in data.files}
    if result['masks'].shape != (5, 8, 8) or result['gaps_m'].shape != (4,):
        raise ValueError('This example requires five 8x8 masks and four air gaps.')
    SCALE = float(result['scale'])
    PERIOD_M = float(result['period_m'])
    ROT = result['rotation_uv_to_lab']
    if SCALE <= 0 or not np.allclose(ROT.T @ ROT, np.eye(2), atol=1e-14):
        raise ValueError('Invalid scale or rotation.')
    return result


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(json_finite(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def matrix_error(approximate, reference):
    denominator = np.linalg.norm(reference, axis=(-2, -1))
    result = np.full(denominator.shape, np.nan)
    np.divide(100*np.linalg.norm(approximate-reference, axis=(-2, -1)), denominator,
              out=result, where=denominator != 0)
    return result


def report(path, out):
    with np.load(path, allow_pickle=False) as data:
        f = data['frequency_GHz']
        base = data['base_sample']
        matrices = {name: data[f'S_{name}'] for name in ('coupled', 'matched')+METHODS}
        if f.ndim != 1 or np.any(np.diff(f) <= 0) or base.shape != f.shape or base.dtype != bool:
            raise ValueError('Frequencies must increase strictly and have Boolean base-sample flags.')
        if not np.any(base):
            raise ValueError('No fixed base frequencies are supplied.')
        for value in matrices.values():
            if value.shape != (len(f), 4, 4) or not np.isfinite(value).all():
                raise ValueError('Response arrays must contain finite complete complex 4x4 matrices.')
        N, Q, M = (int(data[key]) for key in ('N', 'Q', 'M'))
    target = (f >= 6) & (f <= 8)
    if not np.any(target):
        raise ValueError('The response has no points in the 6-8 GHz reporting band.')
    response_rows, jones_rows, functional_errors, matrix_rows, table = [], [], [], [], []
    evaluated = {name: [metrics(value) for value in values] for name, values in matrices.items()}
    for name, values in matrices.items():
        for index, value in enumerate(values):
            common = {'frequency_GHz': float(f[index]), 'base_sample': bool(base[index]),
                      'in_target_band': bool(target[index]), 'method': name}
            for column, pol in enumerate(('x', 'y')):
                response_rows.append({**common, 'input': pol, **evaluated[name][index][pol]})
                e = value[2:, column]
                jones_rows.append({**common, 'input': pol,
                                   'Ex_real': float(e[0].real), 'Ex_imag': float(e[0].imag),
                                   'Ey_real': float(e[1].real), 'Ey_imag': float(e[1].imag)})
    for name in METHODS:
        error = matrix_error(matrices[name], matrices['matched'])
        error_physical = matrix_error(matrices[name], matrices['coupled'])
        for index in range(len(f)):
            common = {'frequency_GHz': float(f[index]), 'base_sample': bool(base[index]), 'method': name}
            matrix_rows.append({**common, 'matched_complex4_error_percent': float(error[index]),
                                'coupled_complex4_error_percent': float(error_physical[index])})
            for reference, label in (('matched', 'matched_exact_M1'), ('coupled', 'coupled_EFIE')):
                for pol in ('x', 'y'):
                    a, b = evaluated[name][index][pol], evaluated[reference][index][pol]
                    functional_errors.append({**common, 'reference': label, 'input': pol,
                                              'absolute_T_error_pp': 100*abs(a['T']-b['T']),
                                              'absolute_AR_error_dB': abs(a['AR_dB']-b['AR_dB'])})
        selected = error[base]
        table.append({'case': 'CP', 'method': name, 'frequency_count': int(base.sum()),
                      'median_error_percent': float(np.median(selected)),
                      'p95_error_percent': float(np.percentile(selected, 95)),
                      'maximum_error_percent': float(np.max(selected))})
    physical = [value for index, value in enumerate(evaluated['coupled']) if target[index]]
    summary = {'N': N, 'Q': Q, 'M': M, 'stored_frequencies': len(f),
               'fixed_base_frequencies': int(base.sum()), 'adaptive_frequencies': int((~base).sum()),
               'performance_band_GHz': [6, 8],
               'minimum_T': min(value[pol]['T'] for value in physical for pol in ('x', 'y')),
               'maximum_AR_dB': max(value[pol]['AR_dB'] for value in physical for pol in ('x', 'y')),
               'opposite_circularity_at_all_evaluated_target_points': all(value['opposite_circularity'] for value in physical),
               'performance_reference': 'Coupled EFIE at the stated N and Q.',
               'extraction_reference': 'Matched exact M=1 layer cascade; complete complex 4x4 fundamental GSM.',
               'statistics': 'Fixed base points only; adaptive samples are excluded from the Table I statistics.'}
    save_csv(out/'functional_response.csv', response_rows)
    save_csv(out/'jones_columns.csv', jones_rows)
    save_csv(out/'functional_errors.csv', functional_errors)
    save_csv(out/'matrix_errors.csv', matrix_rows)
    save_csv(out/'table_I_CP.csv', table)
    save_json(out/'summary.json', summary)
    print(f"CP: {len(f)} frequencies ({base.sum()} base + {(~base).sum()} adaptive); N={N}, Q={Q}, M={M}")
    print(f"6-8 GHz: minimum T approximately {summary['minimum_T']:.3g}; maximum AR {summary['maximum_AR_dB']:.3g} dB")
    print(f"Opposite circularity for both inputs: {summary['opposite_circularity_at_all_evaluated_target_points']}")
    for row in table:
        print(f"{row['method']}: median {row['median_error_percent']:.3g}%, p95 {row['p95_error_percent']:.3g}%")


def readout(path, geom, out, initial=False):
    with np.load(path, allow_pickle=False) as data:
        frequency = float(data['frequency_GHz'])
        N, Q, M = (int(data[key]) for key in ('N', 'Q', 'M'))
        currents = [np.asarray(data[f'Y_{layer}'], complex) for layer in range(5)]
    if N <= 0 or N % 8 or Q < 0 or M != 1 or frequency <= 0:
        raise ValueError('Use positive frequency, N divisible by 8, nonnegative Q, and M=1.')
    kernel = ph.kernel(frequency*SCALE, N, Q, 0, 0)
    ports = ph.ports(frequency*SCALE, N, M, 0, 0)
    layers = {method: [] for method in METHODS}
    diagnostics = []
    for layer, current in enumerate(currents):
        translated = translated_ports(frequency, N, M, geom['shifts_ref_m'][layer])
        ids = ph.active(geom['masks'][layer], N)
        sheet = ph.Sheet(kernel, kernel, translated, translated, ids)
        if initial:
            current, diagnostic = source_guard.refine(sheet, current)
            diagnostics.append(diagnostic)
        for method in METHODS:
            layers[method].append(extraction.extract(sheet, current, method=method)[0])
    arrays = {'frequency_GHz': frequency, 'N': N, 'Q': Q, 'M': M}
    P = len(ports['kz'])
    ix = np.r_[ports['normal'], P+ports['normal']]
    for method, sheets in layers.items():
        connected = sheets[0]
        for layer in range(1, 5):
            D = np.exp(1j*ports['kz']*geom['gaps_ref_m'][layer-1])
            connected = join(connected, sheets[layer], D)
        arrays[f'S_layers_{method}_lab'] = np.stack([rotated_gsm(value) for value in sheets])
        arrays[f'S_{method}_full36_lab'] = rotated_gsm(connected)
        arrays[f'S_{method}_lab'] = rotated_fundamental(connected[np.ix_(ix, ix)])
    np.savez_compressed(out/'readouts.npz', **arrays)
    save_json(out/'current_correction.json', {'initial_currents_corrected': initial, 'layers': diagnostics})
    print(f'Three same-current readouts: {frequency:.3g} GHz, five layers, full 36x36 GSM.')


def reference(geom, frequencies, N, Q, out):
    values = []
    for frequency in frequencies:
        S, _, _ = coupled(geom['masks'], geom['shifts_ref_m'], geom['gaps_m'],
                          frequency, N=N, Q=Q)
        values.append(S)
    np.savez_compressed(out/'coupled_reference.npz', frequency_GHz=np.asarray(frequencies),
                        N=N, Q=Q, S_coupled=np.stack(values))
    print(f'Coupled reference: {len(values)} frequencies; N={N}, Q={Q}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('report', 'readout', 'reference'), default='report')
    parser.add_argument('--out', type=Path, required=True, help='A new output directory.')
    parser.add_argument('--data', type=Path, default=PACKAGE/'data/polarizer/response_matrices.npz')
    parser.add_argument('--geometry', type=Path, default=PACKAGE/'data/polarizer/geometry.npz')
    parser.add_argument('--input', type=Path, help='User-supplied five-layer current NPZ for readout mode.')
    parser.add_argument('--initial', action='store_true', help='Apply the common COCG/source-basis guard to supplied initial currents.')
    parser.add_argument('--frequencies', type=float, nargs='+', default=[7.])
    parser.add_argument('--N', type=int, default=24)
    parser.add_argument('--Q', type=int, default=192)
    parser.add_argument('--verify', action='store_true', help='Verify the supplied distribution checksums.')
    args = parser.parse_args()
    if args.initial and args.mode != 'readout':
        parser.error('--initial applies only to readout mode.')
    if args.mode == 'readout' and args.input is None:
        parser.error('readout mode requires --input.')
    if args.verify:
        print(f'Verified {verify_checksums()} distribution file checksums.')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.mode == 'report':
        report(args.data, args.out)
    else:
        geom = geometry(args.geometry)
        if args.mode == 'readout':
            readout(args.input, geom, args.out, initial=args.initial)
        else:
            reference(geom, args.frequencies, args.N, args.Q, args.out)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
