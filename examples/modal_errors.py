"""Reproduce full-matrix column-normalized modal error contributions."""
import argparse
import json
from pathlib import Path
import numpy as np
from examples.polarizer import verify_checksums

PACKAGE = Path(__file__).resolve().parents[1]
METHODS = ('projection', 'reciprocal', 'stationary')


def column_percent_contribution(approximate, reference):
    approximate = np.asarray(approximate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if approximate.shape != reference.shape or reference.shape[-2:] != (36, 36):
        raise ValueError('Matching complete 36x36 GSM arrays are required.')
    if not np.isfinite(approximate).all() or not np.isfinite(reference).all():
        raise ValueError('Source matrices must be finite.')
    norms = np.linalg.norm(reference, axis=-2)
    zero = norms == 0
    percent = np.full(reference.shape, np.nan, dtype=float)
    np.divide(100*abs(approximate-reference), norms[..., None, :],
              out=percent, where=~zero[..., None, :])
    return percent, norms, zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PACKAGE/'data/modal_errors/layer_matrices.npz')
    parser.add_argument('--out', type=Path, required=True, help='A new output directory.')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(f'Verified {verify_checksums()} distribution file checksums.')
    result, summary = {}, {}
    with np.load(args.data, allow_pickle=False) as data:
        for case in ('fss', 'cp'):
            exact = data[f'{case}_reference']
            if exact.shape != (5, 36, 36):
                raise ValueError('Each case must retain all five complete layer matrices.')
            maxima = {}
            for method in METHODS:
                error, norms, zero = column_percent_contribution(data[f'{case}_{method}'], exact)
                result[f'{case}_{method}_percent'] = error
                finite = error[np.isfinite(error)]
                maxima[method] = float(finite.max()) if finite.size else None
            result[f'{case}_reference_column_2norm'] = norms
            result[f'{case}_zero_reference_columns'] = zero
            finite_maxima = [value for value in maxima.values() if value is not None]
            upper = max(finite_maxima) if finite_maxima else None
            summary[case] = {'frequency_GHz': float(data[f'{case}_frequency_GHz']),
                             'shape_per_method': [5, 36, 36], 'matrix_rows': 'output ports',
                             'matrix_columns': 'input ports', 'blocks': [['S11', 'S12'], ['S21', 'S22']],
                             'zero_reference_columns': int(zero.sum()),
                             'one_common_linear_percent_range': [0., upper] if upper is not None else None,
                             'maximum_percent_by_method': maxima}
            limits = f'0 to {upper:.3g}%' if upper is not None else 'undefined'
            print(f'{case.upper()}: {summary[case]["frequency_GHz"]:.3g} GHz; 3 methods x 5 layers; common range {limits}')
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/'modal_percent.npz', **result)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
