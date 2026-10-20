# Stationary GSM Extraction for Doubly Periodic PEC Electromagnetic Structures

Periodic PEC EFIE operators, reciprocal ports, same-current Projection/Reciprocity/Stationary GSM extraction, and layer connections. The extraction functions read the same approximate currents; the readout does not update those currents.

## Run

Tested with Python 3.12 and the dependencies pinned in `requirements.txt`. Run these commands from the repository root. Each output directory must be new.

```bash
python -m stationary_gsm demo --out local_demo
python -m stationary_gsm extract --input my_currents.npz --out local_GSM
python -m examples.polarizer --mode report --verify --out local_CP_report
python -m examples.modal_errors --verify --out local_modal_errors
```

The CP report reconstructs both Jones columns, transmittance, axial ratio, signed circularity, functional errors, and the CP matrix-error statistics from the supplied complex matrices. Generated CSV/NPZ/JSON files retain full precision. Console summaries use at most three significant figures and omit decimal trailing zeros.

## Supplied numeric inputs

- `data/polarizer/geometry.npz`: the actual five 8x8 PEC masks, lateral origins, four air gaps, physical period, scale, and u/v-to-lab rotation. Mask rows are v and columns are u. The u/v lattice rotates by 45 degrees relative to laboratory x/y. The period is approximately 22.8 mm; use the full-precision file to reproduce the geometry.
- `data/polarizer/response_matrices.npz`: 145 evaluated frequencies over 5-9 GHz, comprising 121 fixed base points and 24 adaptive local-extremum points. Each frequency contains five complete complex 4x4 fundamental GSMs: coupled EFIE, matched exact layer cascade, Projection, Reciprocity, and Stationary. All three approximate readouts use the same corrected currents.
- `data/modal_errors/layer_matrices.npz`: both cases' exact and three approximate isolated-layer GSMs, five complete 36x36 matrices per method and case. FSS uses the full stored frequency near 6.97 GHz; CP uses 7 GHz, matching their respective physical-current maps.
- `data/fss`: the original five-layer FSS geometry, coupled reference curve, and main-peak result.
- `data/resolution`: the required pattern-701, 40-degree complete N=96 GSM endpoints and their matched N=80 reference for the numerical-resolution checks.

The CP response uses N=24, Q=192. Over the evaluated 6-8 GHz band, the minimum coupled transmittance is approximately 0.939 and the maximum axial ratio is 0.597 dB. The two independently evaluated lab x/y inputs have opposite circularity. The stored arrays contain the unrounded values.

The physical reference is the full interlayer coupled EFIE with independent laboratory input columns. Extraction errors instead use the exact N=24, Q=192, M=1 layer cascade. Table I uses one complete complex 4x4 fundamental GSM per fixed base frequency:

`100 * norm(S_method - S_matched, 'fro') / norm(S_matched, 'fro')`.

The median and 95th percentile use only the 121 base points. Adaptive points remain in the response and performance checks. Single-frequency layer maps do not replace these multi-frequency statistics. The 4x4 stored response order is front Ex/Ey followed by rear Ex/Ey for both output rows and input columns. The front-incident Jones matrix is `S[2:4, 0:2]`. At normal incidence, total transmitted power is `abs(Ex)**2 + abs(Ey)**2`; axial ratio comes from the singular values of `[real(E), -imag(E)]`. Signed circularity is `2*imag(Ex*conj(Ey))/T`. Circular output components use the basis `(x+i*y)/sqrt(2), (x-i*y)/sqrt(2)`.

## Full modal error contributions

For each actual layer and incoming column j, the reader calculates

`E[i,j] = 100 * abs(S_method[i,j] - S_exact[i,j]) / norm(S_exact[:,j], 2)`.

This is a percentage amplitude-error contribution relative to the complete reference input-column norm. It is not an element-relative error or a power. Every 36x36 matrix retains all four 18x18 blocks: S11/S12 above S21/S22. Rows are output ports; columns are input ports. Each side has 18 modes, with m varying fastest and n slowest over -1,0,1; each order interleaves lab Ex/Ey. The NPZ includes these labels and each lattice rotation.

A zero reference-column norm produces an explicitly flagged undefined column filled with NaN; no denominator floor is added. Each case has one linear percentage range shared across its three methods and five layers. Use the full-precision range in the generated JSON when plotting; the console maximum is rounded for display. Current-map normalization to 0-1 is a different quantity and is not included in these percentage scales. The reader produces numeric results only.

## User-supplied currents and the shared model

Regenerating the shared 128-unit MLP predictions requires the original unpublished weights and their matching features and normalization inputs. Those inputs and weights are not supplied. Stored GSM matrices support the numeric replay described above; they do not regenerate learned currents.

The actual current procedure is the shared 128-MLP followed by primary COCG, with a conditional reciprocal cos/sin source-basis safeguard (guard v2). First verified original-port residual-0.1 snapshots from the primary stage are preserved. Only unattained columns use the safeguard, which returns the first independently verified original traveling-port snapshots. Both stages have a 512-iteration limit. The original FSS snapshots use the primary stage; its safeguard is not triggered. The CP data use this common conditional procedure, including the safeguard where required.

`stationary_gsm.source_guard` supplies that correction procedure for user-provided initial currents. The three extraction readouts then use the same returned currents. To replay the five-layer readout with your own currents:

```bash
python -m examples.polarizer --mode readout --input my_CP_currents.npz --out local_CP_readout
python -m examples.polarizer --mode readout --input my_CP_initial_currents.npz --initial --out local_CP_corrected
python -m examples.polarizer --mode reference --frequencies 7 --N 24 --Q 192 --out local_CP_reference
```

A CP current NPZ contains `frequency_GHz`, `N`, `Q`, `M=1`, and complex `Y_0` through `Y_4`. Current rows follow `physics.active(mask, N)` for each supplied mask. Coefficients expand eta0*J in the rooftop basis. Columns follow `physics.ports(frequency_GHz*scale, N, 1, 0, 0)` in the u/v lattice frame, including the corresponding layer-origin phase. The returned GSM rotates both input and output components to lab x/y. `--initial` applies the common COCG/source-basis safeguard; otherwise the supplied currents are read directly. Initial-current prediction itself requires the unpublished model inputs described above.

For the generic single-layer `extract` interface, provide `mask`, `N`, `Q`, `M=1`, `frequency_GHz`, and complex `Yplus`; oblique incidence additionally requires `theta_degrees`, `phi_degrees`, and `Yminus`. Its axes follow the supplied mask and ports. Reciprocal bilinear readout uses an ordinary transpose, not a conjugate transpose.

The reference and geometry examples use the exact scale relation `f_reference = f_physical * scale`, `gap_reference = gap_physical / scale`, with a 20 mm canonical period. This preserves the full physical geometry and does not interpolate frequency responses.

The original FSS/reference and resolution tools remain available:

```bash
python -m examples.stack_reference --geometry data/fss/geometry.json --frequencies 7 --polarization x --out local_fss
python -m examples.recompute_resolution --geometry data/resolution/geometry.json --pattern 701 --N 80 96 --Q-ratios 6 8 --theta 40 --out local_resolution
```

Device timing in the manuscript concerns cached-current readout plus cascade. It excludes model prediction, current correction, operator preparation, training, and design; it is not an end-to-end speedup. Model weights, current snapshots, training/test materials, private CPU records, manuscripts, and figure sources are excluded from this distribution.

## Citation

See `CITATION.cff` for the software title, authors, and version 3.8.0. `CODE_SHA256SUMS.txt` verifies the distribution files.
