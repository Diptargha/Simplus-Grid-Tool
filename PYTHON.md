# Python SimplusGT

Python migration of the **non-Simulink** Simplus Grid Tool analysis pipeline.

**Full user guide:** [Documentations/PythonUserManual.md](Documentations/PythonUserManual.md)  
(pipeline steps, flowchart, modules, features, and greybox / \(A\)-matrix / \(Z_{\mathrm{sys}}\) mathematics in plain language).

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick start

Edit and run **`UserMain.py`** (same idea as MATLAB `UserMain.m`):

1. Set `USER_DATA_NAME` (e.g. `"IEEE_14Bus"` or a path).
2. Toggle features (`ENABLE_PLOT_*`, `ENABLE_GREYBOX`, exports, …).
3. Run the file from the repo root (`python UserMain.py` or your IDE).

Optional CLI (`pip install -e .`) is also available if you prefer scripts:

```bash
# Run a case
simplusgt run Examples/AcPowerSystem/IEEE_14Bus/IEEE_14Bus.json

# Export dashboard JSON
simplusgt export Examples/AcPowerSystem/IEEE_14Bus/IEEE_14Bus.json -o Results/ieee14.json

# Greybox impedance / participation analysis
simplusgt greybox Examples/AcPowerSystem/IEEE_14Bus/IEEE_14Bus.json --layers all -o Results/ieee14_greybox.json

# MATLAB-style fundamental plots (pole map, admittance Bode, grid strength)
simplusgt plot Examples/AcPowerSystem/IEEE_14Bus/IEEE_14Bus.json -o Results/plots --greybox
```

## What is implemented

| Area | Status |
|------|--------|
| Case I/O (JSON / Excel) | Done |
| Power flow + load-to-self-branch | Done |
| Apparatus models + frame embedding | Done (most example types) |
| Network DSS / whole-system interconnection | Done |
| DSS→SS (`dss2ss`) and eigenvalues | Done |
| Greybox Ysys / Zsys | Done (parity-tested on IEEE 14) |
| Apparatus Layer 1/2/3 | Implemented |
| Sensitivity Layer 1/2/3 | Implemented (AC dq) |
| Fundamental plots (pole / admittance / strength) | Done |
| Greybox Layer 1/2 plots | Done |
| MATLAB numeric parity (Ybus, Gm, Ysys, Zsys, eigs) | Done for IEEE 14 reference |
| Simulink model generation | Out of scope |
| Synchronisation analysis | Not started |

## MATLAB reference parity

1. Run the MATLAB pipeline and export:

```matlab
UserDataName = 'IEEE_14Bus';
UserDataType = 1;
SimplusGT.Toolbox.Main();
ExportMatlabReference('Results/matlab_reference.mat');
```

2. Run Python tests:

```bash
pytest tests/test_matlab_reference_parity.py -q
```

## Plots vs MATLAB Main.m / Modal

| MATLAB figure | Python |
|---------------|--------|
| PlotPoleMap (+ 10% damping lines) | `plot_pole_map` / `simplusgt plot` |
| PlotAdmittanceSpectrum (Ydd, Ydq+) | `plot_admittance_spectrum` |
| Modal BodeDraw (dd/dq/qd/qq) | `plot_admittance_dq_axes` |
| PlotGridStrength | `plot_grid_strength` |
| Modal Layer 1 pie / Layer 2 bars | `plot_apparatus_layer12` / `--greybox` |
| Sensitivity Layer 1/2 charts | `plot_sensitivity_layer12` / `--greybox` |

## Known gaps

- Sensitivity references in `ExportMatlabReference.m` still skip when `WholeSysZ_cal` cannot invert `Gm` (Control Toolbox rank deficiency). Python sensitivity itself runs; MATLAB `.mat` Sens* fields may be absent.
- Apparatus type 19 (stationary-frame GFL) is approximated by the dq GFL model with a warning.
- Unsupported apparatus types use `PlaceholderApparatus` and emit an explicit warning.
- Eigenvector (`Phi`/`Psi`) element-wise parity is not asserted (state ordering / complex scale).
- README.md remains MATLAB-oriented; this file is the Python entry point.
