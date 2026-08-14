# MATLAB Reference Export Requirements

This document lists the MATLAB variables that should be exported as reference output for validating the Python implementation of the Simplus Grid Tool analysis pipeline.

The goal is to create MATLAB `.mat` reference files that can be loaded by Python tests and compared against the Python-native implementation.

Recommended file naming:

```matlab
ieee14_reference.mat
sg_reference.mat
case_name_reference.mat
```

## Export Format

Use MATLAB `.mat` files as the primary reference format.

The repository includes an exporter script:

```matlab
ExportMatlabReference('Results/ieee14_reference.mat', ...
    'ModeSelect', [1 2 3], ...
    'FrequencyHz', logspace(-1, 3, 80));
```

Run it after the normal SimplusGT pipeline has built the workspace models:

```matlab
UserDataName = 'IEEE_14Bus';
UserDataType = 1;
SimplusGT.Toolbox.Main();

ExportMatlabReference('Results/ieee14_reference.mat', ...
    'ModeSelect', [1 2 3], ...
    'FrequencyHz', logspace(-1, 3, 80));
```

Alternatively, let the exporter run the main pipeline:

```matlab
ExportMatlabReference('Results/ieee14_reference.mat', ...
    'RunMain', true, ...
    'UserDataName', 'IEEE_14Bus', ...
    'UserDataType', 1, ...
    'ModeSelect', [1 2 3]);
```

The exporter saves MATLAB v7 `.mat` by default because Python can read it with `scipy.io.loadmat`. Use `'UseV73', true` only for very large reference files, since MATLAB v7.3 uses HDF5 and needs a different Python reader.

## Greybox Excel export (Python workbook parity)

For spreadsheet comparison with Python `export_greybox_excel()`, run after `Main()`:

```matlab
ExportGreyboxExcel('Results/IEEE_14Bus_greybox_matlab.xlsx', ...
    'FrequencyHz', logspace(-1, 3, 80));
```

This writes the same sheets as Python (`Summary`, `Channels`, `Channels_Zsys`, `Ysys`, `Zsys`, wide MagPhase/RealImag tables). Frequency sampling uses `SimplusGT.sampleImpedanceFrequencyResponse` (algebraic \(Z_{\mathrm{sys}}=(G_m+Y_{\mathrm{bus}})^{-1}\)), shared with `ExportMatlabReference.m`.

Compare the `Zsys` long sheets with `tests/test_greybox_excel_matlab_parity.py`.

Manual save command, if needed:

```matlab
save('ieee14_reference.mat', ...
    'NumBus', 'NumApparatus', 'ApparatusType', 'ApparatusBus', ...
    'Bus', 'Line', 'Load', ...
    'Ybus', ...
    'GmDSS_Cell', 'GmDss', 'GsysDss', 'GsysSs', ...
    'Mode', 'Phi', 'Psi', 'ResidueAll', 'ZmValAll', ...
    'Layer1All', 'Layer2All', 'Layer1', 'Layer2', ...
    'SensMatrix', 'Yre_val', 'SensLayer1', 'SensLayer2', 'Layer12', ...
    'FrequencyHz', 'Ysys_values', 'Zsys_values');
```

Use `-v7.3` only if the file is too large for the default `.mat` format.

## 1. Case And Power-Flow References

Export the original case-level data:

```matlab
NumBus
NumApparatus
ApparatusType
ApparatusBus
Bus
Line
Load
PowerFlowResult
```

Also export post-processed network data after load conversion and power-flow initialization:

```matlab
Busbar
Line
Ybus
```

If MATLAB creates explicit dq-expanded or real/imaginary network matrices, export those as well:

```matlab
Ybus_dq
Ybus_re
Ybus_im
```

Purpose: validate that Python builds the same network, load-converted branches, operating point, and network admittance matrix.

## 2. Apparatus Model References

For each apparatus `k`, export the descriptor state-space model:

```matlab
GmDSS_Cell{k}.A
GmDSS_Cell{k}.B
GmDSS_Cell{k}.C
GmDSS_Cell{k}.D
GmDSS_Cell{k}.E
```

Also export state, input, and output labels:

```matlab
ApparatusStateStr{k}
ApparatusInputStr{k}
ApparatusOutputStr{k}
```

Purpose: validate equation-level parity for every apparatus model.

## 3. Combined Apparatus And Whole-System DSS

Export the combined apparatus descriptor model:

```matlab
GmDss.A
GmDss.B
GmDss.C
GmDss.D
GmDss.E
```

Export the whole-system descriptor model:

```matlab
GsysDss.A
GsysDss.B
GsysDss.C
GsysDss.D
GsysDss.E
```

Export the standard state-space model used for modal analysis:

```matlab
GsysSs.A
GsysSs.B
GsysSs.C
GsysSs.D
```

If available, also export the network impedance descriptor model:

```matlab
ObjZbusDss.A
ObjZbusDss.B
ObjZbusDss.C
ObjZbusDss.D
ObjZbusDss.E
```

Purpose: validate descriptor interconnection, apparatus/network feedback, and DSS-to-SS conversion.

## 4. Modal Analysis References

From the MATLAB modal calculation, export:

```matlab
A
B
C
Phi
Psi
D
Mode
ModeSelect
```

For each selected mode `modei`, export:

```matlab
lambda
ResidueAll{modei}
ZmValAll{modei}
```

Purpose: validate eigenvalues, left and right eigenvectors, modal residues, and apparatus impedance evaluated at modal frequencies.

## 5. Apparatus Impedance Participation References

From apparatus Layer 1 and Layer 2 impedance participation analysis, export for each selected mode:

```matlab
Layer1All
Layer2All
Layer1
Layer2.real
Layer2.imag
Layer2.real_pu
Layer2.imag_pu
ApparatusSel
```

Also export the raw inputs used to calculate these values:

```matlab
Residue
ZmVal
ApparatusType
ApparatusBus
```

Purpose: validate greybox apparatus Layer 1 and Layer 2 results.

## 6. Sensitivity Analysis References

From sensitivity analysis, export for each selected mode:

```matlab
SensMatrix
Yre_val
SensLayer1
SensLayer2
Layer12
```

Also export the normalized plotted outputs:

```matlab
Layer1
Layer2.real
Layer2.imag
Layer2.real_pu
Layer2.imag_pu
```

Purpose: validate nodal and branch admittance sensitivity Layer 1 and Layer 2 results.

## 7. Layer 3 References

For apparatus Layer 3 perturbation analysis, export:

```matlab
SelectedApparatus
PerturbationFactor
OriginalParameterValue
PerturbedParameterValue
OriginalMode
PerturbedMode
ModeSensitivity
```

For branch or line sensitivity Layer 3 perturbation analysis, export:

```matlab
SelectedLine
OriginalLineParameter
PerturbedLineParameter
OriginalMode
PerturbedMode
ModeSensitivity
```

Purpose: validate finite-difference parameter sensitivity.

## 8. Frequency-Response References

Export the frequency grid:

```matlab
FrequencyHz
s_values
```

where:

```matlab
s_values = 1j * 2*pi*FrequencyHz;
```

Export full complex frequency-response matrices:

```matlab
Ysys_values
Zsys_values
```

The expected shape is:

```matlab
Ysys_values(output_index, input_index, frequency_index)
Zsys_values(output_index, input_index, frequency_index)
```

For `Zsys`, use the true whole-system impedance path from MATLAB `WholeSysZ_cal`. Export the intermediate variables if available:

```matlab
GminSS
Zm
Ybus
Zsys
```

Purpose: validate the frequency-dependent whole-system admittance and impedance shown in the dashboard.

## Minimum Required Reference Set

If only one compact reference file is created, include at least:

```matlab
Bus
Line
Load
Ybus

GmDSS_Cell
GmDss
GsysDss
GsysSs

Mode
Phi
Psi
ResidueAll
ZmValAll

Layer1All
Layer2All
Layer1
Layer2

SensMatrix
Yre_val
SensLayer1
SensLayer2
Layer12

FrequencyHz
Ysys_values
Zsys_values
```

## Suggested MATLAB Script Structure

A MATLAB export script can follow this sequence:

1. Load the selected SimplusGT example network.
2. Run the normal power-flow and model-building pipeline.
3. Save case-level and post-processed network variables.
4. Save each apparatus descriptor model from `GmDSS_Cell`.
5. Save combined apparatus and whole-system descriptor models.
6. Run modal analysis for selected modes.
7. Save `Mode`, `Phi`, `Psi`, `ResidueAll`, and `ZmValAll`.
8. Run apparatus Layer 1 and Layer 2 calculations.
9. Run sensitivity Layer 1 and Layer 2 calculations.
10. Run selected Layer 3 perturbation calculations.
11. Evaluate `Ysys` and `Zsys` on the chosen frequency grid.
12. Save all variables to a `.mat` reference file.

## Notes For Python Comparison

Complex matrices should be saved as native MATLAB complex arrays where possible.

State, input, and output label arrays should be saved together with the numeric matrices. This helps Python tests verify both numeric values and channel ordering.

When comparing eigenvectors, Python tests should allow for sign or complex-scale differences. Eigenvalues, residues, transfer functions, admittance matrices, impedance matrices, and normalized layer results are better direct parity targets.
