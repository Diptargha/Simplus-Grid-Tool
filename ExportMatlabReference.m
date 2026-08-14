function ExportMatlabReference(outputPath, varargin)
%EXPORTMATLABREFERENCE Export MATLAB reference data for Python parity tests.
%
% This exporter is intended to be run after the normal SimplusGT model
% building pipeline has populated the MATLAB base workspace, for example:
%
%   UserDataName = 'IEEE_14Bus';
%   UserDataType = 1;
%   SimplusGT.Toolbox.Main();
%   ExportMatlabReference('Results/ieee14_reference.mat', ...
%       'ModeSelect', [1 2 3], ...
%       'FrequencyHz', logspace(-1, 3, 80));
%
% The output is a MATLAB v7 .mat file with flat variables and numeric
% structs. Python can read this format directly with scipy.io.loadmat.
%
% Optional name-value arguments:
%   ModeSelect   - MATLAB 1-based mode indices used for greybox references.
%                  Default: first ten finite modes.
%   FrequencyHz  - Frequency grid for Ysys/Zsys response references.
%                  Default: logspace(-1, 3, 80).
%   ApparatusSel - MATLAB 1-based apparatus indices for Layer 1/2.
%                  Default: all non-floating apparatuses.
%   UseV73       - Save with -v7.3. Default: false. Only use this for large
%                  files because scipy.io.loadmat does not read v7.3 files.
%   RunMain      - If true, assign UserDataName/UserDataType in base and run
%                  SimplusGT.Toolbox.Main() before exporting.
%   UserDataName - Case name used when RunMain is true.
%   UserDataType - 0 for JSON, 1 for Excel. Used when RunMain is true.

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile(pwd, 'Results', 'matlab_reference.mat');
end

p = inputParser;
addParameter(p, 'ModeSelect', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
addParameter(p, 'FrequencyHz', logspace(-1, 3, 80), @(x) isnumeric(x) && isvector(x));
addParameter(p, 'ApparatusSel', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
addParameter(p, 'UseV73', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'RunMain', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'UserDataName', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'UserDataType', 0, @(x) isnumeric(x) && isscalar(x));
parse(p, varargin{:});
opts = p.Results;

if opts.RunMain
    if strlength(string(opts.UserDataName)) == 0
        error('UserDataName must be provided when RunMain is true.');
    end
    assignin('base', 'UserDataName', char(opts.UserDataName));
    assignin('base', 'UserDataType', opts.UserDataType);
    evalin('base', 'SimplusGT.Toolbox.Main();');
end

outputPath = char(outputPath);
outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

warnings = {};
ref = struct();
ref.ReferenceInfo = struct( ...
    'created_at', char(datetime('now')), ...
    'format', 'MATLAB v7 MAT file with flat variables', ...
    'mode_indexing', 'ModeSelect is MATLAB 1-based; ModeSelectPythonZeroBased is provided for Python.', ...
    'frequency_unit', 'Hz');

%% Case and power-flow data
ref = copyBaseIfExists(ref, 'NumBus');
ref = copyBaseIfExists(ref, 'NumApparatus');
ref = copyBaseIfExists(ref, 'ApparatusType');
ref = copyBaseIfExists(ref, 'ApparatusBus');
ref = copyBaseIfExists(ref, 'Para');
ref = copyBaseIfExists(ref, 'Ts');
ref = copyBaseIfExists(ref, 'Fbase');
ref = copyBaseIfExists(ref, 'Sbase');
ref = copyBaseIfExists(ref, 'Vbase');
ref = copyBaseIfExists(ref, 'Ibase');
ref = copyBaseIfExists(ref, 'Zbase');
ref = copyBaseIfExists(ref, 'Ybase');
ref = copyBaseIfExists(ref, 'Wbase');

if baseExists('ListBus')
    ref.Bus = baseValue('ListBus');
    ref.ListBus = ref.Bus;
else
    warnings{end + 1} = 'ListBus was not found; Bus was not exported.';
end

if baseExists('ListLine')
    ref.Line = baseValue('ListLine');
    ref.ListLine = ref.Line;
else
    warnings{end + 1} = 'ListLine was not found; Line was not exported.';
end

if baseExists('ListBusNew')
    ref.Busbar = baseValue('ListBusNew');
    ref.ListBusNew = ref.Busbar;
end

if baseExists('ListLineNew')
    ref.LinePostLoad = baseValue('ListLineNew');
    ref.ListLineNew = ref.LinePostLoad;
end

if baseExists('PowerFlow')
    ref.PowerFlowResult = baseValue('PowerFlow');
    ref.PowerFlow = ref.PowerFlowResult;
end

if baseExists('PowerFlowNew')
    ref.PowerFlowNew = baseValue('PowerFlowNew');
end

if baseExists('UserDataStruct')
    userData = baseValue('UserDataStruct');
    ref.UserDataStruct = userData;
    if isfield(userData, 'Load')
        ref.Load = userData.Load;
    end
end

%% Descriptor/state-space models and labels
if baseExists('GmDssCell')
    GmDssCell = baseValue('GmDssCell');
    ref.GmDSS_Cell = packStateSpaceCell(GmDssCell);
    ref.GmDssCell = ref.GmDSS_Cell;
else
    warnings{end + 1} = 'GmDssCell was not found; apparatus DSS models were not exported.';
end

if baseExists('ObjGm')
    try
        ObjGmRaw = baseValue('ObjGm');
        [~, GmDss] = ObjGmRaw.GetDSS(ObjGmRaw);
        ref.GmDss = packStateSpace(GmDss);
    catch err
        warnings{end + 1} = ['Could not export GmDss from ObjGm: ', err.message];
    end
end

if baseExists('GsysDss')
    ref.GsysDss = packStateSpace(baseValue('GsysDss'));
else
    warnings{end + 1} = 'GsysDss was not found.';
end

if baseExists('GsysSs')
    GsysSs = baseValue('GsysSs');
    ref.GsysSs = packStateSpace(GsysSs);
else
    GsysSs = [];
    warnings{end + 1} = 'GsysSs was not found; modal references cannot be computed.';
end

if baseExists('YbusDss')
    YbusDss = baseValue('YbusDss');
    ref.YbusDss = packStateSpace(YbusDss);
    try
        ref.Ybus = evalfr(YbusDss, 0);
    catch err
        warnings{end + 1} = ['Could not evaluate YbusDss at s=0: ', err.message];
    end
elseif baseExists('ObjYbusDss')
    try
        ObjYbusDssRaw = baseValue('ObjYbusDss');
        [~, YbusDss] = ObjYbusDssRaw.GetDSS(ObjYbusDssRaw);
        ref.YbusDss = packStateSpace(YbusDss);
        ref.Ybus = evalfr(YbusDss, 0);
    catch err
        warnings{end + 1} = ['Could not export YbusDss from ObjYbusDss: ', err.message];
    end
end

if baseExists('ObjZbusDss')
    try
        ObjZbusDssRaw = baseValue('ObjZbusDss');
        [~, ObjZbusDssModel] = ObjZbusDssRaw.GetDSS(ObjZbusDssRaw);
        ref.ObjZbusDss = packStateSpace(ObjZbusDssModel);
    catch err
        warnings{end + 1} = ['Could not export ObjZbusDss: ', err.message];
    end
end

if baseExists('YsysDss')
    ref.YsysDss = packStateSpace(baseValue('YsysDss'));
end

if baseExists('YsysSs')
    ref.YsysSs = packStateSpace(baseValue('YsysSs'));
end

labelVars = { ...
    'ApparatusStateStr', 'ApparatusInputStr', 'ApparatusOutputStr', ...
    'GsysDssStateStr', 'GsysDssInStr', 'GsysDssOutStr', ...
    'GsysSsStateStr', 'GsysSsInStr', 'GsysSsOutStr', ...
    'PortV', 'PortI', 'PortBusV', 'PortBusI'};
for i = 1:numel(labelVars)
    ref = copyBaseIfExists(ref, labelVars{i});
end

%% Modal references
modeSelect = opts.ModeSelect(:).';
if ~isempty(GsysSs)
    try
        A = GsysSs.A;
        B = GsysSs.B;
        C = GsysSs.C;
        [Phi, D] = eig(A);
        Psi = inv(Phi);
        Mode = diag(D) / (2 * pi);
        LambdaRad = diag(D);

        if isempty(modeSelect)
            finiteModeIdx = find(isfinite(Mode));
            modeSelect = finiteModeIdx(1:min(10, numel(finiteModeIdx))).';
        end
        validateModeSelect(modeSelect, numel(Mode));

        ref.A = full(A);
        ref.B = full(B);
        ref.C = full(C);
        ref.Phi = Phi;
        ref.Psi = Psi;
        ref.D = D;
        ref.Mode = Mode;
        ref.LambdaRad = LambdaRad;
        ref.ModeSelect = modeSelect;
        ref.ModeSelectMatlab = modeSelect;
        ref.ModeSelectPythonZeroBased = modeSelect - 1;
    catch err
        warnings{end + 1} = ['Could not compute modal eigensystem: ', err.message];
    end
end

if isfield(ref, 'ModeSelect') && baseExists('NumApparatus') && baseExists('ApparatusType') && ...
        baseExists('GmDssCell') && baseExists('ApparatusInputStr') && baseExists('ApparatusOutputStr')
    try
        N_Apparatus = baseValue('NumApparatus');
        ApparatusType = baseValue('ApparatusType');
        GmDssCell = baseValue('GmDssCell');
        ApparatusInputStr = baseValue('ApparatusInputStr');
        ApparatusOutputStr = baseValue('ApparatusOutputStr');

        try
            [MdMode, ResidueAll, ZmValAll] = SimplusGT.Modal.SSCal( ...
                GsysSs, N_Apparatus, ApparatusType, ref.ModeSelect, ...
                GmDssCell, ApparatusInputStr, ApparatusOutputStr);
        catch
            GsysDssRaw = baseValue('GsysDss');
            [MdMode, ResidueAll, ZmValAll] = SimplusGT.Modal.SSCal( ...
                GsysSs, N_Apparatus, ApparatusType, ref.ModeSelect, ...
                GmDssCell, GsysDssRaw, ApparatusInputStr, ApparatusOutputStr);
        end

        ref.MdMode = MdMode;
        ref.ResidueAll = ResidueAll;
        ref.ZmValAll = ZmValAll;

        [Layer1All, Layer2All, Layer1, Layer2, ApparatusSel] = ...
            computeApparatusLayer12References(ResidueAll, ZmValAll, opts.ApparatusSel);
        ref.Layer1All = Layer1All;
        ref.Layer2All = Layer2All;
        ref.Layer1 = Layer1;
        ref.Layer2 = Layer2;
        ref.ApparatusSel = ApparatusSel;
    catch err
        warnings{end + 1} = ['Could not compute apparatus greybox references: ', err.message];
    end
end

%% Sensitivity references
% Prefer algebraic Z reconstruction for modes (WholeSysZ_cal uses Control
% Toolbox inv(Gm), which is often rank-deficient). Fall back to WholeSysZ_cal
% only when a finite state-space ZminSS is available.
if isfield(ref, 'ModeSelect') && baseExists('ObjGm') && baseExists('ObjYbusDss') && ...
        baseExists('PortI') && baseExists('PortV')
    try
        ObjGm = baseValue('ObjGm');
        ObjYbusDss = baseValue('ObjYbusDss');
        PortI = baseValue('PortI');
        PortV = baseValue('PortV');
        ZminSS = [];
        warning_state = warning('off', 'MATLAB:nearlySingularMatrix');
        warning_state2 = warning('off', 'Control:transformation:InverseNonFinite');
        try
            ZminSS = SimplusGT.WholeSysZ_cal(ObjGm, ObjYbusDss, PortI, PortV);
            if ~all(isfinite(ZminSS.A(:))) || ~all(isfinite(ZminSS.D(:)))
                ZminSS = [];
            end
        catch
            ZminSS = [];
        end
        warning(warning_state);
        warning(warning_state2);

        if isempty(ZminSS)
            error(['WholeSysZ_cal did not produce a finite ZminSS. ', ...
                'Sensitivity Layer references require the Control Toolbox inverse path; ', ...
                'use Python greybox sensitivity instead.']);
        end
        ref.ZminSS = packStateSpace(ZminSS);

        [~, ZD] = eig(ZminSS.A);
        ZMode_rad = diag(ZD);
        ZMode_Hz = ZMode_rad / (2 * pi);
        ref.ZMode_rad = ZMode_rad;
        ref.ZMode_Hz = ZMode_Hz;

        % SensitivityCal expects a scalar complex frequency (rad/s).
        sensitivity = computeSensitivityReferences(ZminSS, ZMode_rad, ZMode_Hz, ref.Mode, ref.ModeSelect, ObjYbusDss);
        ref.SensMatrix = sensitivity.SensMatrix;
        ref.SensMat_exp = sensitivity.SensMat_exp;
        ref.Ybus_val = sensitivity.Ybus_val;
        ref.Ynodal_val = sensitivity.Ynodal_val;
        ref.Yre_val = sensitivity.Yre_val;
        ref.SensLayer1 = sensitivity.SensLayer1;
        ref.SensLayer2 = sensitivity.SensLayer2;
        ref.Layer12 = sensitivity.Layer12;
        ref.SensitivityModeIndex = sensitivity.SensitivityModeIndex;
    catch err
        warnings{end + 1} = ['Could not compute sensitivity references: ', err.message];
    end
end

%% Frequency-response references
freq = opts.FrequencyHz(:).';
ref.FrequencyHz = freq;
ref.s_values = 1j * 2 * pi * freq;

try
    if baseExists('YsysDss')
        YsysModel = baseValue('YsysDss');
    elseif baseExists('ObjYsysDss')
        ObjYsysDssRaw = baseValue('ObjYsysDss');
        [~, YsysModel] = ObjYsysDssRaw.GetDSS(ObjYsysDssRaw);
    else
        YsysModel = [];
    end
    if ~isempty(YsysModel)
        ref.Ysys_values = SimplusGT.sampleFrequencyResponse(YsysModel, ref.s_values);
    else
        warnings{end + 1} = 'YsysDss/ObjYsysDss was not found; Ysys_values was not exported.';
    end
catch err
    warnings{end + 1} = ['Could not sample Ysys_values: ', err.message];
end

try
    % Zsys(s) = inv(Gm(s) + Ybus(s)) at each frequency.
    % Do NOT use Control Toolbox inv() on the apparatus DSS: Gm is often
    % rank-deficient as a transfer-function inverse (non-finite inv model).
    if baseExists('ObjGm') && baseExists('ObjYbusDss') && baseExists('PortI') && baseExists('PortV')
        ObjGmRaw = baseValue('ObjGm');
        ObjYbusRaw = baseValue('ObjYbusDss');
        PortI = baseValue('PortI');
        PortV = baseValue('PortV');
        [~, GmDssRaw] = ObjGmRaw.GetDSS(ObjGmRaw);
        [~, YbusDssRaw] = ObjYbusRaw.GetDSS(ObjYbusRaw);
        GmTrim = GmDssRaw(PortI, PortV);
        ref.Zsys_values = SimplusGT.sampleImpedanceFrequencyResponse(GmTrim, YbusDssRaw, ref.s_values);
        if ~all(isfinite(ref.Zsys_values(:)))
            warnings{end + 1} = 'Zsys_values contains non-finite entries after algebraic sampling.';
        end
        % WholeSysZ_cal uses inv(GmDSS), which warns/fails when Gm is rank
        % deficient. Keep Zsys_values as the source of truth for Python tests.
        warnings{end + 1} = ['ZminSS was not packed via WholeSysZ_cal because Control Toolbox ', ...
            'inv(Gm) is rank-deficient for this case; use Zsys_values instead.'];
    elseif exist('ZminSS', 'var')
        ref.Zsys_values = SimplusGT.sampleFrequencyResponse(ZminSS, ref.s_values);
    else
        warnings{end + 1} = 'Could not build Zsys; Zsys_values was not exported.';
    end
catch err
    warnings{end + 1} = ['Could not sample Zsys_values: ', err.message];
end

ref.ReferenceWarnings = warnings(:);

if opts.UseV73
    save(outputPath, '-struct', 'ref', '-v7.3');
else
    save(outputPath, '-struct', 'ref', '-v7');
end

fprintf('MATLAB reference exported to: %s\n', outputPath);
if ~isempty(warnings)
    fprintf('Export completed with %d warning(s). Inspect ReferenceWarnings in the MAT file.\n', numel(warnings));
end

end

function ref = copyBaseIfExists(ref, name)
if baseExists(name)
    ref.(name) = baseValue(name);
end
end

function tf = baseExists(name)
tf = evalin('base', sprintf('exist(''%s'', ''var'') ~= 0', name));
end

function value = baseValue(name)
value = evalin('base', name);
end

function validateModeSelect(modeSelect, modeCount)
if isempty(modeSelect)
    error('ModeSelect is empty and no finite modes were found.');
end
if any(modeSelect < 1) || any(modeSelect > modeCount) || any(round(modeSelect) ~= modeSelect)
    error('ModeSelect must contain valid MATLAB 1-based integer mode indices.');
end
end

function models = packStateSpaceCell(sysCell)
models = cell(size(sysCell));
for k = 1:numel(sysCell)
    models{k} = packStateSpace(sysCell{k});
end
end

function model = packStateSpace(sys)
model = struct( ...
    'class', class(sys), ...
    'A', [], 'B', [], 'C', [], 'D', [], 'E', [], ...
    'nx', 0, 'nu', 0, 'ny', 0, ...
    'is_descriptor', false);

if isempty(sys)
    return;
end

model.A = safeFull(getProperty(sys, 'A'));
model.B = safeFull(getProperty(sys, 'B'));
model.C = safeFull(getProperty(sys, 'C'));
model.D = safeFull(getProperty(sys, 'D'));
model.E = safeFull(getProperty(sys, 'E'));
model.is_descriptor = ~isempty(model.E);

if ~isempty(model.A)
    model.nx = size(model.A, 1);
end
if ~isempty(model.B)
    model.nu = size(model.B, 2);
end
if ~isempty(model.C)
    model.ny = size(model.C, 1);
elseif ~isempty(model.D)
    model.ny = size(model.D, 1);
end
end

function value = getProperty(obj, propName)
try
    value = obj.(propName);
catch
    value = [];
end
end

function value = safeFull(value)
if isempty(value)
    return;
end
try
    value = full(value);
catch
end
end

function [Layer1All, Layer2All, Layer1, Layer2, ApparatusSel] = computeApparatusLayer12References(ResidueAll, ZmValAll, apparatusSelOverride)
N_Apparatus = baseValue('NumApparatus');
ApparatusType = baseValue('ApparatusType');
ApparatusBus = baseValue('ApparatusBus');

if isempty(apparatusSelOverride)
    ApparatusSel = defaultApparatusSelection(ApparatusType);
else
    ApparatusSel = apparatusSelOverride(:).';
end

modeCount = numel(ResidueAll);
Layer1All = cell(1, modeCount);
Layer2All = cell(1, modeCount);
Layer1 = cell(1, modeCount);
Layer2 = cell(1, modeCount);

for modei = 1:modeCount
    Residue = ResidueAll{modei};
    ZmVal = ZmValAll{modei};
    l1all = zeros(1, N_Apparatus);
    l2all = zeros(1, N_Apparatus);

    for k = 1:N_Apparatus
        if isSupportedApparatusType(ApparatusType{k}) && ~isempty(Residue{k}) && ~isempty(ZmVal{k})
            l1all(k) = norm(Residue{k}, 'fro') * norm(ZmVal{k}, 'fro');
            l2all(k) = -1 * conj(sum(dot(Residue{k}, ZmVal{k}')));
        end
    end

    [l1, l2] = SimplusGT.Modal.MdLayer12(Residue, ZmVal, N_Apparatus, ApparatusBus, ApparatusType, ApparatusSel);
    Layer1All{modei} = l1all;
    Layer2All{modei} = l2all;
    Layer1{modei} = l1;
    Layer2{modei} = l2;
end
end

function apparatusSel = defaultApparatusSelection(ApparatusType)
apparatusSel = [];
for k = 1:numel(ApparatusType)
    if isSupportedApparatusType(ApparatusType{k})
        apparatusSel(end + 1) = k; %#ok<AGROW>
    end
end
end

function tf = isSupportedApparatusType(appType)
tf = (appType <= 89) || ...
    (appType >= 1000 && appType <= 1089) || ...
    (appType >= 2000 && appType <= 2089);
end

function sensitivity = computeSensitivityReferences(ZminSS, ZMode_rad, ZMode_Hz, Mode, ModeSelect, ObjYbusDss)
modeCount = numel(ModeSelect);
sensitivity = struct();
sensitivity.SensMatrix = cell(1, modeCount);
sensitivity.SensMat_exp = cell(1, modeCount);
sensitivity.Ybus_val = cell(1, modeCount);
sensitivity.Ynodal_val = cell(1, modeCount);
sensitivity.Yre_val = cell(1, modeCount);
sensitivity.SensLayer1 = cell(1, modeCount);
sensitivity.SensLayer2 = cell(1, modeCount);
sensitivity.Layer12 = cell(1, modeCount);
sensitivity.SensitivityModeIndex = zeros(1, modeCount);

for modei = 1:modeCount
    lambdaHz = Mode(ModeSelect(modei));
    [~, Ek] = min(abs(lambdaHz - ZMode_Hz));
    if abs(lambdaHz - ZMode_Hz(Ek)) > 1e-3
        error('Could not match Ysys mode %d to Zsys mode within tolerance.', ModeSelect(modei));
    end

    lambdaRad = ZMode_rad(Ek);
    [SensMatrix, Ybus_val, Ynodal_val, Yre_val, SensMat_exp] = ...
        SimplusGT.Modal.SensitivityCal(ZminSS, Ek, lambdaRad, ObjYbusDss);
    [SensLayer1, SensLayer2, Layer12] = ...
        SimplusGT.Modal.SensLayer12(SensMatrix, Yre_val, modei, ZMode_Hz(Ek));

    sensitivity.SensMatrix{modei} = SensMatrix;
    sensitivity.SensMat_exp{modei} = SensMat_exp;
    sensitivity.Ybus_val{modei} = Ybus_val;
    sensitivity.Ynodal_val{modei} = Ynodal_val;
    sensitivity.Yre_val{modei} = Yre_val;
    sensitivity.SensLayer1{modei} = SensLayer1;
    sensitivity.SensLayer2{modei} = SensLayer2;
    sensitivity.Layer12{modei} = Layer12;
    sensitivity.SensitivityModeIndex(modei) = Ek;
end
end
