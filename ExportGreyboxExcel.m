function ExportGreyboxExcel(outputPath, varargin)
%EXPORTGREYBOXEXCEL Write Ysys/Zsys frequency data to a Python-compatible Excel workbook.
%
% Run after SimplusGT.Toolbox.Main() has populated the MATLAB base workspace:
%
%   UserDataName = 'IEEE_14Bus';
%   UserDataType = 0;
%   SimplusGT.Toolbox.Main();
%   ExportGreyboxExcel('Results/IEEE_14Bus_greybox.xlsx', ...
%       'FrequencyHz', logspace(-1, 3, 80));
%
% The workbook layout matches Python simplusgt.export.export_greybox_excel:
%   Summary, Channels, Channels_Zsys, Ysys, Zsys, Ysys_MagPhase, Zsys_MagPhase,
%   Ysys_RealImag, Zsys_RealImag, and optional Eigenvalues / StatePF /
%   Layer1 / Layer2 / Layer3 / Sens_Layer12.
%
% Optional name-value arguments:
%   FrequencyHz  - Frequency grid in Hz. Default: logspace(-1, 3, 80).
%   RunMain      - If true, run SimplusGT.Toolbox.Main() first.
%   UserDataName - Case name used when RunMain is true, and for the default filename.
%   UserDataType - 0 for JSON, 1 for Excel. Used when RunMain is true.
%   IncludeYsys  - If false, skip Ysys sheets. Default: true.
%
% Row/Col in the long tables are 0-based so they match the Python Excel export.

if nargin < 1
    outputPath = '';
end

p = inputParser;
addParameter(p, 'FrequencyHz', logspace(-1, 3, 80), @(x) isnumeric(x) && isvector(x));
addParameter(p, 'RunMain', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'UserDataName', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'UserDataType', 0, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'IncludeYsys', true, @(x) islogical(x) || isnumeric(x));
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

if ~baseExists('ObjGm') || ~baseExists('ObjYbusDss') || ~baseExists('PortI') || ~baseExists('PortV')
    error(['ExportGreyboxExcel requires SimplusGT.Toolbox.Main() to have been run. ', ...
        'Missing ObjGm, ObjYbusDss, PortI, and/or PortV in the base workspace.']);
end

caseName = char(opts.UserDataName);
if isempty(caseName) && baseExists('UserDataName')
    caseName = char(baseValue('UserDataName'));
end
if isempty(outputPath)
    if isempty(caseName)
        caseName = 'greybox';
    end
    outputPath = fullfile(pwd, 'Results', [caseName, '_greybox.xlsx']);
end
outputPath = char(outputPath);
outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if exist(outputPath, 'file') == 2
    delete(outputPath);
end

warnings = {};
freq = opts.FrequencyHz(:).';
sValues = 1j * 2 * pi * freq;

ObjGm = baseValue('ObjGm');
ObjYbusDss = baseValue('ObjYbusDss');
PortI = baseValue('PortI');
PortV = baseValue('PortV');
[~, GmDssRaw] = ObjGm.GetDSS(ObjGm);
[~, YbusDssRaw] = ObjYbusDss.GetDSS(ObjYbusDss);
GmTrim = GmDssRaw(PortI, PortV);

[ysysOutLabels, ysysInLabels, zsysOutLabels, zsysInLabels] = resolvePortLabels(ObjGm, PortI, PortV);

ysysValues = [];
if opts.IncludeYsys
    try
        if baseExists('YsysDss')
            YsysModel = baseValue('YsysDss');
        elseif baseExists('ObjYsysDss')
            ObjYsysDssRaw = baseValue('ObjYsysDss');
            [~, YsysModel] = ObjYsysDssRaw.GetDSS(ObjYsysDssRaw);
        else
            YsysModel = [];
        end
        if isempty(YsysModel)
            warnings{end + 1} = 'YsysDss/ObjYsysDss was not found; Ysys sheets were not written.'; %#ok<AGROW>
        else
            ysysValues = SimplusGT.sampleFrequencyResponse(YsysModel, sValues);
        end
    catch err
        warnings{end + 1} = ['Could not sample Ysys: ', err.message]; %#ok<AGROW>
    end
end

try
    zsysValues = SimplusGT.sampleImpedanceFrequencyResponse(GmTrim, YbusDssRaw, sValues);
    if ~all(isfinite(zsysValues(:)))
        nBad = nnz(~isfinite(zsysValues(:)));
        warnings{end + 1} = sprintf('Zsys contains %d non-finite entries; they are still written.', nBad); %#ok<AGROW>
    end
catch err
    error('Could not sample Zsys: %s', err.message);
end

summary = buildSummaryTable(caseName, freq, warnings, ~isempty(ysysValues));
writetable(summary, outputPath, 'Sheet', 'Summary');

if ~isempty(ysysValues)
    writetable(buildChannelTable('Ysys', ysysOutLabels, ysysInLabels), outputPath, 'Sheet', 'Channels');
end
writetable(buildChannelTable('Zsys', zsysOutLabels, zsysInLabels), outputPath, 'Sheet', 'Channels_Zsys');

if ~isempty(ysysValues)
    writeTransferSheets(outputPath, 'Ysys', freq, ysysValues, ysysOutLabels, ysysInLabels);
end
writeTransferSheets(outputPath, 'Zsys', freq, zsysValues, zsysOutLabels, zsysInLabels);

writeOptionalLayerSheets(outputPath, warnings);

fprintf('Greybox Excel exported to: %s\n', outputPath);
if ~isempty(warnings)
    fprintf('Export completed with %d warning(s):\n', numel(warnings));
    for k = 1:numel(warnings)
        fprintf('  - %s\n', warnings{k});
    end
end
end

function tf = baseExists(name)
tf = evalin('base', sprintf('exist(''%s'', ''var'') ~= 0', name));
end

function value = baseValue(name)
value = evalin('base', name);
end

function [ysysOut, ysysIn, zsysOut, zsysIn] = resolvePortLabels(ObjGm, PortI, PortV)
% Ysys maps voltage -> current (Gm trim). Zsys maps current -> voltage (inverse).
try
    [~, gmIn, gmOut] = ObjGm.GetString(ObjGm);
    ysysIn = pickLabels(gmIn, PortV);
    ysysOut = pickLabels(gmOut, PortI);
catch
    ysysIn = fallbackVoltageLabels(numel(PortV));
    ysysOut = fallbackCurrentLabels(numel(PortI));
end
% Inverse of Gm trim: inputs become former outputs (currents), outputs former inputs (voltages).
zsysIn = ysysOut;
zsysOut = ysysIn;
end

function labels = pickLabels(cellStr, indices)
labels = cell(1, numel(indices));
for k = 1:numel(indices)
    idx = indices(k);
    if iscell(cellStr) && idx >= 1 && idx <= numel(cellStr)
        labels{k} = char(cellStr{idx});
    else
        labels{k} = sprintf('ch%d', k);
    end
end
end

function labels = fallbackVoltageLabels(n)
labels = cell(1, n);
if n >= 2 && mod(n, 2) == 0
    for k = 1:n/2
        labels{2*k-1} = sprintf('v_d%d', k);
        labels{2*k} = sprintf('v_q%d', k);
    end
else
    for k = 1:n
        labels{k} = sprintf('v%d', k);
    end
end
end

function labels = fallbackCurrentLabels(n)
labels = cell(1, n);
if n >= 2 && mod(n, 2) == 0
    for k = 1:n/2
        labels{2*k-1} = sprintf('i_d%d', k);
        labels{2*k} = sprintf('i_q%d', k);
    end
else
    for k = 1:n
        labels{k} = sprintf('i%d', k);
    end
end
end

function T = buildSummaryTable(caseName, freq, warnings, ysysOk)
busCount = optionalNumeric('NumBus');
lineCount = optionalNumeric('NumLine');
if isnan(lineCount) && baseExists('ListLine')
    lineCount = size(baseValue('ListLine'), 1);
end
appCount = optionalNumeric('NumApparatus');
warnText = strjoin(warnings, ' | ');
if isempty(warnText)
    warnText = '';
end
Property = { ...
    'case_path'; 'bus_count'; 'line_count'; 'apparatus_count'; ...
    'layers'; 'modes'; 'freq_count'; 'freq_min_hz'; 'freq_max_hz'; ...
    'ysys_sampled'; 'zsys_sampled'; 'warnings'};
Value = { ...
    caseName; ...
    busCount; ...
    lineCount; ...
    appCount; ...
    ''; ...
    ''; ...
    numel(freq); ...
    min(freq); ...
    max(freq); ...
    false; ...
    false; ...
    warnText};
if ~ysysOk
    Value{10} = '';
end
T = table(Property, Value);
end

function n = optionalNumeric(name)
if ~baseExists(name)
    n = NaN;
    return;
end
raw = baseValue(name);
if isnumeric(raw) && isscalar(raw)
    n = double(raw);
else
    n = NaN;
end
end

function T = buildChannelTable(prefix, outLabels, inLabels)
nOut = numel(outLabels);
nIn = numel(inLabels);
n = nOut * nIn;
transfer = repmat({prefix}, n, 1);
row = zeros(n, 1);
col = zeros(n, 1);
output = strings(n, 1);
input = strings(n, 1);
channel = strings(n, 1);
k = 0;
for r = 1:nOut
    for c = 1:nIn
        k = k + 1;
        row(k) = r - 1;
        col(k) = c - 1;
        output(k) = string(outLabels{r});
        input(k) = string(inLabels{c});
        channel(k) = output(k) + "/" + input(k);
    end
end
T = table(transfer, row, col, output, input, channel);
end

function writeTransferSheets(outputPath, name, freq, values, outLabels, inLabels)
writetable(buildLongTable(freq, values, outLabels, inLabels), outputPath, 'Sheet', name);
wideMag = buildWideMagPhaseTable(freq, values, outLabels, inLabels);
if ~isempty(wideMag)
    writetable(wideMag, outputPath, 'Sheet', sanitizeSheetName([name, '_MagPhase']));
end
wideRi = buildWideRealImagTable(freq, values, outLabels, inLabels);
if ~isempty(wideRi)
    writetable(wideRi, outputPath, 'Sheet', sanitizeSheetName([name, '_RealImag']));
end
end

function T = buildLongTable(freq, values, outLabels, inLabels)
nFreq = numel(freq);
nOut = numel(outLabels);
nIn = numel(inLabels);
n = nFreq * nOut * nIn;
Frequency_Hz = zeros(n, 1);
Output = strings(n, 1);
Input = strings(n, 1);
Row = zeros(n, 1);
Col = zeros(n, 1);
Mag = zeros(n, 1);
Phase_deg = zeros(n, 1);
Real = zeros(n, 1);
Imag = zeros(n, 1);
k = 0;
for fIdx = 1:nFreq
    matrix = values(:, :, fIdx);
    for r = 1:nOut
        for c = 1:nIn
            k = k + 1;
            z = matrix(r, c);
            Frequency_Hz(k) = freq(fIdx);
            Output(k) = string(outLabels{r});
            Input(k) = string(inLabels{c});
            Row(k) = r - 1;
            Col(k) = c - 1;
            Mag(k) = abs(z);
            Phase_deg(k) = angle(z) * 180 / pi;
            Real(k) = real(z);
            Imag(k) = imag(z);
        end
    end
end
T = table(Frequency_Hz, Output, Input, Mag, Phase_deg, Real, Imag);
T.Properties.DimensionNames = {'Observation', 'Variable'};
T.Row = Row;
T.Col = Col;
T = movevars(T, {'Row', 'Col'}, 'After', 'Input');
end

function T = buildWideMagPhaseTable(freq, values, outLabels, inLabels)
nOut = numel(outLabels);
nIn = numel(inLabels);
if 1 + 2 * nOut * nIn > 16384
    T = [];
    return;
end
nFreq = numel(freq);
data = nan(nFreq, 1 + 2 * nOut * nIn);
varNames = cell(1, 1 + 2 * nOut * nIn);
data(:, 1) = freq(:);
varNames{1} = 'Frequency_Hz';
col = 1;
for r = 1:nOut
    for c = 1:nIn
        tag = safeHeader([outLabels{r}, '__', inLabels{c}]);
        channel = squeeze(values(r, c, :));
        col = col + 1;
        data(:, col) = abs(channel);
        varNames{col} = ['Mag_', tag];
        col = col + 1;
        data(:, col) = angle(channel) * 180 / pi;
        varNames{col} = ['Phase_', tag];
    end
end
T = array2table(data, 'VariableNames', varNames);
end

function T = buildWideRealImagTable(freq, values, outLabels, inLabels)
nOut = numel(outLabels);
nIn = numel(inLabels);
if 1 + 2 * nOut * nIn > 16384
    T = [];
    return;
end
nFreq = numel(freq);
data = nan(nFreq, 1 + 2 * nOut * nIn);
varNames = cell(1, 1 + 2 * nOut * nIn);
data(:, 1) = freq(:);
varNames{1} = 'Frequency_Hz';
col = 1;
for r = 1:nOut
    for c = 1:nIn
        tag = safeHeader([outLabels{r}, '__', inLabels{c}]);
        channel = squeeze(values(r, c, :));
        col = col + 1;
        data(:, col) = real(channel);
        varNames{col} = ['Real_', tag];
        col = col + 1;
        data(:, col) = imag(channel);
        varNames{col} = ['Imag_', tag];
    end
end
T = array2table(data, 'VariableNames', varNames);
end

function tag = safeHeader(text)
tag = regexprep(char(string(text)), '[^\w\-]+', '_');
tag = regexprep(tag, '^_+|_+$', '');
if isempty(tag)
    tag = 'ch';
elseif numel(tag) > 40
    tag = tag(1:40);
end
end

function name = sanitizeSheetName(name)
name = regexprep(char(string(name)), '[\[\]\*\:\?\/\\]', '_');
if numel(name) > 31
    name = name(1:31);
end
end

function writeOptionalLayerSheets(outputPath, warnings)
% Eigenvalues from whole-system GsysSs (available after Main).
if baseExists('GsysSs')
    T = eigenvaluesTable(baseValue('GsysSs'));
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'Eigenvalues');
    end
elseif baseExists('MdMode')
    T = mdModeEigenvaluesTable(baseValue('MdMode'));
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'Eigenvalues');
    end
end
% State participation from ModalAnalysis (selected modes/states).
if baseExists('MdStatePF')
    T = statePfTable(baseValue('MdStatePF'));
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'StatePF');
    end
end
if baseExists('MdLayer1')
    layer1 = baseValue('MdLayer1');
    T = layer1Table(layer1);
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'Layer1');
    end
end
if baseExists('MdLayer2')
    layer2 = baseValue('MdLayer2');
    T = layer2Table(layer2);
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'Layer2');
    end
end
if baseExists('MdLayer3')
    layer3 = baseValue('MdLayer3');
    T = layer3Table(layer3);
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'Layer3');
    end
end
if baseExists('MdSensResult')
    sens = baseValue('MdSensResult');
    T = sensLayer12Table(sens);
    if ~isempty(T)
        writetable(T, outputPath, 'Sheet', 'Sens_Layer12');
    end
end
if nargin >= 2 && ~isempty(warnings)
    % warnings is unused here; kept so callers can extend later.
end
end

function T = eigenvaluesTable(GsysSs)
try
    A = GsysSs.A;
catch
    T = [];
    return;
end
eigsRad = eig(A);
eigsRad = eigsRad(isfinite(eigsRad));
if isempty(eigsRad)
    T = [];
    return;
end
% Sort like Python: real ascending, then |imag|.
[~, ord] = sortrows([real(eigsRad), abs(imag(eigsRad))]);
eigsRad = eigsRad(ord);
eigsHz = eigsRad / (2 * pi);
rows = cell(numel(eigsRad), 7);
for i = 1:numel(eigsRad)
    lam = eigsRad(i);
    lamHz = eigsHz(i);
    denom = abs(lam);
    if denom == 0
        zeta = NaN;
    else
        zeta = -real(lam) / denom;
    end
    rows(i, :) = {i - 1, real(lam), imag(lam), real(lamHz), imag(lamHz), ...
        abs(imag(lamHz)), zeta};
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'real_rad_s', 'imag_rad_s', 'real_hz', 'imag_hz', ...
     'frequency_hz', 'damping_ratio'});
end

function T = mdModeEigenvaluesTable(MdMode)
if ~(isnumeric(MdMode) || isvector(MdMode))
    T = [];
    return;
end
eigsHz = MdMode(:);
eigsHz = eigsHz(isfinite(eigsHz));
if isempty(eigsHz)
    T = [];
    return;
end
eigsRad = eigsHz * (2 * pi);
rows = cell(numel(eigsHz), 7);
for i = 1:numel(eigsHz)
    lam = eigsRad(i);
    lamHz = eigsHz(i);
    denom = abs(lam);
    if denom == 0
        zeta = NaN;
    else
        zeta = -real(lam) / denom;
    end
    rows(i, :) = {i - 1, real(lam), imag(lam), real(lamHz), imag(lamHz), ...
        abs(imag(lamHz)), zeta};
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'real_rad_s', 'imag_rad_s', 'real_hz', 'imag_hz', ...
     'frequency_hz', 'damping_ratio'});
end

function T = statePfTable(MdStatePF)
if ~isstruct(MdStatePF)
    T = [];
    return;
end
rows = {};
for modei = 1:numel(MdStatePF)
    item = MdStatePF(modei);
    if ~isfield(item, 'result') || isempty(item.result)
        continue;
    end
    modeLabel = '';
    if isfield(item, 'mode')
        modeLabel = char(string(item.mode));
    end
    for statei = 1:numel(item.result)
        rec = item.result(statei);
        apparatus = '';
        if isfield(rec, 'Apparatus')
            apparatus = cellstrScalar(rec.Apparatus);
        end
        stateName = '';
        if isfield(rec, 'State')
            stateName = cellstrScalar(rec.State);
        end
        pfAbs = fieldOrNan(rec, 'PF_ABS');
        pfRe = fieldOrNan(rec, 'PF_Real');
        pfIm = fieldOrNan(rec, 'PF_Imag');
        if isnan(pfAbs) && isfield(rec, 'PF')
            pf = rec.PF;
            pfAbs = abs(pf);
            pfRe = real(pf);
            pfIm = imag(pf);
        end
        rows(end + 1, :) = {modei - 1, modeLabel, statei - 1, apparatus, stateName, pfRe, pfIm, pfAbs}; %#ok<AGROW>
    end
end
if isempty(rows)
    T = [];
    return;
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'mode', 'state_index', 'apparatus', 'state', 'pf_real', 'pf_imag', 'pf_abs'});
end

function T = layer1Table(layer1)
if ~isstruct(layer1)
    T = [];
    return;
end
rows = {};
for modei = 1:numel(layer1)
    item = layer1(modei);
    if ~isfield(item, 'result') || isempty(item.result)
        continue;
    end
    modeLabel = '';
    if isfield(item, 'mode')
        modeLabel = char(string(item.mode));
    end
    for count = 1:numel(item.result)
        rec = item.result(count);
        label = '';
        if isfield(rec, 'Apparatus')
            label = cellstrScalar(rec.Apparatus);
        end
        value = NaN;
        if isfield(rec, 'Abs_Max')
            value = rec.Abs_Max;
        end
        rows(end + 1, :) = {modei - 1, modeLabel, count, label, value}; %#ok<AGROW>
    end
end
if isempty(rows)
    T = [];
    return;
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'mode', 'apparatus_index', 'label', 'value'});
end

function T = layer2Table(layer2)
if ~isstruct(layer2)
    T = [];
    return;
end
rows = {};
for modei = 1:numel(layer2)
    item = layer2(modei);
    if ~isfield(item, 'result') || isempty(item.result)
        continue;
    end
    modeLabel = '';
    if isfield(item, 'mode')
        modeLabel = char(string(item.mode));
    end
    for count = 1:numel(item.result)
        rec = item.result(count);
        label = '';
        if isfield(rec, 'Apparatus')
            label = cellstrScalar(rec.Apparatus);
        end
        re = fieldOrNan(rec, 'DeltaLambdaReal');
        im = fieldOrNan(rec, 'DeltaLambdaImag');
        rePu = fieldOrNan(rec, 'DeltaLambdaRealpu');
        imPu = fieldOrNan(rec, 'DeltaLambdaImagpu');
        rows(end + 1, :) = {modei - 1, modeLabel, count, label, re, im, rePu, imPu}; %#ok<AGROW>
    end
end
if isempty(rows)
    T = [];
    return;
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'mode', 'apparatus_index', 'label', 'real', 'imag', 'real_normalized', 'imag_normalized'});
end

function T = layer3Table(layer3)
if ~isstruct(layer3)
    T = [];
    return;
end
rows = {};
for modei = 1:numel(layer3)
    item = layer3(modei);
    if ~isfield(item, 'result') || isempty(item.result)
        continue;
    end
    modeLabel = '';
    if isfield(item, 'mode')
        modeLabel = char(string(item.mode));
    end
    appResults = item.result;
    for appCount = 1:numel(appResults)
        appRec = appResults(appCount);
        label = '';
        if isfield(appRec, 'Apparatus')
            label = cellstrScalar(appRec.Apparatus);
        end
        if ~isfield(appRec, 'Result') || isempty(appRec.Result)
            continue;
        end
        for k = 1:numel(appRec.Result)
            paramRec = appRec.Result(k);
            paramName = '';
            if isfield(paramRec, 'ParaName')
                paramName = cellstrScalar(paramRec.ParaName);
            end
            dRad = fieldOrNan(paramRec, 'DLambda_rad');
            dHz = fieldOrNan(paramRec, 'DLambdaRho_Hz');
            dPuHz = fieldOrNan(paramRec, 'DLambdaRho_pu_Hz');
            rows(end + 1, :) = { ...
                modei - 1, modeLabel, appCount, label, paramName, ...
                real(dRad), imag(dRad), ...
                real(dHz), imag(dHz), ...
                real(dPuHz), imag(dPuHz)}; %#ok<AGROW>
        end
    end
end
if isempty(rows)
    T = [];
    return;
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'mode', 'apparatus_index', 'label', 'parameter', ...
     'd_lambda_rad_real', 'd_lambda_rad_imag', ...
     'd_lambda_hz_real', 'd_lambda_hz_imag', ...
     'd_lambda_pu_hz_real', 'd_lambda_pu_hz_imag'});
end

function T = sensLayer12Table(sens)
if ~isstruct(sens)
    T = [];
    return;
end
rows = {};
for modei = 1:numel(sens)
    item = sens(modei);
    if ~isfield(item, 'Layer12') || isempty(item.Layer12)
        continue;
    end
    modeLabel = '';
    if isfield(item, 'mode')
        modeLabel = char(string(item.mode));
    end
    layer12 = item.Layer12;
    for k = 1:numel(layer12)
        rec = layer12(k);
        component = '';
        if isfield(rec, 'Component')
            component = char(string(rec.Component));
        end
        l1 = fieldOrNan(rec, 'L1val_norm');
        l2r = fieldOrNan(rec, 'L2val_real_norm');
        l2i = fieldOrNan(rec, 'L2val_imag_norm');
        rows(end + 1, :) = {modei - 1, modeLabel, component, l1, l2r, l2i}; %#ok<AGROW>
    end
end
if isempty(rows)
    T = [];
    return;
end
T = cell2table(rows, 'VariableNames', ...
    {'mode_index', 'mode', 'component', 'layer1_normalized', 'layer2_real_normalized', 'layer2_imag_normalized'});
end

function text = cellstrScalar(value)
if iscell(value)
    text = char(string(value{1}));
else
    text = char(string(value));
end
end

function value = fieldOrNan(rec, name)
if isfield(rec, name)
    value = rec.(name);
else
    value = NaN;
end
end
