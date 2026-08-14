function values = sampleImpedanceFrequencyResponse(GmTrim, YbusModel, sValues)
%SAMPLEIMPEDANCEFREQUENCYRESPONSE Pointwise Zsys = inv(Gm + Ybus).
% Equivalent to feedback(inv(Gm), Ybus) without forming inv(Gm) as a DSS/SS.
%
%   values(:,:,k) is Zsys at sValues(k).
%   Shape: (nOut, nIn, nFreq)

firstGm = SimplusGT.evalDescriptorFr(GmTrim, sValues(1));
n = size(firstGm, 1);
values = complex(zeros(n, n, numel(sValues)));
I = eye(n);
for i = 1:numel(sValues)
    gm = SimplusGT.evalDescriptorFr(GmTrim, sValues(i));
    ybus = SimplusGT.evalDescriptorFr(YbusModel, sValues(i));
    values(:, :, i) = (gm + ybus) \ I;
end
end
