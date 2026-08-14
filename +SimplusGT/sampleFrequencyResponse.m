function values = sampleFrequencyResponse(model, sValues)
%SAMPLEFREQUENCYRESPONSE Evaluate a DSS/SS model on a complex-frequency grid.
%
%   values(:,:,k) is the transfer matrix at sValues(k).
%   Shape: (nOut, nIn, nFreq)

first = SimplusGT.evalDescriptorFr(model, sValues(1));
values = zeros(size(first, 1), size(first, 2), numel(sValues));
values(:, :, 1) = first;
for i = 2:numel(sValues)
    values(:, :, i) = SimplusGT.evalDescriptorFr(model, sValues(i));
end
end
