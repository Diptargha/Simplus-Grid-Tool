function H = evalDescriptorFr(model, s)
%EVALDESCRIPTORFR Evaluate a state-space or descriptor model at one s.
% Uses (sE-A)\B with a pseudoinverse fallback for singular pencils.

A = full(model.A);
B = full(model.B);
C = full(model.C);
D = full(model.D);
if isempty(A)
    H = D;
    return;
end
try
    E = full(model.E);
catch
    E = [];
end
if isempty(E)
    E = eye(size(A));
end
lhs = s * E - A;
[dynamic, rcondVal] = linsolve(lhs, B);
if isempty(rcondVal) || ~(isfinite(rcondVal) && rcondVal > 1e-12)
    dynamic = pinv(lhs) * B;
end
H = C * dynamic + D;
end
