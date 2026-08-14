% Use the toolbox results to acquire whole-system impedance model
% Output: the state-space model of whole-system impedance, or [] if a
% proper finite realization cannot be formed.
%
% Method: Zsys = inv(Gm + Ybus)
% Equivalent to feedback(inv(Gm), Ybus), but avoids Control Toolbox
% inv(Gm) alone, which is often rank-deficient (0-state model).
%
% For many networks (e.g. IEEE 14 RL lines), inv(Gm+Ybus) is improper or
% non-finite as a state-space realization. Callers must handle [].
%
% Author: Yue Zhu

function Zsys_SS = WholeSysZ_cal(GmObj,YbusObj,Port_i,Port_v)

[~,Gm_dss] = GmObj.GetDSS(GmObj);
[~,YbusDSS] = YbusObj.GetDSS(YbusObj);
Gm_dss_trim = Gm_dss(Port_i,Port_v);

Ysys_port = Gm_dss_trim + YbusDSS;

warnState = warning('off', 'all');
try
    Zsys_dss = inv(Ysys_port);
catch
    Zsys_dss = [];
end
warning(warnState);

Zsys_SS = [];
if ~isempty(Zsys_dss)
    try
        if isempty(Zsys_dss.E)
            % Regular ss (empty E): keep as-is; SimplusGT.dss2ss would wipe states.
            Zsys_SS = ss(Zsys_dss);
        else
            Zsys_SS = SimplusGT.dss2ss(Zsys_dss);
        end
        if isempty(Zsys_SS) || size(Zsys_SS.A,1) == 0 || ~all(isfinite(Zsys_SS.A(:)))
            Zsys_SS = [];
        end
    catch
        Zsys_SS = [];
    end
end

end
