"""Apparatus model framework and concrete Python ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dss import DescriptorStateSpace, append, switch_inputs_outputs


def _para_value(params: dict, name: str, default: float = 0.0) -> float:
    aliases = {
        "V_dc": ("V_dc", "v_dc_r", "v_dc_ref", "Vdc"),
        "C_dc": ("C_dc", "Cdc"),
        "wLf": ("wLf", "wL"),
        "Rf": ("Rf", "R"),
        "fvdq": ("fvdq", "f_v_dq"),
        "fidq": ("fidq", "f_i_dq"),
        "fdroop": ("fdroop", "f_droop"),
        "fvdc": ("fvdc", "f_v_dc"),
        "fpll": ("fpll", "f_pll"),
        "f_tau_pll": ("f_tau_pll",),
        "f_i_dq": ("f_i_dq", "fidq"),
        "fidc": ("fidc", "f_i_sdq"),
        "Vdc": ("Vdc", "V_dc", "v_dc_r", "v_dc_ref"),
        "Cdc": ("Cdc", "C_dc"),
        "fi": ("fi", "f_i_dq"),
    }
    value = default
    for key in aliases.get(name, (name,)):
        if key in params:
            value = params[key]
            break
    if isinstance(value, str):
        return float(value)
    return float(value)


def _add_bus_suffix(names: Sequence[str], buses: tuple[int, ...]) -> list[str]:
    if len(buses) == 1:
        return [f"{name}{buses[0]}" for name in names]
    suffix = "-".join(str(bus) for bus in buses)
    return [f"{name}{suffix}" for name in names]


class ApparatusModel:
    def __init__(self, apparatus_type: int, params: dict, power_flow: np.ndarray, buses: tuple[int, ...], ts: float = 0.0):
        self.apparatus_type = apparatus_type
        self.params = params
        self.power_flow = np.asarray(power_flow, dtype=float)
        self.buses = buses
        self.ts = ts
        self._equilibrium: tuple[np.ndarray, np.ndarray, float] | None = None

    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        raise NotImplementedError

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        raise NotImplementedError

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        raise NotImplementedError

    def output_at_equilibrium(self) -> np.ndarray:
        x, u, _ = self.get_equilibrium()
        return self.state_space_equation(x, u, 2)

    def get_equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self._equilibrium is None:
            self._equilibrium = self.equilibrium()
        return self._equilibrium

    def linearize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_e, u_e, _ = self.get_equilibrium()
        dx_e = self.state_space_equation(x_e, u_e, 1)
        y_e = self.state_space_equation(x_e, u_e, 2)
        lx, lu, ly = len(x_e), len(u_e), len(y_e)
        a = np.zeros((lx, lx))
        b = np.zeros((lx, lu))
        c = np.zeros((ly, lx))
        d = np.zeros((ly, lu))
        perturb_factor = 1e-6
        for idx in range(lx):
            x_p = x_e.copy()
            perturb = perturb_factor * abs(1 + abs(x_e[idx]))
            x_p[idx] += perturb
            a[:, idx] = (self.state_space_equation(x_p, u_e, 1) - dx_e) / perturb
            c[:, idx] = (self.state_space_equation(x_p, u_e, 2) - y_e) / perturb
        for idx in range(lu):
            u_p = u_e.copy()
            perturb = perturb_factor * abs(1 + abs(u_e[idx]))
            u_p[idx] += perturb
            b[:, idx] = (self.state_space_equation(x_e, u_p, 1) - dx_e) / perturb
            d[:, idx] = (self.state_space_equation(x_e, u_p, 2) - y_e) / perturb
        return a, b, c, d

    def to_dss(self) -> DescriptorStateSpace:
        state_names, input_names, output_names = self.signal_list()
        a, b, c, d = self.linearize()
        model = DescriptorStateSpace.from_state_space(
            a,
            b,
            c,
            d,
            states=_add_bus_suffix(state_names, self.buses),
            inputs=_add_bus_suffix(input_names, self.buses),
            outputs=_add_bus_suffix(output_names, self.buses),
        )
        return model


class InfiniteBusAc(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        return [], ["i_d", "i_q"], ["v_d", "v_q", "w"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, q, v, xi, _ = self.power_flow
        return np.array([]), np.array([p / v, -q / v]), float(xi)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        if flag == 1:
            return np.array([])
        return np.array([self.power_flow[2], 0.0, self.power_flow[4]])


class FloatingBusAc(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        return [], ["v_d", "v_q"], ["i_d", "i_q"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        return np.array([]), np.array([self.power_flow[2], 0.0]), float(self.power_flow[3])

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        return np.array([]) if flag == 1 else np.array([0.0, 0.0])


class InfiniteBusDc(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        return [], ["i"], ["v"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, _, v, xi, _ = self.power_flow
        return np.array([]), np.array([p / v]), float(xi)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        return np.array([]) if flag == 1 else np.array([self.power_flow[2]])


class FloatingBusDc(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        return [], ["v"], ["i"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        return np.array([]), np.array([self.power_flow[2]]), float(self.power_flow[3])

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        return np.array([]) if flag == 1 else np.array([0.0])


class SynchronousMachine(ApparatusModel):
    def __init__(self, apparatus_type: int, params: dict, power_flow: np.ndarray, buses: tuple[int, ...], ts: float = 0.0):
        super().__init__(apparatus_type, params, power_flow, buses, ts)
        self.psi_f = 0.0

    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        return ["i_d", "i_q", "w", "theta"], ["v_d", "v_q", "T_m", "v_ex"], ["i_d", "i_q", "w", "i_ex", "theta"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, q, v, xi, w = self.power_flow
        d = _para_value(self.params, "D")
        wl = _para_value(self.params, "wL")
        r = _para_value(self.params, "R")
        w0 = _para_value(self.params, "w0")
        d = d / (w0 ** 2)
        inductance = wl / w0
        i_d_global = p / v
        i_q_global = -q / v
        i_dq_global = i_d_global + 1j * i_q_global
        e_dq = v - i_dq_global * (r + 1j * inductance * w)
        arg_e = np.angle(e_dq)
        xi_new = xi + arg_e
        v_dq = v * np.exp(-1j * arg_e)
        i_dq = i_dq_global * np.exp(-1j * arg_e)
        self.psi_f = abs(e_dq) / w
        t_m = self.psi_f * float(np.real(i_dq)) - d * w
        x_e = np.array([np.real(i_dq), np.imag(i_dq), w, xi_new], dtype=float)
        u_e = np.array([np.real(v_dq), np.imag(v_dq), t_m, 0.0], dtype=float)
        return x_e, u_e, float(xi_new)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        i_d, i_q, w, theta = x
        v_d, v_q, t_m, _v_ex = u
        j = _para_value(self.params, "J") * 2 / (_para_value(self.params, "w0") ** 2)
        d = _para_value(self.params, "D") / (_para_value(self.params, "w0") ** 2)
        wl = _para_value(self.params, "wL")
        r = _para_value(self.params, "R")
        w0 = _para_value(self.params, "w0")
        inductance = wl / w0
        if flag == 1:
            psi_d = inductance * i_d
            psi_q = inductance * i_q - self.psi_f
            if self.apparatus_type == 0:
                te = self.psi_f * i_d
                di_d = (v_d - r * i_d + w * psi_q) / inductance
                di_q = (v_q - r * i_q - w * psi_d) / inductance
                dw = (te - t_m - d * w) / j
            else:
                pe = self.psi_f * w0 * i_d
                di_d = (v_d - r * i_d + w * inductance * i_q - self.psi_f * w0) / inductance
                di_q = (v_q - r * i_q - w * inductance * i_d) / inductance
                dw = (pe - t_m * w0 - d * w * w0) / (j * w0)
            return np.array([di_d, di_q, dw, w], dtype=float)
        return np.array([i_d, i_q, w, 0.0, theta], dtype=float)


class GridFeedingBuck(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        states = ["i", "i_i"] if self.apparatus_type == 1010 else ["i", "i_i", "v_dc", "v_dc_i"]
        return states, ["v", "P_dc"], ["i", "v_dc"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, _, v, *_ = self.power_flow
        v_dc = _para_value(self.params, "Vdc")
        r = _para_value(self.params, "R")
        current = p / v
        e = v - current * r
        base = [current, e]
        if self.apparatus_type == 1011:
            base.extend([v_dc, current])
        return np.array(base, dtype=float), np.array([v, e * current], dtype=float), 0.0

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        p, _, v_pf, *_ = self.power_flow
        x_vdc = _para_value(self.params, "Vdc")
        x_cdc = _para_value(self.params, "Cdc")
        x_wl = _para_value(self.params, "wL")
        x_r = _para_value(self.params, "R")
        x_fi = _para_value(self.params, "fi")
        x_fvdc = _para_value(self.params, "fvdc")
        w0 = _para_value(self.params, "w0")
        w_vdc = x_fvdc * 2 * np.pi
        w_i = x_fi * 2 * np.pi
        inductance = x_wl / w0
        kp_v_dc = x_vdc * x_cdc * w_vdc
        ki_v_dc = kp_v_dc * w_vdc / 4
        kp_i = inductance * w_i
        ki_i = inductance * w_i ** 2 / 4
        current, i_i = x[0], x[1]
        if self.apparatus_type == 1011:
            v_dc, v_dc_i = x[2], x[3]
        else:
            v_dc, v_dc_i = x_vdc, 0.0
        v, p_dc = u
        if flag == 1:
            i_ref = (x_vdc - v_dc) * kp_v_dc + v_dc_i if self.apparatus_type == 1011 else p / v_pf
            e = -(i_ref - current) * kp_i + i_i
            di_i = -(i_ref - current) * ki_i
            di = (v - x_r * current - e) / inductance
            if self.apparatus_type == 1011:
                dv_dc = (e * current - p_dc) / v_dc / x_cdc
                dv_dc_i = (x_vdc - v_dc) * ki_v_dc
                return np.array([di, di_i, dv_dc, dv_dc_i], dtype=float)
            return np.array([di, di_i], dtype=float)
        return np.array([current, v_dc], dtype=float)


class GridFollowingVSI(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        states = ["i_d", "i_q", "i_d_i", "i_q_i", "w_pll_i", "w", "theta"]
        if self.apparatus_type in {10, 12}:
            states += ["v_dc", "v_dc_i"]
        return states, ["v_d", "v_q", "ang_r", "P_dc"], ["i_d", "i_q", "w", "v_dc", "theta"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, q, v, xi, w = self.power_flow
        r = _para_value(self.params, "R")
        wlf = _para_value(self.params, "wLf")
        vdc = _para_value(self.params, "V_dc")
        i_d = p / v
        i_q = -q / v
        e_d = v - r * i_d + wlf * i_q
        e_q = -r * i_q - wlf * i_d
        p_dc = e_d * i_d + e_q * i_q
        self.i_q_r = i_q
        base = [i_d, i_q, e_d, e_q, w, w, xi]
        if self.apparatus_type in {10, 12}:
            base += [vdc, i_d]
        return np.array(base, dtype=float), np.array([v, 0.0, 0.0, p_dc], dtype=float), float(xi)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        cdc = _para_value(self.params, "C_dc", 1.0)
        vdc_ref = _para_value(self.params, "V_dc", 1.0)
        w_vdc = _para_value(self.params, "f_v_dc") * 2 * np.pi
        w_pll = _para_value(self.params, "f_pll") * 2 * np.pi
        w_tau = _para_value(self.params, "f_tau_pll") * 2 * np.pi
        w_i = _para_value(self.params, "f_i_dq") * 2 * np.pi
        wl = _para_value(self.params, "wLf")
        r = _para_value(self.params, "R")
        w0 = _para_value(self.params, "w0")
        l_filter = wl / w0 if w0 else wl
        kp_i, ki_i = l_filter * w_i, l_filter * w_i ** 2 / 4
        kp_pll, ki_pll = w_pll, w_pll ** 2 / 4
        kp_vdc, ki_vdc = cdc * vdc_ref * w_vdc, cdc * vdc_ref * w_vdc ** 2 / 4
        i_d, i_q, i_d_i, i_q_i, w_pll_i, w, theta = x[:7]
        v_dc = x[7] if self.apparatus_type in {10, 12} else vdc_ref
        v_dc_i = x[8] if self.apparatus_type in {10, 12} else 0.0
        v_d, v_q, ang_r, p_dc = u
        e_ang = v_q - ang_r
        w_pll_out = kp_pll * e_ang + w_pll_i
        if self.apparatus_type == 11:
            i_d_r = self.power_flow[0] / self.power_flow[2]
        else:
            i_d_r = (vdc_ref - v_dc) * kp_vdc + v_dc_i
        i_q_r = getattr(self, "i_q_r", 0.0)
        e_d = -(i_d_r - i_d) * kp_i + i_d_i
        e_q = -(i_q_r - i_q) * kp_i + i_q_i
        if flag == 1:
            di_d = (v_d - r * i_d + wl * i_q - e_d) / l_filter
            di_q = (v_q - r * i_q - wl * i_d - e_q) / l_filter
            di_d_i = -(i_d_r - i_d) * ki_i
            di_q_i = -(i_q_r - i_q) * ki_i
            dw_pll_i = e_ang * ki_pll
            dw = (w_pll_out - w) * w_tau
            dtheta = w
            base = [di_d, di_q, di_d_i, di_q_i, dw_pll_i, dw, dtheta]
            if self.apparatus_type in {10, 12}:
                dv_dc = (e_d * i_d + e_q * i_q - p_dc) / max(abs(v_dc), 1e-9) / cdc
                dv_dc_i = (vdc_ref - v_dc) * ki_vdc
                base += [dv_dc, dv_dc_i]
            return np.array(base, dtype=float)
        return np.array([i_d, i_q, w, v_dc, theta], dtype=float)


class GridFormingVSI(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        states = [
            "i_ld", "i_lq", "i_ld_i", "i_lq_i", "v_od", "v_oq", "v_od_i", "v_oq_i",
            "i_od", "i_oq", "v_d_ref", "w", "theta",
        ]
        return states, ["v_d", "v_q", "P0"], ["i_d", "i_q", "w", "theta"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, q, v, xi, w = self.power_flow
        i_od, i_oq = p / v, -q / v
        self.P0, self.Q0 = -p, -q
        self.v_od_r, self.v_oq_r = v, 0.0
        x = [i_od, i_oq, v, 0.0, v, 0.0, i_od, i_oq, i_od, i_oq, v, w, xi]
        return np.array(x, dtype=float), np.array([v, 0.0, 0.0], dtype=float), float(xi)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        wlf = _para_value(self.params, "wLf")
        rf = _para_value(self.params, "Rf")
        wcf = _para_value(self.params, "wCf")
        wlc = _para_value(self.params, "wLc")
        rc = _para_value(self.params, "Rc")
        dw_gain = _para_value(self.params, "Dw")
        fdroop = _para_value(self.params, "fdroop")
        fvdq = _para_value(self.params, "fvdq")
        fidq = _para_value(self.params, "fidq")
        w0 = _para_value(self.params, "w0")
        lf, cf, lc = wlf / w0, wcf / w0, wlc / w0
        w_droop, w_v, w_i = fdroop * 2 * np.pi, fvdq * 2 * np.pi, fidq * 2 * np.pi
        kp_v, ki_v = cf * w_v * 50, cf * w_v ** 2 / 4 * 50
        kp_i, ki_i = lf * w_i, lf * w_i ** 2 / 4
        i_ld, i_lq, i_ld_i, i_lq_i, v_od, v_oq, v_od_i, v_oq_i, i_od, i_oq, v_od_r_state, w, theta = x
        v_gd, v_gq, p0_input = u
        p = -(v_od * i_od + v_oq * i_oq)
        v_od_r = getattr(self, "v_od_r", v_od_r_state)
        v_oq_r = getattr(self, "v_oq_r", 0.0)
        i_ld_r = (v_od_r - v_od) * kp_v + v_od_i + i_od
        i_lq_r = (v_oq_r - v_oq) * kp_v + v_oq_i + i_oq
        e_d = -(i_ld_r - i_ld) * kp_i + i_ld_i
        e_q = -(i_lq_r - i_lq) * kp_i + i_lq_i
        if flag == 1:
            di_ld = (e_d - rf * i_ld + wlf * i_lq - v_od) / lf
            di_lq = (e_q - rf * i_lq - wlf * i_ld - v_oq) / lf
            dv_od = (i_ld - i_od + wcf * v_oq) / cf
            dv_oq = (i_lq - i_oq - wcf * v_od) / cf
            di_od = (v_od - rc * i_od + wlc * i_oq - v_gd) / lc
            di_oq = (v_oq - rc * i_oq - wlc * i_od - v_gq) / lc
            dw = (w0 + dw_gain * (p0_input + getattr(self, "P0", 0.0) - p) - w) * w_droop
            return np.array([
                di_ld, di_lq, -(i_ld_r - i_ld) * ki_i, -(i_lq_r - i_lq) * ki_i,
                dv_od, dv_oq, (v_od_r - v_od) * ki_v, (v_oq_r - v_oq) * ki_v,
                di_od, di_oq, 0.0, dw, w,
            ], dtype=float)
        return np.array([i_od, i_oq, w, theta], dtype=float)


class Battery(GridFormingVSI):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        base_states, inputs, _ = super().signal_list()
        return base_states + ["i_bat", "v_dc", "i_bat_ref", "duty_cycle"], inputs, ["i_d", "i_q", "w", "theta", "v_dc", "i_bat", "p"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        x, u, xi = super().equilibrium()
        p = self.power_flow[0]
        v_dc = _para_value(self.params, "v_dc_ref", 1.0)
        v_ocv = _para_value(self.params, "v_ocv", v_dc)
        r_bat = _para_value(self.params, "R_bat", 1.0)
        disc = max(v_ocv ** 2 - 4 * r_bat * max(p, 0.0), 0.0)
        i_bat = (v_ocv - np.sqrt(disc)) / (2 * r_bat) if r_bat else p / max(v_dc, 1e-9)
        duty = np.clip(1 - (p / max(v_dc, 1e-9)) / max(abs(i_bat), 1e-9), 0.0, 1.0)
        return np.concatenate([x, [i_bat, v_dc, i_bat, duty]]), u, xi

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        gfm_dx_or_y = super().state_space_equation(x[:13], u, flag)
        if flag == 2:
            p = -(x[4] * x[8] + x[5] * x[9])
            return np.concatenate([gfm_dx_or_y, [x[14], x[13], p]])
        i_bat, v_dc, i_bat_ref, duty = x[13], x[14], x[15], x[16]
        l_dc = _para_value(self.params, "L_dc", 1.0)
        c_dc = _para_value(self.params, "C_dc", 1.0)
        v_ocv = _para_value(self.params, "v_ocv", v_dc)
        r_bat = _para_value(self.params, "R_bat", 0.0)
        v_ref = _para_value(self.params, "v_dc_ref", v_dc)
        w_v = _para_value(self.params, "fvdc") * 2 * np.pi
        w_i = _para_value(self.params, "fibat") * 2 * np.pi
        p = -(x[4] * x[8] + x[5] * x[9])
        i_o = -p / max(abs(v_dc), 1e-9)
        v_bat = v_ocv - r_bat * i_bat
        di_bat = (v_bat - (1 - duty) * v_dc) / l_dc
        dv_dc = ((1 - duty) * i_bat - i_o) / c_dc
        di_bat_ref = (v_ref - v_dc) * w_v
        dduty = (i_bat_ref - i_bat) * w_i
        return np.concatenate([gfm_dx_or_y, [di_bat, dv_dc, di_bat_ref, dduty]])


class PhotovoltaicGFM(GridFormingVSI):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        states = ["v_i", "i_i", "i_l", "v_pv"] + super().signal_list()[0][2:]
        return states, ["v_d", "v_q", "P0"], ["i_d", "i_q", "w", "theta"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        x, _u, xi = super().equilibrium()
        v_pv = _para_value(self.params, "v_pv_ref", 1.0)
        p = self.power_flow[0]
        front = np.array([0.0, 0.0, -p / max(v_pv, 1e-9), v_pv])
        return np.concatenate([front, x[2:]]), np.array([self.power_flow[2], 0.0, p]), xi

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        gfm_x = np.concatenate([x[2:4], x[4:]])
        if flag == 2:
            return super().state_space_equation(gfm_x, u, 2)
        v_i, i_i, i_l, v_pv = x[:4]
        v_ref = _para_value(self.params, "v_pv_ref", v_pv)
        w_v = _para_value(self.params, "fvdc") * 2 * np.pi
        w_i = _para_value(self.params, "fidc") * 2 * np.pi
        p_ref = u[2]
        i_ref = (v_ref - v_pv) * w_v + i_i
        dv_i = (v_ref - v_pv) * w_v
        di_i = (i_ref - i_l) * w_i
        di_l = di_i
        dv_pv = (i_l * max(v_i, 1.0) - p_ref) / max(abs(v_pv), 1e-9)
        return np.concatenate([[dv_i, di_i, di_l, dv_pv], super().state_space_equation(gfm_x, u, 1)[2:]])


class PhotovoltaicGFL(GridFollowingVSI):
    pass


class WindTurbineGFM(Battery):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        states, inputs, outputs = super().signal_list()
        return states + ["i_sd", "i_sq", "i_sd_i", "i_sq_i", "w_m", "w_m_i", "theta_m"], inputs, outputs + ["T_e", "T_m", "n", "theta_e", "beta"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        x, u, xi = super().equilibrium()
        return np.concatenate([x, [0.0, 0.0, 0.0, 0.0, _para_value(self.params, "n_r", 1.0), 0.0, 0.0]]), u, xi

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        base = super().state_space_equation(x[:17], u, flag)
        if flag == 2:
            w_m = x[21]
            return np.concatenate([base, [0.0, 0.0, w_m, x[23], 0.0]])
        w_i = _para_value(self.params, "f_i_sdq", 1.0) * 2 * np.pi
        w_mf = _para_value(self.params, "f_w_m", 1.0) * 2 * np.pi
        return np.concatenate([base, [-w_i * x[17], -w_i * x[18], -w_i * x[19], -w_i * x[20], -w_mf * (x[21] - _para_value(self.params, "n_r", 1.0)), 0.0, x[21]]])


class WindTurbineGFL(GridFollowingVSI):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        states, inputs, outputs = super().signal_list()
        return states + ["i_sd", "i_sq", "i_sd_i", "i_sq_i", "w_m", "w_m_i", "theta_m"], inputs[:3], outputs

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        x, u, xi = super().equilibrium()
        return np.concatenate([x, [0.0, 0.0, 0.0, 0.0, _para_value(self.params, "n_r", 1.0), 0.0, 0.0]]), u[:3], xi

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        u4 = np.array([u[0], u[1], u[2], 0.0])
        base = super().state_space_equation(x[:7], u4, flag)
        if flag == 2:
            return base
        w_i = _para_value(self.params, "f_i_sdq", 1.0) * 2 * np.pi
        w_mf = _para_value(self.params, "f_w_m", 1.0) * 2 * np.pi
        return np.concatenate([base, [-w_i * x[7], -w_i * x[8], -w_i * x[9], -w_i * x[10], -w_mf * (x[11] - _para_value(self.params, "n_r", 1.0)), 0.0, x[11]]])


class InterlinkAcDc(ApparatusModel):
    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        if self.apparatus_type == 2002:
            states = ["i_d", "i_q", "v_dc", "i", "theta"]
        else:
            states = ["i_d", "i_q", "i_d_i", "i_q_i", "w_pll_i", "w", "theta", "v_dc", "v_dc_i", "i"]
        return states, ["v_d", "v_q", "v", "ang_r"], ["i_d", "i_q", "i", "w", "v_dc", "theta"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p_ac, q_ac, v_ac, xi, w = self.power_flow[:5]
        p_dc = self.power_flow[5] if len(self.power_flow) > 5 else -p_ac
        v_dc_bus = self.power_flow[7] if len(self.power_flow) > 7 else 1.0
        r_ac = _para_value(self.params, "R_ac")
        r_dc = _para_value(self.params, "R_dc")
        i_d, i_q = p_ac / v_ac, -q_ac / v_ac
        current_dc = p_dc / max(abs(v_dc_bus), 1e-9)
        v_dc = v_dc_bus - current_dc * r_dc
        if self.apparatus_type == 2002:
            x = [i_d, i_q, v_dc, current_dc, xi]
        else:
            x = [i_d, i_q, v_ac - r_ac * i_d, -r_ac * i_q, w, w, xi, v_dc, current_dc, current_dc]
        return np.array(x, dtype=float), np.array([v_ac, 0.0, v_dc_bus, 0.0], dtype=float), float(xi)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        cdc = _para_value(self.params, "C_dc", 1.0)
        wl_ac = _para_value(self.params, "wL_ac", 1.0)
        r_ac = _para_value(self.params, "R_ac", 0.0)
        wl_dc = _para_value(self.params, "wL_dc", 1.0)
        r_dc = _para_value(self.params, "R_dc", 0.0)
        w_i = _para_value(self.params, "fidq", 1.0) * 2 * np.pi
        w_v = _para_value(self.params, "fvdc", 1.0) * 2 * np.pi
        w_pll = _para_value(self.params, "fpll", 1.0) * 2 * np.pi
        w0 = _para_value(self.params, "w0", 1.0)
        l_ac, l_dc = wl_ac / w0, wl_dc / w0
        v_d, v_q, v_bus_dc, ang_r = u
        if self.apparatus_type == 2002:
            i_d, i_q, v_dc, i_dc, theta = x
            p = v_d * i_d + v_q * i_q
            w = w0 + _para_value(self.params, "N", 0.0) * (_para_value(self.params, "v_dc_ref", v_dc) - v_dc)
            if flag == 1:
                return np.array([
                    (1 - r_ac * i_d - v_d) / l_ac,
                    (-r_ac * i_q - v_q) / l_ac,
                    (v_bus_dc * i_dc - p) / max(abs(v_dc), 1e-9) / cdc,
                    (v_dc - r_dc * i_dc - v_bus_dc) / l_dc,
                    w,
                ])
            return np.array([i_d, i_q, i_dc, w, v_dc, theta])
        i_d, i_q, i_d_i, i_q_i, w_pll_i, w, theta, v_dc, v_dc_i, i_dc = x
        i_d_r = self.power_flow[0] / self.power_flow[2]
        if self.apparatus_type == 2001:
            i_d_r = (_para_value(self.params, "v_dc_ref", v_dc) - v_bus_dc) * w_v + v_dc_i
        i_q_r = -self.power_flow[1] / self.power_flow[2]
        e_d = -(i_d_r - i_d) * l_ac * w_i + i_d_i
        e_q = -(i_q_r - i_q) * l_ac * w_i + i_q_i
        w_pll_out = w_pll_i + w_pll * (v_q - ang_r)
        if flag == 1:
            p_ac = e_d * i_d + e_q * i_q
            return np.array([
                (v_d - r_ac * i_d + wl_ac * i_q - e_d) / l_ac,
                (v_q - r_ac * i_q - wl_ac * i_d - e_q) / l_ac,
                -(i_d_r - i_d) * l_ac * w_i ** 2 / 4,
                -(i_q_r - i_q) * l_ac * w_i ** 2 / 4,
                (v_q - ang_r) * w_pll ** 2 / 4,
                (w_pll_out - w) * w_pll,
                w,
                (v_bus_dc * i_dc - p_ac) / max(abs(v_dc), 1e-9) / cdc,
                (_para_value(self.params, "v_dc_ref", v_dc) - v_bus_dc) * w_v if self.apparatus_type == 2001 else 0.0,
                (v_dc - r_dc * i_dc - v_bus_dc) / l_dc,
            ])
        return np.array([i_d, i_q, i_dc, w, v_dc, theta])


class PlaceholderApparatus(ApparatusModel):
    """Unsupported apparatus sentinel; factory should not route example types here."""

    def signal_list(self) -> tuple[list[str], list[str], list[str]]:
        if self.apparatus_type < 1000:
            return [], ["v_d", "v_q"], ["i_d", "i_q", "w"]
        if self.apparatus_type < 2000:
            return [], ["v"], ["i"]
        return [], ["v_d", "v_q", "v"], ["i_d", "i_q", "i"]

    def equilibrium(self) -> tuple[np.ndarray, np.ndarray, float]:
        p, q, v, xi, w = self.power_flow[:5]
        if self.apparatus_type < 1000:
            return np.array([]), np.array([v, 0.0]), float(xi)
        if self.apparatus_type < 2000:
            return np.array([]), np.array([v]), float(xi)
        return np.array([]), np.array([v, 0.0, v]), float(xi)

    def state_space_equation(self, x: np.ndarray, u: np.ndarray, flag: int) -> np.ndarray:
        if flag == 1:
            return np.array([])
        if self.apparatus_type < 1000:
            return np.array([0.0, 0.0, self.power_flow[4]])
        if self.apparatus_type < 2000:
            return np.array([0.0])
        return np.array([0.0, 0.0, 0.0])


def create_apparatus_model(
    buses: tuple[int, ...],
    apparatus_type: int,
    power_flow: np.ndarray,
    params: dict,
    ts: float,
) -> DescriptorStateSpace:
    cls: type[ApparatusModel]
    switch_length = 0
    family = apparatus_type // 10
    if family == 0:
        cls = SynchronousMachine
    elif family == 1:
        cls = GridFollowingVSI
    elif family == 2:
        cls = GridFormingVSI
    elif family == 3:
        cls = Battery
    elif family == 4 and apparatus_type == 40:
        cls = PhotovoltaicGFM
    elif family == 4 and apparatus_type == 41:
        cls = PhotovoltaicGFL
    elif family == 5 and apparatus_type == 50:
        cls = WindTurbineGFM
    elif family == 5 and apparatus_type == 51:
        cls = WindTurbineGFL
    elif family == 9:
        cls = InfiniteBusAc
        switch_length = 2
    elif family == 10:
        cls = FloatingBusAc
    elif family == 101:
        cls = GridFeedingBuck
    elif family == 109:
        cls = InfiniteBusDc
        switch_length = 1
    elif family == 110:
        cls = FloatingBusDc
    elif family == 200:
        cls = InterlinkAcDc
    else:
        cls = PlaceholderApparatus
    model = cls(apparatus_type, params, power_flow, buses, ts)
    dss = model.to_dss()
    if len(buses) == 2:
        dss.inputs[:3] = [f"v_d{buses[0]}", f"v_q{buses[0]}", f"v{buses[1]}"]
        dss.outputs[:3] = [f"i_d{buses[0]}", f"i_q{buses[0]}", f"i{buses[1]}"]
    if switch_length:
        dss = switch_inputs_outputs(dss, switch_length)
    return dss


def link_apparatus(models: list[DescriptorStateSpace]) -> DescriptorStateSpace:
    if not models:
        return DescriptorStateSpace.static(np.zeros((0, 0)))
    current = models[0]
    for model in models[1:]:
        current = append(current, model)
    return current
