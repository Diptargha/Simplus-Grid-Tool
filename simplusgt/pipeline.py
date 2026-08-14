"""End-to-end non-Simulink SimplusGT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dss import DescriptorStateSpace, feedback
from .io import load_case
from .models import create_apparatus_model, link_apparatus
from .netlists import NormalizedNetlists, normalize_case
from .network import NetworkDSS, network_dss
from .powerflow import PowerFlowResult, load_to_self_branch, run_power_flow
from .schema import CaseData


@dataclass
class RunResult:
    case: CaseData
    netlists: NormalizedNetlists
    power_flow: PowerFlowResult
    buses_after_load: np.ndarray
    lines_after_load: np.ndarray
    power_flow_after_load: list[np.ndarray]
    apparatus_models: list[DescriptorStateSpace]
    apparatus_block: DescriptorStateSpace
    network: NetworkDSS
    whole_system_dss: DescriptorStateSpace
    whole_system_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    eigenvalues: np.ndarray
    port_v: list[int] = field(default_factory=list)
    port_i: list[int] = field(default_factory=list)
    bus_port_v: list[list[int]] = field(default_factory=list)
    bus_port_i: list[list[int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _apparatus_power_flow(power_flow: list[np.ndarray], buses: tuple[int, ...]) -> np.ndarray:
    if len(buses) == 1:
        return power_flow[buses[0] - 1]
    return np.concatenate([power_flow[bus - 1] for bus in buses])


def _connect_gm_zbus(gm: DescriptorStateSpace, zbus: DescriptorStateSpace, num_bus: int) -> tuple[DescriptorStateSpace, list[int], list[int], list[list[int]], list[list[int]]]:
    port_v_feedin: list[int] = []
    port_i_feedout: list[int] = []
    bus_port_v: list[list[int]] = []
    bus_port_i: list[list[int]] = []
    for bus in range(1, num_bus + 1):
        if f"v_d{bus}" in gm.inputs:
            in1 = gm.inputs.index(f"v_d{bus}")
            out1 = gm.outputs.index(f"i_d{bus}")
            ports_v = [in1, in1 + 1]
            ports_i = [out1, out1 + 1]
        elif f"v{bus}" in gm.inputs:
            in1 = gm.inputs.index(f"v{bus}")
            out1 = gm.outputs.index(f"i{bus}")
            ports_v = [in1]
            ports_i = [out1]
        else:
            raise ValueError(f"Unable to find voltage/current ports for bus {bus}")
        port_v_feedin.extend(ports_v)
        port_i_feedout.extend(ports_i)
        bus_port_v.append(ports_v)
        bus_port_i.append(ports_i)
    gsys = feedback(gm, zbus, port_v_feedin, port_i_feedout)
    return gsys, port_v_feedin, port_i_feedout, bus_port_v, bus_port_i


def run_case(path: str | Path, input_format: str | None = None) -> RunResult:
    case = load_case(path, input_format)
    return run_case_data(case)


def run_case_data(case: CaseData) -> RunResult:
    warnings: list[str] = []
    netlists = normalize_case(case)
    fs = case.Basic.Fs
    ts = 1 / fs
    wbase = case.Basic.Fbase * 2 * np.pi
    algorithm = case.Advance.PowerFlowAlgorithm
    if np.any(netlists.buses[:, 11] == 2) and algorithm != 1:
        warnings.append("DC area present; Gauss-Seidel power flow was forced")
    pf = run_power_flow(netlists.buses, netlists.lines, wbase, algorithm)
    buses_new, lines_new, pf_new = load_to_self_branch(netlists.buses, netlists.lines, pf.power_flow)
    app_models: list[DescriptorStateSpace] = []
    for buses, app_type, params in zip(netlists.apparatus_buses, netlists.apparatus_types, netlists.apparatus_params):
        app_pf = _apparatus_power_flow(pf_new, buses)
        model = create_apparatus_model(buses, app_type, app_pf, params, ts)
        app_models.append(model)
        for message in getattr(model, "model_warnings", []) or []:
            warnings.append(message)
    gm = link_apparatus(app_models)
    net = network_dss(buses_new, lines_new, wbase)
    gsys, port_v, port_i, bus_port_v, bus_port_i = _connect_gm_zbus(gm, net.zbus, int(np.max(buses_new[:, 0])))
    try:
        whole_ss = gsys.to_state_space()
        eigenvalues = np.linalg.eigvals(whole_ss[0]) if whole_ss[0].size else np.array([], dtype=complex)
        eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.append(f"Whole-system DSS-to-SS conversion failed ({exc}); using generalized eigenvalues")
        whole_ss = None
        eigenvalues = gsys.eigenvalues()
        eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    return RunResult(
        case=case,
        netlists=netlists,
        power_flow=pf,
        buses_after_load=buses_new,
        lines_after_load=lines_new,
        power_flow_after_load=pf_new,
        apparatus_models=app_models,
        apparatus_block=gm,
        network=net,
        whole_system_dss=gsys,
        whole_system_ss=whole_ss,
        eigenvalues=eigenvalues,
        port_v=port_v,
        port_i=port_i,
        bus_port_v=bus_port_v,
        bus_port_i=bus_port_i,
        warnings=warnings,
    )
