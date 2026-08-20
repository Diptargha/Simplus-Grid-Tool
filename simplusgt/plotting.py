"""MATLAB-style fundamental and greybox plots for the Python SimplusGT pipeline.

Mirrors the figures produced by ``SimplusGT.Toolbox.Main`` and the greybox
Layer 1/2 charts from ``+Modal``:
- pole map (global + zoomed)
- bus admittance Bode (dq Ydd and complex-vector Ydq+)
- grid-strength layout heatmap
- apparatus Layer 1 pie / Layer 2 bars
- sensitivity Layer 1 pie / Layer 2 bars
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import bus_strength, stability_report
from .dss import dss2ss
from .greybox import GreyboxResult
from .pipeline import RunResult


COMPLEX_T = np.array([[1.0, 1j], [1.0, -1j]], dtype=complex)


def _active_apparatus_bus(apparatus_type: int) -> bool:
    return (
        0 <= apparatus_type < 90
        or 1000 <= apparatus_type < 1090
        or 2000 <= apparatus_type < 2090
    )


def _bus_type_vif(apparatus_types: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """MATLAB BusTypeVIF indices (1-based apparatus indices = bus numbers for 1-bus units)."""

    vbus: list[int] = []
    ibus: list[int] = []
    fbus: list[int] = []
    for idx, app_type in enumerate(apparatus_types, start=1):
        if (0 <= app_type <= 9) or app_type == 90 or (20 <= app_type <= 40) or app_type == 50:
            vbus.append(idx)
        elif (10 <= app_type <= 19) or app_type in {41, 51}:
            ibus.append(idx)
        elif app_type == 100:
            fbus.append(idx)
    return vbus, ibus, fbus


def plot_pole_map(eigenvalues: np.ndarray, *, fig=None, show: bool = False):
    """Plot global and zoomed pole maps in Hz (MATLAB PlotPoleMap / GScatterMode)."""

    import matplotlib.pyplot as plt

    report = stability_report(eigenvalues)
    eig_hz = report.eigenvalues_hz
    fig = fig or plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax, title, zoom in (
        (ax1, "Global pole map", False),
        (ax2, "Zoomed pole map", True),
    ):
        ax.scatter(np.real(eig_hz), np.imag(eig_hz), marker="x", linewidths=1.5)
        ax.set_xlabel("Real Part (Hz)")
        ax.set_ylabel("Imaginary Part (Hz)")
        ax.set_title(title)
        ax.grid(True)
        if zoom:
            ax.axis([-80, 20, -150, 150])
            # 10% damping lines (MATLAB GScatterMode).
            ax.plot([-80.0, 0.0], [800.0, 0.0], "--", color="tab:blue", linewidth=1.5, label="10% damping")
            ax.plot([-80.0, 0.0], [-800.0, 0.0], "--", color="tab:blue", linewidth=1.5)
            ax.legend(loc="best")
    fig.suptitle("Pole map")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _ss_frequency_response(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    values = np.zeros((omega.size, c.shape[0], b.shape[1]), dtype=complex)
    eye = np.eye(a.shape[0], dtype=complex) if a.size else None
    for idx, w in enumerate(omega):
        if a.size == 0:
            values[idx] = d
        else:
            values[idx] = c @ np.linalg.solve(1j * w * eye - a, b) + d
    return values


def _admittance_plot_freq_count(nx: int, requested: int = 500) -> int:
    """Fewer Bode samples for large state-space models (dense FRF is O(n_freq*nx^3))."""

    if nx >= 200:
        return min(requested, 80)
    if nx >= 80:
        return min(requested, 160)
    return requested


def _bode_mag_phase(ax_mag, ax_phase, freq_hz: np.ndarray, values: np.ndarray, label: str) -> None:
    mag = np.abs(values)
    phase = np.unwrap(np.angle(values)) * 180 / np.pi
    ax_mag.loglog(freq_hz, np.maximum(mag, 1e-30), label=label)
    ax_phase.semilogx(freq_hz, phase, label=label)
    ax_mag.grid(True, which="both")
    ax_phase.grid(True, which="both")
    ax_mag.set_ylabel("|Y|")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")


def plot_admittance_dq_axes(
    result: RunResult,
    *,
    axes: Sequence[str] = ("dd", "dq", "qd", "qq"),
    apparatus_indices: Sequence[int] | None = None,
    n_freq: int = 500,
    f_min: float = 0.1,
    f_max: float = 1e4,
    show: bool = False,
) -> dict[str, Any]:
    """Per-axis bus admittance Bode plots (MATLAB ``Modal.BodeDraw``).

    ``axes`` entries are ``dd`` / ``dq`` / ``qd`` / ``qq``.
    ``apparatus_indices`` are zero-based; default is all active apparatuses.
    """

    import matplotlib.pyplot as plt

    axis_map = {"dd": (0, 0), "dq": (0, 1), "qd": (1, 0), "qq": (1, 1)}
    selected_axes = [name.lower() for name in axes]
    unknown = [name for name in selected_axes if name not in axis_map]
    if unknown:
        raise ValueError(f"Unknown admittance axes {unknown}; expected dd/dq/qd/qq")

    if result.whole_system_ss is None:
        a, b, c, d = dss2ss(result.whole_system_dss)
    else:
        a, b, c, d = result.whole_system_ss
    n_freq = _admittance_plot_freq_count(int(a.shape[0]), n_freq)
    omega = np.logspace(np.log10(f_min), np.log10(f_max), n_freq) * 2 * np.pi
    freq = omega / (2 * np.pi)
    print(f"  [plot] Admittance dq-axes Bode ({n_freq} freq pts)...", flush=True)

    if apparatus_indices is None:
        apparatus_indices = [
            idx
            for idx, app_type in enumerate(result.netlists.apparatus_types)
            if _active_apparatus_bus(int(app_type))
        ]

    figures: dict[str, Any] = {}
    for axis_name in selected_axes:
        out_bias, in_bias = axis_map[axis_name]
        fig = plt.figure()
        ax_mag = fig.add_subplot(2, 1, 1)
        ax_phase = fig.add_subplot(2, 1, 2)
        legends: list[str] = []
        for app_idx in apparatus_indices:
            buses = result.netlists.apparatus_buses[app_idx]
            if not buses:
                continue
            bus = int(buses[0])
            if bus - 1 >= len(result.bus_port_i) or bus - 1 >= len(result.bus_port_v):
                continue
            out_idx = result.bus_port_i[bus - 1]
            in_idx = result.bus_port_v[bus - 1]
            if len(out_idx) <= out_bias or len(in_idx) <= in_bias:
                continue
            b_ch = b[:, [in_idx[in_bias]]]
            c_ch = c[[out_idx[out_bias]], :]
            d_ch = d[np.ix_([out_idx[out_bias]], [in_idx[in_bias]])]
            values = _ss_frequency_response(a, b_ch, c_ch, d_ch, omega)[:, 0, 0]
            label = f"Node{bus}"
            legends.append(label)
            _bode_mag_phase(ax_mag, ax_phase, freq, values, label)
        ax_mag.set_title(f"Admittance Bode Diagram: {axis_name[0]}-{axis_name[1]} axis")
        if legends:
            ax_mag.legend(loc="best")
        fig.tight_layout()
        figures[axis_name] = fig
    if show:
        plt.show()
    return figures


def plot_admittance_spectrum(
    result: RunResult,
    *,
    n_freq: int = 500,
    f_min: float = 0.1,
    f_max: float = 1e4,
    fig_dq=None,
    fig_cplx=None,
    show: bool = False,
):
    """Plot per-bus Ydd (dq) and Ydq+ (complex-vector) Bode curves.

    Matches ``PlotAdmittanceSpectrum`` in MATLAB ``Main.m``.
    """

    import matplotlib.pyplot as plt

    if result.whole_system_ss is None:
        a, b, c, d = dss2ss(result.whole_system_dss)
    else:
        a, b, c, d = result.whole_system_ss
    n_freq = _admittance_plot_freq_count(int(a.shape[0]), n_freq)

    omega_p = np.logspace(np.log10(f_min), np.log10(f_max), n_freq) * 2 * np.pi
    omega_pn = np.concatenate([-np.flip(omega_p), omega_p])
    freq_p = omega_p / (2 * np.pi)
    freq_n = omega_pn[: omega_p.size] / (2 * np.pi)
    freq_pos = omega_pn[omega_p.size :] / (2 * np.pi)

    fig_dq = fig_dq or plt.figure()
    fig_cplx = fig_cplx or plt.figure()
    ax_dq_mag = fig_dq.add_subplot(2, 1, 1)
    ax_dq_phase = fig_dq.add_subplot(2, 1, 2)
    ax_cn_mag = fig_cplx.add_subplot(2, 2, 1)
    ax_cn_phase = fig_cplx.add_subplot(2, 2, 3)
    ax_cp_mag = fig_cplx.add_subplot(2, 2, 2)
    ax_cp_phase = fig_cplx.add_subplot(2, 2, 4)

    legends: list[str] = []
    num_bus = int(np.max(result.buses_after_load[:, 0]))
    t_inv = np.linalg.inv(COMPLEX_T)
    for bus in range(1, num_bus + 1):
        app_idx = next(
            (idx for idx, buses in enumerate(result.netlists.apparatus_buses) if bus in buses),
            None,
        )
        if app_idx is None:
            continue
        app_type = int(result.netlists.apparatus_types[app_idx])
        if not _active_apparatus_bus(app_type):
            continue
        if bus - 1 >= len(result.bus_port_i) or bus - 1 >= len(result.bus_port_v):
            continue
        out_idx = result.bus_port_i[bus - 1]
        in_idx = result.bus_port_v[bus - 1]
        if len(out_idx) < 1 or len(in_idx) < 1:
            continue
        print(f"  [plot] Admittance spectrum: Bus{bus} ({n_freq} freq pts)...", flush=True)
        b_bus = b[:, in_idx]
        c_bus = c[out_idx, :]
        d_bus = d[np.ix_(out_idx, in_idx)]
        # One sweep over +/- omega; Ydd uses positive freqs; complex-vector via T Y T^{-1}.
        y_dq = _ss_frequency_response(a, b_bus, c_bus, d_bus, omega_pn)
        half = omega_p.size
        y_dd = y_dq[half:, 0, 0]
        label = f"Bus{bus}"
        legends.append(label)
        _bode_mag_phase(ax_dq_mag, ax_dq_phase, freq_p, y_dd, label)

        if d_bus.shape[0] >= 2 and d_bus.shape[1] >= 2:
            y_cplx = np.empty_like(y_dq)
            for k in range(y_dq.shape[0]):
                y_cplx[k] = COMPLEX_T @ y_dq[k] @ t_inv
            y11 = y_cplx[:, 0, 0]
            _bode_mag_phase(ax_cn_mag, ax_cn_phase, np.abs(freq_n), y11[:half], label)
            _bode_mag_phase(ax_cp_mag, ax_cp_phase, freq_pos, y11[half:], label)

    ax_dq_mag.set_title(r"Transfer Function Matrix dq frame: $Y_{dd}$")
    ax_cn_mag.set_title(r"Complex Vector dq frame: $Y_{dq+}$ (neg. f)")
    ax_cp_mag.set_title(r"Complex Vector dq frame: $Y_{dq+}$ (pos. f)")
    if legends:
        ax_dq_mag.legend(loc="best")
        ax_cp_mag.legend(loc="best")
    fig_dq.tight_layout()
    fig_cplx.tight_layout()
    if show:
        plt.show()
    return fig_dq, fig_cplx


def plot_grid_strength(result: RunResult, *, fig=None, show: bool = False):
    """Plot network layout colored by log10 bus admittance strength."""

    import matplotlib.pyplot as plt
    import networkx as nx

    if np.any(result.netlists.buses[:, 11] != 1):
        raise ValueError("Grid-strength plot currently supports AC networks only")

    ydiag, graph_matrix = bus_strength(result.lines_after_load, list(result.netlists.apparatus_types))
    graph = nx.Graph()
    n_bus = len(ydiag)
    graph.add_nodes_from(range(1, n_bus + 1))
    for i in range(n_bus):
        for j in range(i + 1, n_bus):
            if graph_matrix[i, j] > 0:
                graph.add_edge(i + 1, j + 1, weight=float(graph_matrix[i, j]))
    pos = nx.spring_layout(graph, seed=0, weight="weight")

    vbus, ibus, fbus = _bus_type_vif(result.netlists.apparatus_types)
    color_map = []
    for node in graph.nodes:
        if node in vbus:
            color_map.append("green")
        elif node in ibus:
            color_map.append("red")
        elif node in fbus:
            color_map.append("0.7")
        else:
            color_map.append("tab:blue")

    fig = fig or plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = np.array([pos[n][0] for n in graph.nodes])
    ys = np.array([pos[n][1] for n in graph.nodes])
    strength = np.log10(np.maximum(np.abs(ydiag), 1e-30))
    sc = ax.scatter(xs, ys, c=strength, s=120, cmap="viridis", zorder=1)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5, zorder=0)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=80, ax=ax, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}$(Bus Admittance)")
    ax.set_title("Grid strength")
    ax.set_axis_off()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_apparatus_layer12(mode_result: Any, *, fig=None, show: bool = False):
    """Pie/bar charts for apparatus greybox Layer 1 and Layer 2."""

    import matplotlib.pyplot as plt

    layer1 = list(getattr(mode_result, "layer1", []) or [])
    layer2 = list(getattr(mode_result, "layer2", []) or [])
    if not layer1 and not layer2:
        raise ValueError("Mode result has no apparatus Layer 1/2 data")

    fig = fig or plt.figure()
    ax_pie = fig.add_subplot(2, 2, (1, 3))
    ax_real = fig.add_subplot(2, 2, 2)
    ax_imag = fig.add_subplot(2, 2, 4)

    if layer1:
        labels = [item.get("label", f"App {item.get('apparatus_index', '?')}") for item in layer1]
        values = np.array([abs(float(item.get("normalized", item.get("value", 0.0)))) for item in layer1], dtype=float)
        if values.sum() <= 0:
            values = np.ones(len(values)) / max(len(values), 1)
        ax_pie.pie(values, labels=labels, autopct="%1.1f%%")
        ax_pie.set_title(f"Apparatus Layer 1 (mode {getattr(mode_result, 'mode_index', '?')})")

    if layer2:
        labels = [item.get("label", f"App {item.get('apparatus_index', '?')}") for item in layer2]
        real_pu = [float(item.get("real_normalized", 0.0)) for item in layer2]
        imag_pu = [float(item.get("imag_normalized", 0.0)) for item in layer2]
        x = np.arange(len(labels))
        ax_real.bar(x, real_pu)
        ax_imag.bar(x, imag_pu)
        ax_real.set_xticks(x)
        ax_imag.set_xticks(x)
        ax_real.set_xticklabels(labels, rotation=45, ha="right")
        ax_imag.set_xticklabels(labels, rotation=45, ha="right")
        ax_real.set_title("Layer 2 real (pu) — damping sense")
        ax_imag.set_title("Layer 2 imag (pu) — frequency sense")
        ax_real.set_ylabel("Apparatus")
        ax_imag.set_ylabel("Apparatus")
        ax_real.set_xlabel("Normalized Re(L₂): damping")
        ax_imag.set_xlabel("Normalized Im(L₂): frequency")
        ax_real.grid(True, axis="y")
        ax_imag.grid(True, axis="y")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_sensitivity_layer12(sens_result: Any, *, fig=None, show: bool = False):
    """Pie/bar charts for nodal/branch sensitivity Layer 1 and Layer 2."""

    import matplotlib.pyplot as plt

    records = list(getattr(sens_result, "layer12", []) or [])
    if not records:
        raise ValueError("Sensitivity result has no Layer 1/2 data")

    fig = fig or plt.figure()
    ax_pie = fig.add_subplot(2, 2, (1, 3))
    ax_real = fig.add_subplot(2, 2, 2)
    ax_imag = fig.add_subplot(2, 2, 4)

    labels = [item.get("component", "?") for item in records]
    l1 = np.array([abs(float(item.get("layer1_normalized", item.get("layer1", 0.0)))) for item in records], dtype=float)
    if l1.sum() <= 0:
        l1 = np.ones(len(l1)) / max(len(l1), 1)
    ax_pie.pie(l1, labels=labels, autopct="%1.1f%%")
    ax_pie.set_title(f"Sensitivity Layer 1 (mode {getattr(sens_result, 'mode_index', '?')})")

    x = np.arange(len(labels))
    ax_real.bar(x, [float(item.get("layer2_real_normalized", 0.0)) for item in records])
    ax_imag.bar(x, [float(item.get("layer2_imag_normalized", 0.0)) for item in records])
    ax_real.set_xticks(x)
    ax_imag.set_xticks(x)
    ax_real.set_xticklabels(labels, rotation=45, ha="right")
    ax_imag.set_xticklabels(labels, rotation=45, ha="right")
    ax_real.set_title("Sens Layer 2 real (pu) — damping sense")
    ax_imag.set_title("Sens Layer 2 imag (pu) — frequency sense")
    ax_real.set_ylabel("Component")
    ax_imag.set_ylabel("Component")
    ax_real.set_xlabel("Normalized Re(L₂): damping")
    ax_imag.set_xlabel("Normalized Im(L₂): frequency")
    ax_real.grid(True, axis="y")
    ax_imag.grid(True, axis="y")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_whole_system_transfer(
    transfer_values: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    row: int = 0,
    col: int = 0,
    title: str = "Whole-system transfer",
    fig=None,
    show: bool = False,
):
    """Bode plot of one channel from greybox Ysys/Zsys samples."""

    import matplotlib.pyplot as plt

    values = np.asarray(transfer_values, dtype=complex)
    freq = np.asarray(frequencies_hz, dtype=float)
    channel = values[:, row, col]
    fig = fig or plt.figure()
    ax_mag = fig.add_subplot(2, 1, 1)
    ax_phase = fig.add_subplot(2, 1, 2)
    _bode_mag_phase(ax_mag, ax_phase, freq, channel, f"({row},{col})")
    ax_mag.set_title(title)
    ax_mag.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_case_fundamentals(
    result: RunResult,
    *,
    output_dir: str | Path | None = None,
    show: bool = False,
    include_pole: bool = True,
    include_admittance: bool = True,
    include_dq_axes: bool = True,
    include_strength: bool = True,
) -> dict[str, Path | None]:
    """Create the MATLAB Main.m / Modal BodeDraw fundamental plot set for a run result."""

    import matplotlib.pyplot as plt

    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path | None] = {}

    if include_pole:
        fig_poles = plot_pole_map(result.eigenvalues, show=False)
        saved["pole_map"] = _maybe_save(fig_poles, output_path, "pole_map.png", show)

    if include_admittance:
        try:
            fig_dq, fig_cplx = plot_admittance_spectrum(result, show=False)
            saved["admittance_dq"] = _maybe_save(fig_dq, output_path, "admittance_ydd.png", show)
            saved["admittance_cplx"] = _maybe_save(fig_cplx, output_path, "admittance_ydq_plus.png", show)
        except Exception as exc:  # noqa: BLE001 - plotting should not abort the whole suite
            saved["admittance_error"] = None
            result.warnings.append(f"Admittance spectrum plot skipped: {exc}")

    if include_dq_axes:
        try:
            figs = plot_admittance_dq_axes(result, show=False)
            for axis_name, fig in figs.items():
                saved[f"admittance_{axis_name}"] = _maybe_save(
                    fig, output_path, f"admittance_{axis_name}.png", show
                )
        except Exception as exc:  # noqa: BLE001
            saved["admittance_dq_axes_error"] = None
            result.warnings.append(f"Admittance dq-axis Bode plots skipped: {exc}")

    if include_strength:
        try:
            fig_strength = plot_grid_strength(result, show=False)
            saved["grid_strength"] = _maybe_save(fig_strength, output_path, "grid_strength.png", show)
        except Exception as exc:  # noqa: BLE001
            saved["grid_strength_error"] = None
            result.warnings.append(f"Grid-strength plot skipped: {exc}")


    if not show:
        plt.close("all")
    return saved


def plot_greybox_summary(
    greybox: GreyboxResult,
    *,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Path | None]:
    """Plot Ysys/Zsys channels plus Layer 1/2 charts when available."""

    import matplotlib.pyplot as plt

    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path | None] = {}

    fig_y = plot_whole_system_transfer(
        greybox.admittance.values,
        greybox.admittance.frequencies_hz,
        title="Whole-system admittance Ysys(0,0)",
        show=False,
    )
    saved["ysys"] = _maybe_save(fig_y, output_path, "ysys_00.png", show)
    fig_z = plot_whole_system_transfer(
        greybox.impedance.values,
        greybox.impedance.frequencies_hz,
        title="Whole-system impedance Zsys(0,0)",
        show=False,
    )
    saved["zsys"] = _maybe_save(fig_z, output_path, "zsys_00.png", show)

    for mode in greybox.modes:
        if mode.layer1 or mode.layer2:
            fig = plot_apparatus_layer12(mode, show=False)
            saved[f"apparatus_layer12_mode{mode.mode_index}"] = _maybe_save(
                fig, output_path, f"apparatus_layer12_mode{mode.mode_index}.png", show
            )
    for sens in greybox.sensitivity:
        if sens.layer12:
            fig = plot_sensitivity_layer12(sens, show=False)
            saved[f"sensitivity_layer12_mode{sens.mode_index}"] = _maybe_save(
                fig, output_path, f"sensitivity_layer12_mode{sens.mode_index}.png", show
            )

    if not show:
        plt.close("all")
    return saved


def _maybe_save(fig, output_dir: Path | None, filename: str, show: bool) -> Path | None:
    import matplotlib.pyplot as plt

    path = None
    if output_dir is not None:
        path = output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        print(f"  [plot] Showing {filename} — close the figure window to continue...")
        plt.show()
    return path
