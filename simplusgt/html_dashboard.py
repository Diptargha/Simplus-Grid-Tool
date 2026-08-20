"""Interactive Plotly HTML dashboard for SimplusGT analysis plots.

Produces a single self-contained HTML file with hover/zoom and a mode
selector for apparatus / sensitivity Layer 1/2 charts.
"""

from __future__ import annotations

import html
import webbrowser
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import bus_strength, stability_report
from .dss import dss2ss
from .greybox import GreyboxResult
from .pipeline import RunResult
from .plotting import (
    COMPLEX_T,
    _active_apparatus_bus,
    _admittance_plot_freq_count,
    _bus_type_vif,
    _ss_frequency_response,
)

_PIE_TOP_N = 8


def write_analysis_dashboard(
    result: RunResult,
    greybox: GreyboxResult | None = None,
    *,
    output_path: str | Path,
    include_pole: bool = True,
    include_strength: bool = True,
    include_admittance: bool = False,
    include_dq_axes: bool = False,
    include_greybox: bool = True,
    case_label: str | None = None,
    open_browser: bool = False,
) -> Path:
    """Build an interactive HTML dashboard and write it to ``output_path``."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = case_label or "SimplusGT analysis"
    sections: list[tuple[str, str, str]] = []  # (section_id, heading, figure_html)
    plotly_js_url = _plotly_js_cdn_url()

    if include_pole:
        sections.append(
            (
                "pole-map",
                "Pole map",
                _fig_to_div(_pole_map_figure(result.eigenvalues, go=go, make_subplots=make_subplots)),
            )
        )

    if include_strength:
        try:
            sections.append(
                (
                    "grid-strength",
                    "Grid strength",
                    _fig_to_div(_grid_strength_figure(result, go=go)),
                )
            )
        except ValueError as exc:
            sections.append(
                (
                    "grid-strength",
                    "Grid strength",
                    f"<p class='note'>Skipped: {html.escape(str(exc))}</p>",
                )
            )

    if include_admittance:
        sections.extend(_admittance_spectrum_sections(result, go=go, make_subplots=make_subplots))

    if include_dq_axes:
        sections.extend(_admittance_dq_sections(result, go=go, make_subplots=make_subplots))

    mode_panels: list[dict[str, Any]] = []
    if include_greybox and greybox is not None:
        sections.append(
            (
                "ysys",
                "Whole-system admittance Ysys(0,0)",
                _fig_to_div(
                    _transfer_bode_figure(
                        greybox.admittance.values,
                        greybox.admittance.frequencies_hz,
                        title="Ysys(0,0)",
                        go=go,
                        make_subplots=make_subplots,
                    )
                ),
            )
        )
        sections.append(
            (
                "zsys",
                "Whole-system impedance Zsys(0,0)",
                _fig_to_div(
                    _transfer_bode_figure(
                        greybox.impedance.values,
                        greybox.impedance.frequencies_hz,
                        title="Zsys(0,0)",
                        go=go,
                        make_subplots=make_subplots,
                    )
                ),
            )
        )
        mode_panels = _build_mode_panels(result, greybox, go=go, make_subplots=make_subplots)

    page = _assemble_html(
        title=title,
        sections=sections,
        mode_panels=mode_panels,
        plotly_js_url=plotly_js_url,
    )
    output.write_text(page, encoding="utf-8")
    if open_browser:
        webbrowser.open(output.resolve().as_uri())
    return output


def _plotly_js_cdn_url() -> str:
    """CDN URL matching the plotly.js version bundled with the installed Python package.

    Do not use ``plotly-latest.min.js`` — that alias is frozen at v1.58.5 and cannot
    render figures produced by plotly.py 5+/6+ (binary ``bdata`` arrays, modern subplots).
    """

    try:
        from plotly.offline import get_plotlyjs_version

        version = get_plotlyjs_version()
    except Exception:
        version = "3.7.0"
    return f"https://cdn.plot.ly/plotly-{version}.min.js"


def _fig_to_div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _pole_map_figure(eigenvalues: np.ndarray, *, go, make_subplots):
    report = stability_report(eigenvalues)
    eig_hz = report.eigenvalues_hz
    re = np.real(eig_hz)
    im = np.imag(eig_hz)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Global pole map", "Zoomed pole map"))
    hover = [
        f"λ={r:.4g}{i:+.4g}j Hz<br>f={abs(i):.4g} Hz"
        for r, i in zip(re, im, strict=False)
    ]
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(
                x=re,
                y=im,
                mode="markers",
                marker=dict(symbol="x", size=8, color="#1f77b4"),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=[-80.0, 0.0],
            y=[800.0, 0.0],
            mode="lines",
            line=dict(dash="dash", color="#1f77b4", width=1.5),
            name="10% damping",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[-80.0, 0.0],
            y=[-800.0, 0.0],
            mode="lines",
            line=dict(dash="dash", color="#1f77b4", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Real Part (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Real Part (Hz)", range=[-80, 20], row=1, col=2)
    fig.update_yaxes(title_text="Imaginary Part (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Imaginary Part (Hz)", range=[-150, 150], row=1, col=2)
    fig.update_layout(title_text="Pole map", height=480, margin=dict(t=60, b=40))
    return fig


def _grid_strength_figure(result: RunResult, *, go):
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

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in graph.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    nodes = list(graph.nodes)
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    strength = np.log10(np.maximum(np.abs(ydiag), 1e-30))
    node_strength = [float(strength[n - 1]) for n in nodes]
    colors = []
    for node in nodes:
        if node in vbus:
            colors.append("green")
        elif node in ibus:
            colors.append("red")
        elif node in fbus:
            colors.append("#b0b0b0")
        else:
            colors.append("#1f77b4")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#aaaaaa"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[str(n) for n in nodes],
            textposition="top center",
            marker=dict(
                size=18,
                color=node_strength,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="log10(|Y|)"),
                line=dict(width=2, color=colors),
            ),
            customdata=np.column_stack([nodes, node_strength]),
            hovertemplate="Bus %{customdata[0]}<br>log10(|Y|)=%{customdata[1]:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title_text="Grid strength",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(t=50, b=20),
    )
    return fig


def _transfer_bode_figure(values: np.ndarray, frequencies_hz: np.ndarray, *, title: str, go, make_subplots, row: int = 0, col: int = 0):
    vals = np.asarray(values, dtype=complex)
    freq = np.asarray(frequencies_hz, dtype=float)
    if vals.ndim != 3 or vals.shape[0] == 0:
        channel = np.array([], dtype=complex)
    else:
        r = min(row, vals.shape[1] - 1) if vals.shape[1] else 0
        c = min(col, vals.shape[2] - 1) if vals.shape[2] else 0
        channel = vals[:, r, c]
    mag = np.maximum(np.abs(channel), 1e-30)
    phase = np.unwrap(np.angle(channel)) * 180 / np.pi if channel.size else np.array([])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=freq, y=mag, mode="lines", name="|G|", hovertemplate="f=%{x:.4g} Hz<br>|G|=%{y:.4g}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=freq, y=phase, mode="lines", name="phase", hovertemplate="f=%{x:.4g} Hz<br>∠=%{y:.2f}°<extra></extra>"),
        row=2,
        col=1,
    )
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(type="log", title_text="|G|", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig.update_layout(title_text=title, height=480, showlegend=False, margin=dict(t=50, b=40))
    return fig


def _admittance_spectrum_sections(result: RunResult, *, go, make_subplots) -> list[tuple[str, str, str]]:
    if result.whole_system_ss is None:
        a, b, c, d = dss2ss(result.whole_system_dss)
    else:
        a, b, c, d = result.whole_system_ss
    n_freq = _admittance_plot_freq_count(int(a.shape[0]), 500)
    omega_p = np.logspace(-1, 4, n_freq) * 2 * np.pi
    freq_p = omega_p / (2 * np.pi)
    omega_pn = np.concatenate([-np.flip(omega_p), omega_p])
    freq_n = omega_pn / (2 * np.pi)
    half = omega_p.size
    freq_pos = freq_n[half:]
    t_inv = np.linalg.inv(COMPLEX_T)

    fig_dd = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Ydd magnitude", "Ydd phase"))
    fig_cplx = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=("Ydq+ mag (neg f)", "Ydq+ mag (pos f)", "Ydq+ phase (neg f)", "Ydq+ phase (pos f)"),
    )
    num_bus = int(np.max(result.buses_after_load[:, 0]))
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
        y_dq = _ss_frequency_response(a, b[:, in_idx], c[out_idx, :], d[np.ix_(out_idx, in_idx)], omega_pn)
        y_dd = y_dq[half:, 0, 0]
        label = f"Bus{bus}"
        mag = np.maximum(np.abs(y_dd), 1e-30)
        phase = np.unwrap(np.angle(y_dd)) * 180 / np.pi
        fig_dd.add_trace(go.Scatter(x=freq_p, y=mag, mode="lines", name=label), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=freq_p, y=phase, mode="lines", name=label, showlegend=False), row=2, col=1)
        if d[np.ix_(out_idx, in_idx)].shape[0] >= 2 and d[np.ix_(out_idx, in_idx)].shape[1] >= 2:
            y_cplx = np.empty_like(y_dq)
            for k in range(y_dq.shape[0]):
                y_cplx[k] = COMPLEX_T @ y_dq[k] @ t_inv
            y11 = y_cplx[:, 0, 0]
            fig_cplx.add_trace(
                go.Scatter(x=np.abs(freq_n[:half]), y=np.maximum(np.abs(y11[:half]), 1e-30), mode="lines", name=label),
                row=1,
                col=1,
            )
            fig_cplx.add_trace(
                go.Scatter(x=freq_pos, y=np.maximum(np.abs(y11[half:]), 1e-30), mode="lines", name=label, showlegend=False),
                row=1,
                col=2,
            )
            fig_cplx.add_trace(
                go.Scatter(x=np.abs(freq_n[:half]), y=np.unwrap(np.angle(y11[:half])) * 180 / np.pi, mode="lines", name=label, showlegend=False),
                row=2,
                col=1,
            )
            fig_cplx.add_trace(
                go.Scatter(x=freq_pos, y=np.unwrap(np.angle(y11[half:])) * 180 / np.pi, mode="lines", name=label, showlegend=False),
                row=2,
                col=2,
            )

    fig_dd.update_xaxes(type="log", title_text="Frequency (Hz)", row=2, col=1)
    fig_dd.update_yaxes(type="log", title_text="|Y|", row=1, col=1)
    fig_dd.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig_dd.update_layout(title_text="Transfer Function Matrix dq frame: Ydd", height=520)
    for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig_cplx.update_xaxes(type="log", row=r, col=c)
    fig_cplx.update_yaxes(type="log", row=1, col=1)
    fig_cplx.update_yaxes(type="log", row=1, col=2)
    fig_cplx.update_layout(title_text="Complex Vector dq frame: Ydq+", height=560)
    return [
        ("admittance-ydd", "Bus admittance Ydd", _fig_to_div(fig_dd)),
        ("admittance-ydq", "Bus admittance Ydq+", _fig_to_div(fig_cplx)),
    ]


def _admittance_dq_sections(result: RunResult, *, go, make_subplots) -> list[tuple[str, str, str]]:
    if result.whole_system_ss is None:
        a, b, c, d = dss2ss(result.whole_system_dss)
    else:
        a, b, c, d = result.whole_system_ss
    n_freq = _admittance_plot_freq_count(int(a.shape[0]), 500)
    omega = np.logspace(-1, 4, n_freq) * 2 * np.pi
    freq = omega / (2 * np.pi)
    axis_map = {"dd": (0, 0), "dq": (0, 1), "qd": (1, 0), "qq": (1, 1)}
    apparatus_indices = [
        idx
        for idx, app_type in enumerate(result.netlists.apparatus_types)
        if _active_apparatus_bus(int(app_type))
    ]
    sections: list[tuple[str, str, str]] = []
    for axis_name, (out_bias, in_bias) in axis_map.items():
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"|Y| {axis_name}", f"Phase {axis_name}"))
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
            values = _ss_frequency_response(
                a,
                b[:, [in_idx[in_bias]]],
                c[[out_idx[out_bias]], :],
                d[np.ix_([out_idx[out_bias]], [in_idx[in_bias]])],
                omega,
            )[:, 0, 0]
            label = f"Node{bus}"
            fig.add_trace(
                go.Scatter(x=freq, y=np.maximum(np.abs(values), 1e-30), mode="lines", name=label),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=np.unwrap(np.angle(values)) * 180 / np.pi,
                    mode="lines",
                    name=label,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(type="log", title_text="|Y|", row=1, col=1)
        fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
        fig.update_layout(title_text=f"Admittance Bode: {axis_name[0]}-{axis_name[1]} axis", height=480)
        sections.append((f"admittance-{axis_name}", f"Admittance {axis_name}", _fig_to_div(fig)))
    return sections


def _build_mode_panels(result: RunResult, greybox: GreyboxResult, *, go, make_subplots) -> list[dict[str, Any]]:
    eig = np.asarray(result.eigenvalues, dtype=complex).ravel()
    modes_by_index = {int(m.mode_index): m for m in greybox.modes}
    sens_by_index = {int(s.mode_index): s for s in greybox.sensitivity}
    mode_indices = sorted(set(modes_by_index) | set(sens_by_index))
    panels: list[dict[str, Any]] = []
    for mode_index in mode_indices:
        eig_val = eig[mode_index] if 0 <= mode_index < eig.size else complex("nan")
        label = _mode_option_label(mode_index, eig_val)
        parts: list[str] = []
        mode = modes_by_index.get(mode_index)
        n_l3 = len(getattr(mode, "layer3", []) or []) if mode is not None else 0
        # #region agent log
        try:
            import json as _json
            import time as _time

            with open(
                r"C:\Users\z004z29x\PycharmProjects\Simplus-Grid-Tool\debug-e9b6f6.log",
                "a",
                encoding="utf-8",
            ) as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "e9b6f6",
                            "runId": "post-fix",
                            "hypothesisId": "A",
                            "location": "html_dashboard.py:_build_mode_panels",
                            "message": "mode panel layer availability",
                            "data": {
                                "mode_index": mode_index,
                                "n_layer1": len(getattr(mode, "layer1", []) or []) if mode else 0,
                                "n_layer2": len(getattr(mode, "layer2", []) or []) if mode else 0,
                                "n_layer3": n_l3,
                                "will_plot_layer3": n_l3 > 0,
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        if mode is not None and (mode.layer1 or mode.layer2):
            parts.append("<h3>Apparatus Layer 1 / Layer 2</h3>")
            parts.append(_fig_to_div(_apparatus_layer12_figure(mode, go=go, make_subplots=make_subplots)))
        if mode is not None and mode.layer3:
            parts.append("<h3>Apparatus Layer 3 (parameter sensitivity)</h3>")
            parts.append(
                "<p class='note'>Bars show ∂λ/∂p in pu·Hz: real = damping sense, imag = frequency sense.</p>"
            )
            parts.append(_fig_to_div(_apparatus_layer3_figure(mode, go=go, make_subplots=make_subplots)))
        sens = sens_by_index.get(mode_index)
        if sens is not None and sens.layer12:
            parts.append("<h3>Sensitivity Layer 1 / Layer 2</h3>")
            parts.append(_fig_to_div(_sensitivity_layer12_figure(sens, go=go, make_subplots=make_subplots)))
        if sens is not None and sens.layer3:
            parts.append("<h3>Sensitivity Layer 3 (line R/X)</h3>")
            parts.append(_fig_to_div(_sensitivity_layer3_figure(sens, go=go, make_subplots=make_subplots)))
        if not parts:
            parts.append("<p class='note'>No Layer 1/2/3 data for this mode.</p>")
        panels.append({"mode_index": mode_index, "label": label, "body": "\n".join(parts)})
    return panels


def _mode_option_label(mode_index: int, eigenvalue: complex) -> str:
    if not np.isfinite(eigenvalue):
        return f"Mode {mode_index}"
    eig_hz = eigenvalue / (2 * np.pi)
    freq = abs(float(np.imag(eig_hz)))
    sigma = float(np.real(eigenvalue))
    omega = abs(float(np.imag(eigenvalue)))
    denom = abs(eigenvalue)
    damp = None if denom == 0 else -sigma / denom
    damp_txt = f", ζ={damp:.3f}" if damp is not None else ""
    return f"Mode {mode_index} — f={freq:.3g} Hz{damp_txt}"


def _pie_with_other(labels: Sequence[str], values: np.ndarray, *, top_n: int = _PIE_TOP_N):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [], np.array([])
    if values.sum() <= 0:
        values = np.ones(len(values)) / max(len(values), 1)
    order = np.argsort(-values)
    labels_arr = [labels[i] for i in order]
    values = values[order]
    if len(values) <= top_n:
        return labels_arr, values
    keep_labels = labels_arr[:top_n]
    keep_values = values[:top_n]
    other = float(values[top_n:].sum())
    keep_labels.append(f"Other ({len(values) - top_n})")
    keep_values = np.append(keep_values, other)
    return keep_labels, keep_values


def _short_apparatus_label(item: dict[str, Any]) -> str:
    idx = item.get("apparatus_index")
    if idx is not None:
        return f"App {int(idx) + 1}"
    label = str(item.get("label", "?"))
    return label if len(label) <= 16 else label[:13] + "…"


def _short_component_label(component: str) -> str:
    text = str(component)
    if text.startswith("Branch "):
        return "Br " + text[len("Branch ") :]
    if text.startswith("Node "):
        return text
    return text if len(text) <= 16 else text[:13] + "…"


def _short_layer3_param_label(item: dict[str, Any]) -> str:
    app = _short_apparatus_label(item)
    param = str(item.get("parameter", "?"))
    return f"{app}:{param}"


_LAYER3_TOP_N = 30


def _complex_field(item: dict[str, Any], key: str) -> complex:
    value = item.get(key, 0.0)
    try:
        return complex(value)
    except (TypeError, ValueError):
        return complex(0.0)


def _apparatus_layer3_figure(mode_result: Any, *, go, make_subplots, top_n: int = _LAYER3_TOP_N):
    """Horizontal bars of parameter ∂λ sensitivities (pu·Hz), real/imag."""

    records = list(getattr(mode_result, "layer3", []) or [])
    mode_index = getattr(mode_result, "mode_index", "?")
    if not records:
        fig = go.Figure()
        fig.update_layout(title_text=f"Apparatus Layer 3 (mode {mode_index}) — no data", height=320)
        return fig

    scored = []
    for item in records:
        d_pu = _complex_field(item, "d_lambda_pu_hz")
        scored.append((abs(float(np.real(d_pu))) + abs(float(np.imag(d_pu))), item, d_pu))
    scored.sort(key=lambda row: row[0], reverse=True)
    scored = scored[: max(1, int(top_n))]

    short = [_short_layer3_param_label(item) for _, item, _ in scored]
    full = [
        f"{item.get('label', short[i])} · {item.get('parameter', '?')}"
        for i, (_, item, _) in enumerate(scored)
    ]
    real_pu = [float(np.real(d_pu)) for _, _, d_pu in scored]
    imag_pu = [float(np.imag(d_pu)) for _, _, d_pu in scored]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_yaxes=True,
        subplot_titles=(
            f"Layer 3 real (pu·Hz) — damping sense (mode {mode_index})",
            f"Layer 3 imag (pu·Hz) — frequency sense (mode {mode_index})",
        ),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            y=short,
            x=real_pu,
            orientation="h",
            customdata=full,
            hovertemplate="%{customdata}<br>Re(∂λ/∂p)=%{x:.4g} pu·Hz<extra></extra>",
            marker_color="#9467bd",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=short,
            x=imag_pu,
            orientation="h",
            customdata=full,
            hovertemplate="%{customdata}<br>Im(∂λ/∂p)=%{x:.4g} pu·Hz<extra></extra>",
            marker_color="#8c564b",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="Re(∂λ/∂p) [pu·Hz]", row=1, col=1)
    fig.update_xaxes(title_text="Im(∂λ/∂p) [pu·Hz]", row=2, col=1)
    height = max(520, 28 * len(short) + 220)
    fig.update_layout(height=height, margin=dict(t=70, b=50, l=100, r=30))
    return fig


def _sensitivity_layer3_figure(sens_result: Any, *, go, make_subplots, top_n: int = _LAYER3_TOP_N):
    """Horizontal bars for sensitivity Layer 3 (branch R/X parameter ∂λ)."""

    records = list(getattr(sens_result, "layer3", []) or [])
    mode_index = getattr(sens_result, "mode_index", "?")
    if not records:
        fig = go.Figure()
        fig.update_layout(title_text=f"Sensitivity Layer 3 (mode {mode_index}) — no data", height=320)
        return fig

    scored = []
    for item in records:
        d_pu = _complex_field(item, "d_lambda_pu_hz")
        scored.append((abs(float(np.real(d_pu))) + abs(float(np.imag(d_pu))), item, d_pu))
    scored.sort(key=lambda row: row[0], reverse=True)
    scored = scored[: max(1, int(top_n))]

    short = [_short_component_label(str(item.get("component", "?"))) for _, item, _ in scored]
    full = [str(item.get("component", "?")) for _, item, _ in scored]
    real_pu = [float(np.real(d_pu)) for _, _, d_pu in scored]
    imag_pu = [float(np.imag(d_pu)) for _, _, d_pu in scored]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_yaxes=True,
        subplot_titles=(
            f"Sens Layer 3 real (pu·Hz) — damping (mode {mode_index})",
            f"Sens Layer 3 imag (pu·Hz) — frequency (mode {mode_index})",
        ),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            y=short,
            x=real_pu,
            orientation="h",
            customdata=full,
            hovertemplate="%{customdata}<br>Re(∂λ/∂p)=%{x:.4g} pu·Hz<extra></extra>",
            marker_color="#17becf",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=short,
            x=imag_pu,
            orientation="h",
            customdata=full,
            hovertemplate="%{customdata}<br>Im(∂λ/∂p)=%{x:.4g} pu·Hz<extra></extra>",
            marker_color="#bcbd22",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    height = max(520, 28 * len(short) + 220)
    fig.update_layout(height=height, margin=dict(t=70, b=50, l=100, r=30))
    return fig


def _apparatus_layer12_figure(mode_result: Any, *, go, make_subplots):
    layer1 = list(getattr(mode_result, "layer1", []) or [])
    layer2 = list(getattr(mode_result, "layer2", []) or [])
    mode_index = getattr(mode_result, "mode_index", "?")
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]],
        column_widths=[0.45, 0.55],
        subplot_titles=(
            f"Apparatus Layer 1 (mode {mode_index})",
            "Layer 2 real (pu) — damping sense",
            "Layer 2 imag (pu) — frequency sense",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    if layer1:
        full_labels = [item.get("label", f"App {item.get('apparatus_index', '?')}") for item in layer1]
        raw = np.array([abs(float(item.get("normalized", item.get("value", 0.0)))) for item in layer1], dtype=float)
        pie_labels, pie_values = _pie_with_other(full_labels, raw)
        fig.add_trace(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                textinfo="percent",
                hovertemplate="%{label}<br>%{percent}<br>value=%{value:.4g}<extra></extra>",
                sort=False,
            ),
            row=1,
            col=1,
        )
    if layer2:
        order = np.argsort(
            -np.array(
                [
                    abs(float(item.get("real_normalized", 0.0))) + abs(float(item.get("imag_normalized", 0.0)))
                    for item in layer2
                ]
            )
        )
        items = [layer2[i] for i in order]
        short = [_short_apparatus_label(item) for item in items]
        full = [item.get("label", short[i]) for i, item in enumerate(items)]
        real_pu = [float(item.get("real_normalized", 0.0)) for item in items]
        imag_pu = [float(item.get("imag_normalized", 0.0)) for item in items]
        fig.add_trace(
            go.Bar(
                y=short,
                x=real_pu,
                orientation="h",
                customdata=full,
                hovertemplate="%{customdata}<br>damping (Re)=%{x:.4g} pu<extra></extra>",
                marker_color="#1f77b4",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                y=short,
                x=imag_pu,
                orientation="h",
                customdata=full,
                hovertemplate="%{customdata}<br>frequency (Im)=%{x:.4g} pu<extra></extra>",
                marker_color="#ff7f0e",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=2, col=2)
        # Axis meaning is already in subplot titles; avoid x-title vs pie-legend overlap.
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_layout(
        height=640,
        margin=dict(t=70, b=140, l=80, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            x=0.0,
            xanchor="left",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return fig


def _sensitivity_layer12_figure(sens_result: Any, *, go, make_subplots):
    records = list(getattr(sens_result, "layer12", []) or [])
    mode_index = getattr(sens_result, "mode_index", "?")
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]],
        column_widths=[0.45, 0.55],
        subplot_titles=(
            f"Sensitivity Layer 1 (mode {mode_index})",
            "Sens Layer 2 real (pu) — damping sense",
            "Sens Layer 2 imag (pu) — frequency sense",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    full_labels = [item.get("component", "?") for item in records]
    raw = np.array([abs(float(item.get("layer1_normalized", item.get("layer1", 0.0)))) for item in records], dtype=float)
    pie_labels, pie_values = _pie_with_other(full_labels, raw)
    fig.add_trace(
        go.Pie(
            labels=pie_labels,
            values=pie_values,
            textinfo="percent",
            hovertemplate="%{label}<br>%{percent}<br>value=%{value:.4g}<extra></extra>",
            sort=False,
        ),
        row=1,
        col=1,
    )
    order = np.argsort(
        -np.array(
            [
                abs(float(item.get("layer2_real_normalized", 0.0)))
                + abs(float(item.get("layer2_imag_normalized", 0.0)))
                for item in records
            ]
        )
    )
    items = [records[i] for i in order]
    short = [_short_component_label(str(item.get("component", "?"))) for item in items]
    full = [str(item.get("component", "?")) for item in items]
    real_pu = [float(item.get("layer2_real_normalized", 0.0)) for item in items]
    imag_pu = [float(item.get("layer2_imag_normalized", 0.0)) for item in items]
    fig.add_trace(
        go.Bar(
            y=short,
            x=real_pu,
            orientation="h",
            customdata=full,
            hovertemplate="%{customdata}<br>damping (Re)=%{x:.4g} pu<extra></extra>",
            marker_color="#2ca02c",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            y=short,
            x=imag_pu,
            orientation="h",
            customdata=full,
            hovertemplate="%{customdata}<br>frequency (Im)=%{x:.4g} pu<extra></extra>",
            marker_color="#d62728",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_layout(
        height=700,
        margin=dict(t=70, b=160, l=80, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            x=0.0,
            xanchor="left",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return fig


def _assemble_html(
    *,
    title: str,
    sections: list[tuple[str, str, str]],
    mode_panels: list[dict[str, Any]],
    plotly_js_url: str | None = None,
) -> str:
    js_url = plotly_js_url or _plotly_js_cdn_url()
    nav_links = "".join(
        f'<a href="#{html.escape(sec_id)}">{html.escape(heading)}</a>' for sec_id, heading, _ in sections
    )
    if mode_panels:
        nav_links += '<a href="#mode-layers">Mode Layer 1/2/3</a>'

    section_html = []
    for sec_id, heading, body in sections:
        section_html.append(
            f'<section id="{html.escape(sec_id)}" class="card">'
            f"<h2>{html.escape(heading)}</h2>{body}</section>"
        )

    mode_html = ""
    if mode_panels:
        options = "".join(
            f'<option value="mode-{panel["mode_index"]}">{html.escape(panel["label"])}</option>'
            for panel in mode_panels
        )
        panels = []
        for idx, panel in enumerate(mode_panels):
            hidden = "" if idx == 0 else " hidden"
            panels.append(
                f'<div class="mode-panel" id="mode-{panel["mode_index"]}"{hidden}>'
                f'<p class="mode-title">{html.escape(panel["label"])}</p>'
                f'{panel["body"]}</div>'
            )
        mode_html = f"""
<section id="mode-layers" class="card">
  <h2>Apparatus &amp; sensitivity Layer 1/2/3</h2>
  <label for="mode-select">Mode</label>
  <select id="mode-select">{options}</select>
  {"".join(panels)}
</section>
<script>
(function () {{
  const select = document.getElementById("mode-select");
  if (!select) return;
  const panels = Array.from(document.querySelectorAll(".mode-panel"));
  function resizePlots(root) {{
    if (!root || !window.Plotly) return;
    root.querySelectorAll(".plotly-graph-div").forEach((gd) => {{
      try {{ Plotly.Plots.resize(gd); }} catch (err) {{}}
    }});
  }}
  function showMode(id) {{
    panels.forEach((el) => {{
      el.hidden = el.id !== id;
    }});
    const active = document.getElementById(id);
    // Plots created while hidden get a collapsed layout; resize after becoming visible.
    requestAnimationFrame(() => {{
      resizePlots(active);
      requestAnimationFrame(() => resizePlots(active));
    }});
  }}
  select.addEventListener("change", () => showMode(select.value));
  showMode(select.value);
}})();
</script>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)} — SimplusGT dashboard</title>
  <script charset="utf-8" src="{html.escape(js_url)}"></script>
  <style>
    :root {{
      --bg: #f6f7f9;
      --card: #ffffff;
      --text: #1b1f24;
      --muted: #5b6570;
      --accent: #0b6bcb;
      --border: #d8dee6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255,255,255,0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(6px);
      padding: 0.75rem 1.25rem;
    }}
    header h1 {{
      margin: 0 0 0.4rem 0;
      font-size: 1.25rem;
      font-weight: 650;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 0.9rem;
    }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 0.92rem;
    }}
    nav a:hover {{ text-decoration: underline; }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 1rem 1.25rem 2.5rem;
      display: grid;
      gap: 1rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.85rem 1rem 1rem;
      box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }}
    .card h2 {{
      margin: 0 0 0.6rem 0;
      font-size: 1.1rem;
    }}
    .card h3 {{
      margin: 1rem 0 0.4rem 0;
      font-size: 1rem;
    }}
    .note {{ color: var(--muted); }}
    .mode-title {{ font-weight: 600; color: var(--muted); }}
    #mode-select {{
      margin: 0.25rem 0 0.75rem 0.5rem;
      padding: 0.35rem 0.55rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.95rem;
    }}
    label[for="mode-select"] {{ font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)} — interactive dashboard</h1>
    <nav>{nav_links}</nav>
  </header>
  <main>
    {"".join(section_html)}
    {mode_html}
  </main>
</body>
</html>
"""
