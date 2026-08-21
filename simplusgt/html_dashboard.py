"""Interactive Plotly HTML dashboard for SimplusGT analysis plots.

Produces a single self-contained HTML file with hover/zoom and a mode
selector for apparatus / sensitivity Layer 1/2 charts.
"""

from __future__ import annotations

import html
import json
import re
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
    ysys_modal_metrics: list[dict[str, Any]] | None = None
    if include_pole or (include_greybox and greybox is not None):
        ysys_modal_metrics = _ysys_modal_metrics(result)

    if include_pole:
        sections.append(
            (
                "pole-map",
                "Pole map",
                _fig_to_div(
                    _pole_map_figure(
                        result.eigenvalues,
                        go=go,
                        make_subplots=make_subplots,
                        modal_metrics=ysys_modal_metrics,
                    )
                ),
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
                "Whole-system admittance Ysys",
                _transfer_bode_section_html(
                    greybox.admittance.values,
                    greybox.admittance.frequencies_hz,
                    input_labels=greybox.admittance.input_labels,
                    output_labels=greybox.admittance.output_labels,
                    name="Ysys",
                    mag_axis_title="|Y| (p.u.)",
                    go=go,
                    make_subplots=make_subplots,
                    modal_overlay=_ysys_modal_overlay_payload(
                        result,
                        greybox.admittance.frequencies_hz,
                        metrics=ysys_modal_metrics,
                    ),
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
    html_snippet = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    # Plotly may bake a fixed pixel width into the outer wrapper; force full card width.
    return (
        html_snippet.replace('style="height:', 'style="width:100%; height:', 1)
        if 'style="width:100%' not in html_snippet[:200]
        else html_snippet
    )


def _pole_map_figure(
    eigenvalues: np.ndarray,
    *,
    go,
    make_subplots,
    modal_metrics: list[dict[str, Any]] | None = None,
):
    report = stability_report(eigenvalues)
    eig_hz = report.eigenvalues_hz
    re = np.real(eig_hz)
    im = np.imag(eig_hz)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Global pole map", "Zoomed pole map"))
    hover: list[str] = []
    vis_colors: list[float] = []
    for r, i, lam_hz in zip(re, im, eig_hz, strict=False):
        lam_rad = complex(lam_hz) * 2.0 * np.pi
        metric = _nearest_modal_metric(lam_rad, modal_metrics)
        zeta = -np.real(lam_rad) / abs(lam_rad) if abs(lam_rad) > 0 else float("nan")
        if metric is None:
            hover.append(
                f"λ={r:.4g}{i:+.4g}j Hz<br>f={abs(i):.4g} Hz<br>ζ={zeta:.4g}<br>"
                "Ysys residue: n/a"
            )
            vis_colors.append(float("nan"))
        else:
            hover.append(
                f"λ={r:.4g}{i:+.4g}j Hz<br>f={abs(i):.4g} Hz<br>ζ={zeta:.4g}<br>"
                f"||R||_F={metric['r_fro']:.4g}<br>"
                f"max|R|={metric['r_max']:.4g}<br>"
                f"Bode visibility max|R|/|σ|={metric['visibility']:.4g}<br>"
                f"||R||_F/|σ|={metric.get('visibility_fro', float('nan')):.4g}<br>"
                f"{'unstable' if np.real(lam_rad) > 1e-6 else 'stable'}"
            )
            vis_colors.append(float(np.log10(max(metric["visibility"], 1e-30))))

    marker_kwargs: dict[str, Any] = dict(symbol="x", size=8)
    if modal_metrics and np.any(np.isfinite(vis_colors)):
        marker_kwargs.update(
            color=vis_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="log10(max|R|/|σ|)", side="right"),
                x=1.02,
                len=0.72,
                y=0.42,
                yanchor="middle",
                thickness=14,
            ),
            cmin=float(np.nanmin(vis_colors)),
            cmax=float(np.nanmax(vis_colors)),
        )
    else:
        marker_kwargs["color"] = "#1f77b4"

    for col in (1, 2):
        mk = dict(marker_kwargs)
        if col == 1 and "showscale" in mk:
            mk = dict(mk)
            mk["showscale"] = False
        fig.add_trace(
            go.Scatter(
                x=re,
                y=im,
                mode="markers",
                marker=mk,
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
            showlegend=True,
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
    fig.update_layout(
        title_text="Pole map (hover: Ysys residue / Bode visibility)",
        height=520,
        margin=dict(t=60, b=70, r=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            x=0.75,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    return fig


def _ysys_ss_abcd(result: RunResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not result.port_i or not result.port_v:
        return None
    ysys = result.whole_system_dss.truncate(result.port_i, result.port_v)
    try:
        return dss2ss(ysys)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _ysys_modal_metrics(result: RunResult) -> list[dict[str, Any]]:
    """Per-eigenvalue Ysys residue norms and Bode-visibility |R|_F / |σ| (σ in rad/s)."""

    from scipy.linalg import eig

    abcd = _ysys_ss_abcd(result)
    if abcd is None:
        return []
    a, b, c, _d = abcd
    if a.size == 0:
        return []
    evals, left, right = eig(a, left=True, right=True)
    metrics: list[dict[str, Any]] = []
    for idx, lam in enumerate(evals):
        if not np.isfinite(lam):
            continue
        phi = right[:, idx : idx + 1]
        psi = left[:, idx].conj()[None, :]
        den = (psi @ phi)[0, 0]
        if abs(den) < 1e-18:
            residue = np.zeros((c.shape[0], b.shape[1]), dtype=complex)
        else:
            residue = c @ phi @ (psi / den) @ b
        r_fro = float(np.linalg.norm(residue, "fro"))
        r_max = float(np.max(np.abs(residue))) if residue.size else 0.0
        sigma = abs(float(np.real(lam)))
        # Bode-axis peak scale for the strongest port entry (more relevant than Frobenius).
        visibility = r_max / max(sigma, 1e-12)
        visibility_fro = r_fro / max(sigma, 1e-12)
        metrics.append(
            {
                "eigenvalue": complex(lam),
                "r_fro": r_fro,
                "r_max": r_max,
                "visibility": visibility,
                "visibility_fro": visibility_fro,
                "residue": residue,
            }
        )
    return metrics


def _nearest_modal_metric(
    lam_rad: complex,
    metrics: list[dict[str, Any]] | None,
    *,
    rel_tol: float = 1e-6,
) -> dict[str, Any] | None:
    if not metrics:
        return None
    best = None
    best_dist = float("inf")
    scale = max(abs(lam_rad), 1.0)
    for item in metrics:
        dist = abs(complex(item["eigenvalue"]) - complex(lam_rad))
        if dist < best_dist:
            best_dist = dist
            best = item
    if best is None or best_dist > max(1e-6, rel_tol * scale):
        return None
    return best


def _ysys_modal_overlay_payload(
    result: RunResult,
    frequencies_hz: np.ndarray,
    *,
    max_modes: int = 40,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compact mode list for client-side |R/(j2πf-λ)| overlays on Ysys magnitude."""

    metrics = metrics if metrics is not None else _ysys_modal_metrics(result)
    if not metrics:
        return {"modes": []}

    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in metrics:
        lam = complex(item["eigenvalue"])
        f_hz = abs(np.imag(lam) / (2.0 * np.pi))
        unstable = np.real(lam) > 1e-6
        # Prefer unstable and oscillatory modes with non-trivial residue.
        if item["r_fro"] < 1e-10 and not unstable:
            continue
        score = (1e6 if unstable else 0.0) + f_hz + 0.01 * item["visibility"]
        ranked.append((score, item))
    ranked.sort(key=lambda pair: -pair[0])

    # Keep one of each conjugate pair (prefer Im >= 0).
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, float]] = set()
    for _score, item in ranked:
        lam = complex(item["eigenvalue"])
        key = (round(np.real(lam), 4), round(abs(np.imag(lam)), 4))
        if key in seen_keys:
            continue
        if np.imag(lam) < -1e-9:
            # Prefer the +imag twin if present later/earlier; skip negative imag.
            continue
        seen_keys.add(key)
        residue = np.asarray(item["residue"], dtype=complex)
        selected.append(
            {
                "id": f"mode-{len(selected)}",
                "label": _modal_overlay_label(lam, item),
                "lambda_re": float(np.real(lam)),
                "lambda_im": float(np.imag(lam)),
                "r_fro": float(item["r_fro"]),
                "visibility": float(item["visibility"]),
                "residue_re": np.real(residue).astype(float).tolist(),
                "residue_im": np.imag(residue).astype(float).tolist(),
            }
        )
        if len(selected) >= max_modes:
            break

    freq = np.asarray(frequencies_hz, dtype=float).ravel()
    return {"freq": freq.astype(float).tolist(), "modes": selected}


def _modal_overlay_label(lam: complex, metric: dict[str, Any]) -> str:
    f_hz = abs(np.imag(lam) / (2.0 * np.pi))
    sigma_hz = np.real(lam) / (2.0 * np.pi)
    tag = "unstable" if np.real(lam) > 1e-6 else "stable"
    return (
        f"f≈{f_hz:.3g} Hz, σ≈{sigma_hz:.3g} Hz ({tag}); "
        f"max|R|/|σ|={metric['visibility']:.3g}"
    )


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


def _transfer_bode_section_html(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    input_labels: Sequence[str] | None,
    output_labels: Sequence[str] | None,
    name: str,
    mag_axis_title: str,
    go,
    make_subplots,
    modal_overlay: dict[str, Any] | None = None,
) -> str:
    """Dropdown of bus-to-bus channels; selected channel shows its full dq/scalar block."""

    vals = np.asarray(values, dtype=complex)
    freq = np.asarray(frequencies_hz, dtype=float).ravel()
    n_out = int(vals.shape[1]) if vals.ndim == 3 else 0
    n_in = int(vals.shape[2]) if vals.ndim == 3 else 0
    out_labs = _transfer_axis_labels(output_labels, n_out, prefix="y")
    in_labs = _transfer_axis_labels(input_labels, n_in, prefix="u")
    blocks = _mimo_dq_blocks(out_labs, in_labs)
    prefix = re.sub(r"[^a-z0-9]+", "", name.lower()) or "xfer"
    if not blocks or vals.ndim != 3 or freq.size == 0:
        return "<p class='note'>No transfer channels available.</p>"

    channel_payload: list[dict[str, Any]] = []
    for block in blocks:
        out_idx = list(block["out_indices"])
        in_idx = list(block["in_indices"])
        element_labels = [f"{out_labs[r]}/{in_labs[c]}" for r in out_idx for c in in_idx]
        mag: list[list[float]] = []
        phase: list[list[float]] = []
        for r in out_idx:
            for c in in_idx:
                channel = vals[:, r, c]
                mag.append(np.abs(channel).astype(float).tolist())
                phase.append((np.unwrap(np.angle(channel)) * 180.0 / np.pi).astype(float).tolist())
        channel_payload.append(
            {
                "id": block["id"],
                "label": block["label"],
                "n_out": len(out_idx),
                "n_in": len(in_idx),
                "out_indices": out_idx,
                "in_indices": in_idx,
                "element_labels": element_labels,
                "mag": mag,
                "phase": phase,
            }
        )

    first = blocks[0]
    mag_fig = _transfer_block_grid_figure(
        vals,
        freq,
        out_indices=first["out_indices"],
        in_indices=first["in_indices"],
        out_labels=out_labs,
        in_labels=in_labs,
        title=f"{name} magnitude — {first['label']}",
        quantity="mag",
        y_axis_title=mag_axis_title,
        go=go,
        make_subplots=make_subplots,
        include_modal_overlay_traces=True,
    )
    phase_fig = _transfer_block_grid_figure(
        vals,
        freq,
        out_indices=first["out_indices"],
        in_indices=first["in_indices"],
        out_labels=out_labs,
        in_labels=in_labs,
        title=f"{name} phase — {first['label']}",
        quantity="phase",
        y_axis_title="Phase (deg)",
        go=go,
        make_subplots=make_subplots,
        include_modal_overlay_traces=False,
    )

    select_id = f"{prefix}-channel-select"
    mode_select_id = f"{prefix}-mode-select"
    data_id = f"{prefix}-channel-data"
    modal_data_id = f"{prefix}-modal-data"
    mag_wrap_id = f"{prefix}-mag-wrap"
    phase_wrap_id = f"{prefix}-phase-wrap"
    title_id = f"{prefix}-channel-title"
    note_id = f"{prefix}-modal-note"
    options = "".join(
        f'<option value="{html.escape(ch["id"])}">{html.escape(ch["label"])}</option>'
        for ch in channel_payload
    )
    overlay = modal_overlay or {"modes": []}
    mode_options = '<option value="">None (Ysys only)</option>' + "".join(
        f'<option value="{html.escape(mode["id"])}">{html.escape(mode["label"])}</option>'
        for mode in overlay.get("modes", [])
    )
    payload_json = json.dumps(
        {"freq": freq.astype(float).tolist(), "channels": channel_payload},
        separators=(",", ":"),
    )
    modal_json = json.dumps(overlay, separators=(",", ":"))
    first_title = f"{channel_payload[0]['label']} ({channel_payload[0]['n_out']}×{channel_payload[0]['n_in']})"
    return f"""
<label for="{html.escape(select_id)}">Channel (output bus ← input bus)</label>
<select id="{html.escape(select_id)}">{options}</select>
<label for="{html.escape(mode_select_id)}">Modal contribution overlay</label>
<select id="{html.escape(mode_select_id)}">{mode_options}</select>
<p class="mode-title" id="{html.escape(title_id)}">{html.escape(first_title)}</p>
<p class="note" id="{html.escape(note_id)}">Select a mode to overlay |R/(j2πf−λ)| on the magnitude plots. Visibility is max|R|/|σ| over ports (σ in rad/s). A globally strong mode can still be weak on the selected bus channel.</p>
<div id="{html.escape(mag_wrap_id)}">{_fig_to_div(mag_fig)}</div>
<div id="{html.escape(phase_wrap_id)}">{_fig_to_div(phase_fig)}</div>
<script type="application/json" id="{html.escape(data_id)}">{payload_json}</script>
<script type="application/json" id="{html.escape(modal_data_id)}">{modal_json}</script>
<script>
(function () {{
  const select = document.getElementById({select_id!r});
  const modeSelect = document.getElementById({mode_select_id!r});
  const dataEl = document.getElementById({data_id!r});
  const modalEl = document.getElementById({modal_data_id!r});
  const titleEl = document.getElementById({title_id!r});
  const noteEl = document.getElementById({note_id!r});
  const magWrap = document.getElementById({mag_wrap_id!r});
  const phaseWrap = document.getElementById({phase_wrap_id!r});
  if (!select || !dataEl) return;
  const payload = JSON.parse(dataEl.textContent);
  const modalPayload = modalEl ? JSON.parse(modalEl.textContent) : {{modes: []}};
  const byId = Object.fromEntries(payload.channels.map((ch) => [ch.id, ch]));
  const modesById = Object.fromEntries((modalPayload.modes || []).map((m) => [m.id, m]));
  const freq = payload.freq;

  function plotDiv(wrap) {{
    return wrap ? wrap.querySelector(".plotly-graph-div") : null;
  }}

  function updateAnnotations(gd, labels) {{
    if (!gd || !gd.layout || !gd.layout.annotations) return labels.map((lab) => ({{text: lab}}));
    return gd.layout.annotations.map((ann, idx) => {{
      const next = Object.assign({{}}, ann);
      if (idx < labels.length) next.text = labels[idx];
      return next;
    }});
  }}

  function modalContribution(mode, outIdx, inIdx) {{
    const nOutFull = mode.residue_re.length;
    const nInFull = nOutFull ? mode.residue_re[0].length : 0;
    const lam = {{re: mode.lambda_re, im: mode.lambda_im}};
    const curves = [];
    let blockRmax = 0;
    for (let r = 0; r < outIdx.length; r++) {{
      for (let c = 0; c < inIdx.length; c++) {{
        const rr = outIdx[r];
        const cc = inIdx[c];
        const reR = (rr < nOutFull && cc < nInFull) ? mode.residue_re[rr][cc] : 0;
        const imR = (rr < nOutFull && cc < nInFull) ? mode.residue_im[rr][cc] : 0;
        blockRmax = Math.max(blockRmax, Math.hypot(reR, imR));
        const y = freq.map((f) => {{
          const wi = 2 * Math.PI * f;
          const dr = 0 - lam.re;
          const di = wi - lam.im;
          const den = dr * dr + di * di;
          if (den < 1e-30) return 0;
          return Math.hypot(reR, imR) / Math.sqrt(den);
        }});
        curves.push(y);
      }}
    }}
    const sigma = Math.abs(lam.re);
    return {{curves: curves, blockRmax: blockRmax, blockVis: blockRmax / Math.max(sigma, 1e-12)}};
  }}

  function overlayTraceIndices(gd, nWanted) {{
    if (!gd || !gd.data) return [];
    const idxs = [];
    for (let i = 0; i < gd.data.length; i++) {{
      const nm = (gd.data[i].name || "");
      if (nm.startsWith("modal")) idxs.push(i);
    }}
    if (idxs.length >= nWanted) return idxs.slice(0, nWanted);
    // Fallback: assume overlays were appended after the nWanted channel traces.
    return Array.from({{length: nWanted}}, (_, i) => nWanted + i);
  }}

  function applyOverlay(ch) {{
    const magGd = plotDiv(magWrap);
    if (!magGd || !window.Plotly || !modeSelect) return;
    const nTrace = ch.element_labels.length;
    const modeId = modeSelect.value;
    const mode = modeId ? modesById[modeId] : null;
    const overlayIdx = overlayTraceIndices(magGd, nTrace);
    if (!mode) {{
      Plotly.restyle(magGd, {{
        y: Array.from({{length: nTrace}}, () => freq.map(() => null)),
        visible: Array.from({{length: nTrace}}, () => false),
      }}, overlayIdx);
      if (noteEl) {{
        noteEl.textContent = "Select a mode to overlay |R/(j2πf−λ)| on the magnitude plots. Visibility uses max|R|/|σ| over ports (σ in rad/s). A mode can look strong globally yet be weak on the selected bus channel.";
      }}
      return;
    }}
    const contrib = modalContribution(mode, ch.out_indices, ch.in_indices);
    // Restyle each overlay trace explicitly (subplot-safe).
    for (let i = 0; i < nTrace; i++) {{
      const idx = overlayIdx[i];
      if (idx === undefined) continue;
      Plotly.restyle(
        magGd,
        {{
          x: [freq],
          y: [contrib.curves[i]],
          name: ["modal " + ch.element_labels[i]],
          visible: [true],
          "line.dash": ["dash"],
          "line.color": ["#c51b8a"],
          hovertemplate: [
            "modal " + ch.element_labels[i] +
            "<br>f=%{{x:.4g}} Hz<br>|R/(j2πf−λ)|=%{{y:.4g}}<extra></extra>"
          ],
        }},
        [idx]
      );
    }}
    if (noteEl) {{
      const fHz = Math.abs(mode.lambda_im) / (2 * Math.PI);
      noteEl.textContent = "Modal overlay: " + mode.label +
        ". Channel-block max|R|/|σ|=" + Number(contrib.blockVis).toPrecision(4) +
        " (global max|R|/|σ|=" + Number(mode.visibility).toPrecision(4) +
        "). |R/(j2πf−λ)| is the single-mode contribution on this bus pair. " +
        (contrib.blockVis < 0.05 * Math.max(mode.visibility, 1e-12)
          ? "This mode is weak on the selected channel — try another bus pair or another mode. "
          : "") +
        "Expected peak near f≈" + fHz.toPrecision(4) + " Hz if visible.";
    }}
  }}

  function applyChannel(id) {{
    const ch = byId[id];
    if (!ch) return;
    if (titleEl) titleEl.textContent = ch.label + " (" + ch.n_out + "\\u00d7" + ch.n_in + ")";
    const nTrace = ch.element_labels.length;
    const indices = Array.from({{length: nTrace}}, (_, i) => i);
    const magGd = plotDiv(magWrap);
    const phaseGd = plotDiv(phaseWrap);
    if (magGd && window.Plotly) {{
      Plotly.update(
        magGd,
        {{
          x: Array.from({{length: nTrace}}, () => freq),
          y: ch.mag,
          name: ch.element_labels,
          hovertemplate: ch.element_labels.map(
            (lab) => lab + "<br>f=%{{x:.4g}} Hz<br>|G|=%{{y:.4g}} p.u.<extra></extra>"
          ),
        }},
        {{
          title: {{text: {name!r} + " magnitude — " + ch.label}},
          annotations: updateAnnotations(magGd, ch.element_labels),
        }},
        indices
      );
    }}
    if (phaseGd && window.Plotly) {{
      Plotly.update(
        phaseGd,
        {{
          x: Array.from({{length: nTrace}}, () => freq),
          y: ch.phase,
          name: ch.element_labels,
          hovertemplate: ch.element_labels.map(
            (lab) => lab + "<br>f=%{{x:.4g}} Hz<br>\\u2220=%{{y:.2f}}\\u00b0<extra></extra>"
          ),
        }},
        {{
          title: {{text: {name!r} + " phase — " + ch.label}},
          annotations: updateAnnotations(phaseGd, ch.element_labels),
        }},
        indices
      );
    }}
    applyOverlay(ch);
  }}

  select.addEventListener("change", () => applyChannel(select.value));
  if (modeSelect) modeSelect.addEventListener("change", () => applyChannel(select.value));
  function resizeChannelPlots() {{
    [magWrap, phaseWrap].forEach((wrap) => {{
      const gd = plotDiv(wrap);
      if (gd && window.Plotly) {{
        try {{ Plotly.Plots.resize(gd); }} catch (err) {{}}
      }}
    }});
  }}
  requestAnimationFrame(() => {{
    applyChannel(select.value);
    resizeChannelPlots();
  }});
  window.addEventListener("resize", resizeChannelPlots);
}})();
</script>
"""


def _transfer_block_grid_figure(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    out_indices: Sequence[int],
    in_indices: Sequence[int],
    out_labels: Sequence[str],
    in_labels: Sequence[str],
    title: str,
    quantity: str,
    y_axis_title: str,
    go,
    make_subplots,
    include_modal_overlay_traces: bool = False,
):
    """Magnitude or phase Bode grid for one MIMO block (typically 2×2 dq)."""

    vals = np.asarray(values, dtype=complex)
    freq = np.asarray(frequencies_hz, dtype=float).ravel()
    n_out = max(1, len(out_indices))
    n_in = max(1, len(in_indices))
    empty = vals.ndim != 3 or vals.size == 0 or freq.size == 0 or not out_indices or not in_indices

    titles = []
    for r in out_indices:
        for c in in_indices:
            titles.append(f"{out_labels[r]}/{in_labels[c]}")
    if empty:
        titles = ["(empty)"]

    fig = make_subplots(
        rows=n_out,
        cols=n_in,
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    for rr, r in enumerate(out_indices if not empty else [0]):
        for cc, c in enumerate(in_indices if not empty else [0]):
            label = titles[rr * n_in + cc] if titles else f"({r},{c})"
            if empty:
                y = np.array([])
            else:
                channel = vals[:, r, c]
                if quantity == "phase":
                    y = np.unwrap(np.angle(channel)) * 180.0 / np.pi
                else:
                    y = np.abs(channel)
            if quantity == "phase":
                hover = f"{label}<br>f=%{{x:.4g}} Hz<br>∠=%{{y:.2f}}°<extra></extra>"
            else:
                hover = f"{label}<br>f=%{{x:.4g}} Hz<br>|G|=%{{y:.4g}} p.u.<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=y,
                    mode="lines",
                    name=label,
                    showlegend=False,
                    hovertemplate=hover,
                ),
                row=rr + 1,
                col=cc + 1,
            )
            if rr == n_out - 1:
                fig.update_xaxes(title_text="Frequency (Hz)", row=rr + 1, col=cc + 1)
            if cc == 0:
                fig.update_yaxes(title_text=y_axis_title, row=rr + 1, col=cc + 1)

    if include_modal_overlay_traces and not empty:
        # Placeholder traces updated by the channel/mode dropdown JS.
        for rr, _r in enumerate(out_indices):
            for cc, _c in enumerate(in_indices):
                label = titles[rr * n_in + cc]
                fig.add_trace(
                    go.Scatter(
                        x=freq,
                        y=[None] * len(freq),
                        mode="lines",
                        name=f"modal {label}",
                        line=dict(dash="dash", width=1.5, color="#c51b8a"),
                        showlegend=False,
                        visible=False,
                        hovertemplate=(
                            f"modal {label}<br>f=%{{x:.4g}} Hz<br>|R/(j2πf−λ)|=%{{y:.4g}}<extra></extra>"
                        ),
                    ),
                    row=rr + 1,
                    col=cc + 1,
                )

    fig.update_layout(
        title_text=title,
        height=max(420, 220 * n_out),
        autosize=True,
        showlegend=False,
        margin=dict(t=60, b=40, l=60, r=20),
        hovermode="x unified",
    )
    # Omit fixed layout.width so the figure fills the card (responsive config).
    fig.layout.width = None
    return fig


def _mimo_dq_blocks(output_labels: Sequence[str], input_labels: Sequence[str]) -> list[dict[str, Any]]:
    """Build selectable bus-to-bus blocks (2×2 dq when both sides are dq ports)."""

    out_groups = _port_axis_groups(output_labels)
    in_groups = _port_axis_groups(input_labels)
    blocks: list[dict[str, Any]] = []
    # Prefer self blocks first, then off-diagonal.
    ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for og in out_groups:
        for ig in in_groups:
            if og["key"] == ig["key"]:
                ordered.append((og, ig))
    for og in out_groups:
        for ig in in_groups:
            if og["key"] != ig["key"]:
                ordered.append((og, ig))
    for og, ig in ordered:
        out_idx = list(og["indices"])
        in_idx = list(ig["indices"])
        # Keep square leading block (2×2 dq or 1×1 dc).
        size = min(len(out_idx), len(in_idx), 2)
        if size == 0:
            continue
        out_idx = out_idx[:size]
        in_idx = in_idx[:size]
        block_id = f"{og['key']}_from_{ig['key']}"
        if og["kind"] == "bus" and ig["kind"] == "bus":
            label = f"Bus {og['bus']} ← Bus {ig['bus']}"
        else:
            label = f"{og['title']} ← {ig['title']}"
        blocks.append(
            {
                "id": block_id,
                "label": label,
                "out_indices": out_idx,
                "in_indices": in_idx,
            }
        )
    return blocks


def _port_axis_groups(labels: Sequence[str]) -> list[dict[str, Any]]:
    """Group consecutive dq ports by bus number; fall back to singleton ports."""

    groups: list[dict[str, Any]] = []
    i = 0
    while i < len(labels):
        lab = str(labels[i])
        m = re.fullmatch(r"([vi])_d(\d+)", lab, flags=re.IGNORECASE)
        if m and i + 1 < len(labels):
            nxt = str(labels[i + 1])
            m2 = re.fullmatch(rf"{re.escape(m.group(1))}_q{m.group(2)}", nxt, flags=re.IGNORECASE)
            if m2:
                bus = int(m.group(2))
                groups.append(
                    {
                        "key": f"bus{bus}",
                        "kind": "bus",
                        "bus": bus,
                        "title": f"Bus {bus}",
                        "indices": (i, i + 1),
                    }
                )
                i += 2
                continue
        m_dc = re.fullmatch(r"([vi])(\d+)", lab, flags=re.IGNORECASE)
        if m_dc:
            bus = int(m_dc.group(2))
            groups.append(
                {
                    "key": f"bus{bus}",
                    "kind": "bus",
                    "bus": bus,
                    "title": f"Bus {bus}",
                    "indices": (i,),
                }
            )
            i += 1
            continue
        groups.append(
            {
                "key": f"port{i}",
                "kind": "port",
                "bus": None,
                "title": lab,
                "indices": (i,),
            }
        )
        i += 1
    return groups


def _transfer_axis_labels(labels: Sequence[str] | None, count: int, *, prefix: str) -> list[str]:
    if labels is not None and len(labels) >= count:
        return [str(labels[i]) for i in range(count)]
    return [f"{prefix}{i}" for i in range(count)]


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
      overflow-x: auto;
    }}
    .card [id$="-mag-wrap"],
    .card [id$="-phase-wrap"],
    .card .plotly-graph-div,
    .card .js-plotly-plot {{
      width: 100% !important;
      max-width: 100%;
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
    label[for="mode-select"],
    label[for$="-channel-select"],
    label[for$="-mode-select"] {{ font-weight: 600; }}
    select[id$="-channel-select"],
    select[id$="-mode-select"] {{
      margin: 0.25rem 0 0.75rem 0.5rem;
      padding: 0.35rem 0.55rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.95rem;
      max-width: min(100%, 40rem);
    }}
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
