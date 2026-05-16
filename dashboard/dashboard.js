const fileInput = document.getElementById("fileInput");
const casePath = document.getElementById("casePath");
const stats = document.getElementById("stats");
const plot = document.getElementById("plot");
const modesDiv = document.getElementById("modes");
const participationDiv = document.getElementById("participation");
const matrixMeta = document.getElementById("matrixMeta");
const matrixHeatmap = document.getElementById("matrixHeatmap");
const tooltip = document.getElementById("tooltip");
const zoomIn = document.getElementById("zoomIn");
const zoomOut = document.getElementById("zoomOut");
const resetZoom = document.getElementById("resetZoom");
const networkGraph = document.getElementById("networkGraph");
const greyboxSummary = document.getElementById("greyboxSummary");
const admittanceChannel = document.getElementById("admittanceChannel");
const admittanceMeta = document.getElementById("admittanceMeta");
const admittancePlot = document.getElementById("admittancePlot");
const admittanceZoomIn = document.getElementById("admittanceZoomIn");
const admittanceZoomOut = document.getElementById("admittanceZoomOut");
const admittanceResetZoom = document.getElementById("admittanceResetZoom");
const greyboxLayers = document.getElementById("greyboxLayers");

let currentData = null;
let selectedMode = 0;
let plotView = null;
let dragging = null;
let admittanceView = null;
let admittanceDragging = null;

fileInput.addEventListener("click", () => {
  fileInput.value = "";
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  resetDashboardForLoad();
  try {
    const text = await file.text();
    currentData = JSON.parse(text);
    render(currentData);
  } catch (error) {
    resetDashboardForLoad();
    casePath.textContent = `Could not load ${file.name}: ${error.message}`;
    stats.innerHTML = `<div class="panel stat"><div class="value danger">Load failed</div><div class="label">The result file is not valid dashboard JSON.</div></div>`;
  }
});

zoomIn.addEventListener("click", () => zoomPlot(0.75));
zoomOut.addEventListener("click", () => zoomPlot(1.35));
resetZoom.addEventListener("click", () => {
  plotView = null;
  if (currentData) renderPlot(currentData);
});

admittanceZoomIn.addEventListener("click", () => zoomAdmittance(0.75));
admittanceZoomOut.addEventListener("click", () => zoomAdmittance(1.35));
admittanceResetZoom.addEventListener("click", () => {
  admittanceView = null;
  if (currentData?.analysis_type === "greybox") {
    renderAdmittancePlot(currentGreyboxTransfer());
  }
});

plot.addEventListener("wheel", (event) => {
  if (!currentData || !plotView) return;
  event.preventDefault();
  zoomPlot(event.deltaY < 0 ? 0.85 : 1.18);
});

plot.addEventListener("mousedown", (event) => {
  if (!plotView) return;
  dragging = { x: event.clientX, y: event.clientY, view: { ...plotView } };
});

admittancePlot.addEventListener("wheel", (event) => {
  if (currentData?.analysis_type !== "greybox" || !admittanceView) return;
  event.preventDefault();
  zoomAdmittance(event.deltaY < 0 ? 0.85 : 1.18);
});

admittancePlot.addEventListener("mousedown", (event) => {
  if (!admittanceView) return;
  admittanceDragging = { x: event.clientX, view: { ...admittanceView } };
});

window.addEventListener("mousemove", (event) => {
  if (!dragging || !plotView) return;
  const width = plot.clientWidth || 700;
  const height = 420;
  const dx = (event.clientX - dragging.x) / width * (dragging.view.maxX - dragging.view.minX);
  const dy = (event.clientY - dragging.y) / height * (dragging.view.maxY - dragging.view.minY);
  plotView = {
    minX: dragging.view.minX - dx,
    maxX: dragging.view.maxX - dx,
    minY: dragging.view.minY + dy,
    maxY: dragging.view.maxY + dy,
  };
  renderPlot(currentData);
});

window.addEventListener("mousemove", (event) => {
  if (!admittanceDragging || !admittanceView || currentData?.analysis_type !== "greybox") return;
  const width = admittancePlot.clientWidth || 900;
  const dx = (event.clientX - admittanceDragging.x) / width * (admittanceDragging.view.maxLogF - admittanceDragging.view.minLogF);
  admittanceView = {
    ...admittanceView,
    minLogF: admittanceDragging.view.minLogF - dx,
    maxLogF: admittanceDragging.view.maxLogF - dx,
  };
  renderAdmittancePlot(currentGreyboxTransfer());
});

window.addEventListener("mouseup", () => {
  dragging = null;
  admittanceDragging = null;
});

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return Number(value).toFixed(digits);
}

function fmtCompact(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const number = Number(value);
  if (number !== 0 && Math.abs(number) < 10 ** -digits) return number.toExponential(3);
  if (Math.abs(number) >= 1e5) return number.toExponential(3);
  return number.toFixed(digits);
}

function resetDashboardForLoad() {
  currentData = null;
  selectedMode = 0;
  plotView = null;
  dragging = null;
  admittanceView = null;
  admittanceDragging = null;
  hideTooltip();
  casePath.textContent = "Loading results...";
  stats.innerHTML = "";
  plot.innerHTML = "";
  modesDiv.innerHTML = "";
  participationDiv.innerHTML = "";
  networkGraph.innerHTML = "";
  matrixMeta.textContent = "";
  matrixHeatmap.innerHTML = "";
  resetGreybox();
}

function render(data) {
  if (data.analysis_type === "greybox") {
    renderGreybox(data);
    return;
  }
  casePath.textContent = data.case.path || "Loaded results";
  renderStats(data);
  renderPlot(data);
  renderModes(data);
  renderParticipation(data.modes[selectedMode]);
  renderNetwork(data);
  renderMatrix(data);
  resetGreybox();
}

function currentGreyboxTransfer() {
  return currentData?.whole_system_impedance || currentData?.whole_system_admittance;
}

function renderStats(data) {
  const stableClass = data.stability.stable ? "ok" : "danger";
  stats.innerHTML = [
    stat("Buses", data.case.bus_count),
    stat("States", data.case.state_count),
    stat("Finite Modes", data.stability.finite_mode_count),
    stat("Stable", `<span class="${stableClass}">${data.stability.stable}</span>`),
  ].join("");
}

function stat(label, value) {
  return `<div class="panel stat"><div class="value">${value}</div><div class="label">${label}</div></div>`;
}

function renderPlot(data) {
  const values = data.eigenvalues;
  plot.innerHTML = "";
  const width = plot.clientWidth || 700;
  const height = 420;
  const pad = 44;
  plot.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!values.length) return;
  if (!plotView) {
    const xs = values.map((item) => item.real_hz);
    const ys = values.map((item) => item.imag_hz);
    const minX = Math.min(...xs, -1);
    const maxX = Math.max(...xs, 1);
    const maxAbsY = Math.max(...ys.map(Math.abs), 1);
    const marginX = Math.max((maxX - minX) * 0.12, 0.5);
    plotView = { minX: minX - marginX, maxX: maxX + marginX, minY: -maxAbsY * 1.15, maxY: maxAbsY * 1.15 };
  }
  const { minX, maxX, minY, maxY } = plotView;
  const xScale = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (width - 2 * pad);
  const yScale = (y) => height - pad - ((y - minY) / (maxY - minY || 1)) * (height - 2 * pad);
  addLine(xScale(0), pad, xScale(0), height - pad, "#d0d5dd");
  addLine(pad, yScale(0), width - pad, yScale(0), "#d0d5dd");
  values.forEach((item, index) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", xScale(item.real_hz));
    circle.setAttribute("cy", yScale(item.imag_hz));
    circle.setAttribute("r", index === selectedMode ? 5 : 3);
    circle.setAttribute("fill", item.real_hz > 0 ? "#b42318" : item.imag_hz === 0 ? "#0891b2" : "#4f46e5");
    circle.setAttribute("stroke", index === selectedMode ? "#111827" : "none");
    circle.setAttribute("stroke-width", index === selectedMode ? "1.5" : "0");
    circle.addEventListener("mousemove", (event) => showPointTooltip(event, item, data.modes[index]));
    circle.addEventListener("mouseleave", hideTooltip);
    circle.addEventListener("click", () => {
      selectedMode = index;
      render(data);
    });
    plot.appendChild(circle);
  });
  addText(pad, 22, "Real (Hz)");
  addText(width - 160, height - 12, `View: ${fmt(minX, 2)} to ${fmt(maxX, 2)} Hz`);
}

function zoomPlot(factor) {
  if (!plotView || !currentData) return;
  const cx = (plotView.minX + plotView.maxX) / 2;
  const cy = (plotView.minY + plotView.maxY) / 2;
  const halfX = (plotView.maxX - plotView.minX) * factor / 2;
  const halfY = (plotView.maxY - plotView.minY) * factor / 2;
  plotView = { minX: cx - halfX, maxX: cx + halfX, minY: cy - halfY, maxY: cy + halfY };
  renderPlot(currentData);
}

function showPointTooltip(event, item, mode) {
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.innerHTML = `
    <strong>Mode ${mode?.index ?? item.index}</strong><br>
    Real: ${fmt(item.real_hz)} Hz<br>
    Imag: ${fmt(item.imag_hz)} Hz<br>
    Frequency: ${fmt(item.frequency_hz)} Hz<br>
    Damping: ${fmt(mode?.damping_ratio)}<br>
    Top state: ${mode?.participation?.[0]?.state || "n/a"}<br>
    Component: ${mode?.participation?.[0]?.component || "n/a"}
  `;
}

function hideTooltip() {
  tooltip.style.display = "none";
}

function addLine(x1, y1, x2, y2, stroke) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", stroke);
  plot.appendChild(line);
}

function addText(x, y, textValue) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("fill", "#667085");
  text.setAttribute("font-size", "12");
  text.textContent = textValue;
  plot.appendChild(text);
}

function renderModes(data) {
  const rows = data.modes.map((mode, index) => `
    <tr class="mode-row ${index === selectedMode ? "selected" : ""}" data-index="${index}">
      <td>${mode.index}</td>
      <td>${fmt(mode.eigenvalue_hz.real)}</td>
      <td>${fmt(mode.eigenvalue_hz.imag)}</td>
      <td>${fmt(mode.frequency_hz)}</td>
      <td>${fmt(mode.damping_ratio)}</td>
      <td>${mode.participation[0]?.state || "n/a"}</td>
      <td>${mode.participation[0]?.component || "n/a"}</td>
    </tr>
  `).join("");
  modesDiv.innerHTML = `
    <table>
      <thead><tr><th>Mode</th><th>Real Hz</th><th>Imag Hz</th><th>Frequency Hz</th><th>Damping</th><th>Top State</th><th>Component</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  modesDiv.querySelectorAll("tr.mode-row").forEach((row) => {
    row.addEventListener("click", () => {
      selectedMode = Number(row.dataset.index);
      render(data);
    });
  });
}

function renderParticipation(mode) {
  if (!mode) {
    participationDiv.innerHTML = "<span class=\"muted\">No mode selected.</span>";
    return;
  }
  participationDiv.innerHTML = `
    <p><strong>Mode ${mode.index}</strong></p>
    <p class="muted">Eigenvalue: ${fmt(mode.eigenvalue.real)} ${mode.eigenvalue.imag >= 0 ? "+" : "-"} ${fmt(Math.abs(mode.eigenvalue.imag))}j rad/s</p>
    ${mode.participation.map((item) => `
      <div style="margin: 10px 0;">
        <div style="display:flex; justify-content:space-between; gap:12px;">
          <span>
            ${item.state}<br>
            <span class="muted">${item.component || "Unknown component"}</span>
          </span>
          <span>${fmt(item.factor, 3)}</span>
        </div>
        <div class="bar"><span style="width:${Math.max(0, Math.min(100, item.factor * 100))}%"></span></div>
      </div>
    `).join("")}
  `;
}

function renderMatrix(data) {
  const matrix = data.matrices?.A;
  if (!matrix) {
    matrixMeta.textContent = "No A matrix data in this result file.";
    matrixHeatmap.innerHTML = "";
    return;
  }
  const rows = matrix.values.length;
  const cols = matrix.values[0]?.length || 0;
  matrixMeta.textContent = `${matrix.name}: ${matrix.rows} x ${matrix.cols}${matrix.sampled ? `, sampled to ${rows} x ${cols}` : ""}. Color intensity is normalized by max |Aij| = ${fmt(matrix.max_abs, 3)}.`;
  matrixHeatmap.style.gridTemplateColumns = `repeat(${cols}, minmax(5px, 1fr))`;
  const maxAbs = matrix.max_abs || 1;
  matrixHeatmap.innerHTML = matrix.values.flatMap((row, r) => row.map((value, c) => {
    const intensity = Math.min(1, Math.abs(value) / maxAbs);
    const alpha = 0.08 + 0.85 * intensity;
    const color = value >= 0 ? `rgba(79, 70, 229, ${alpha})` : `rgba(180, 35, 24, ${alpha})`;
    const rowLabel = matrix.row_labels[r] || `row ${r}`;
    const colLabel = matrix.col_labels[c] || `col ${c}`;
    const rowComponent = matrix.row_components?.[r] || "Unknown";
    const colComponent = matrix.col_components?.[c] || "Unknown";
    return `<div class="matrix-cell" title="${rowLabel} (${rowComponent}) -> ${colLabel} (${colComponent}): ${fmt(value, 6)}" style="background:${color};"></div>`;
  })).join("");
}

function renderNetwork(data) {
  const graph = data.network_graph;
  networkGraph.innerHTML = "";
  const width = networkGraph.clientWidth || 900;
  const height = 520;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(120, Math.min(width, height) * 0.36);
  networkGraph.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!graph || !graph.nodes?.length) {
    addNetworkText(cx - 70, cy, "No network graph data");
    return;
  }

  const positions = {};
  graph.nodes.forEach((node, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / graph.nodes.length;
    positions[node.id] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });

  graph.edges.forEach((edge) => {
    const a = positions[edge.from];
    const b = positions[edge.to];
    if (!a || !b) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    line.setAttribute("stroke", edge.ac_dc === "AC" ? "#98a2b3" : "#0891b2");
    line.setAttribute("stroke-width", "1.6");
    line.addEventListener("mousemove", (event) => {
      tooltip.style.display = "block";
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
      tooltip.innerHTML = `
        <strong>Branch ${edge.from}-${edge.to}</strong><br>
        Type: ${edge.ac_dc}<br>
        R: ${fmt(edge.r, 5)}<br>
        X/wL: ${fmt(edge.x, 5)}<br>
        Tap: ${fmt(edge.tap, 3)}
      `;
    });
    line.addEventListener("mouseleave", hideTooltip);
    networkGraph.appendChild(line);
  });

  graph.nodes.forEach((node) => {
    const pos = positions[node.id];
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", pos.x);
    circle.setAttribute("cy", pos.y);
    circle.setAttribute("r", "15");
    circle.setAttribute("fill", node.ac_dc === "AC" ? "#4f46e5" : "#0891b2");
    circle.setAttribute("stroke", "#ffffff");
    circle.setAttribute("stroke-width", "2");
    circle.addEventListener("mousemove", (event) => showBusTooltip(event, node));
    circle.addEventListener("mouseleave", hideTooltip);
    group.appendChild(circle);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", pos.x);
    label.setAttribute("y", pos.y + 4);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", "#ffffff");
    label.setAttribute("font-size", "11");
    label.setAttribute("font-weight", "700");
    label.textContent = node.id;
    group.appendChild(label);

    const outerLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    outerLabel.setAttribute("x", pos.x);
    outerLabel.setAttribute("y", pos.y + 31);
    outerLabel.setAttribute("text-anchor", "middle");
    outerLabel.setAttribute("fill", "#475467");
    outerLabel.setAttribute("font-size", "11");
    outerLabel.textContent = node.apparatus?.[0] ? `${node.apparatus[0].name} T${node.apparatus[0].type}` : "No apparatus";
    group.appendChild(outerLabel);

    node.apparatus?.forEach((apparatus, index) => {
      const badge = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      badge.setAttribute("x", pos.x + 18);
      badge.setAttribute("y", pos.y - 15 + index * 11);
      badge.setAttribute("width", "8");
      badge.setAttribute("height", "8");
      badge.setAttribute("rx", "2");
      badge.setAttribute("fill", "#7c3aed");
      badge.addEventListener("mousemove", (event) => showApparatusTooltip(event, apparatus, node));
      badge.addEventListener("mouseleave", hideTooltip);
      group.appendChild(badge);
    });

    node.shunts?.forEach((shunt, index) => {
      const shuntY = pos.y + 18 + index * 11;
      const stem = document.createElementNS("http://www.w3.org/2000/svg", "line");
      stem.setAttribute("x1", pos.x - 18);
      stem.setAttribute("y1", pos.y + 8);
      stem.setAttribute("x2", pos.x - 18);
      stem.setAttribute("y2", shuntY);
      stem.setAttribute("stroke", "#b42318");
      stem.setAttribute("stroke-width", "1.4");
      group.appendChild(stem);

      const marker = document.createElementNS("http://www.w3.org/2000/svg", "path");
      marker.setAttribute("d", `M ${pos.x - 26} ${shuntY} L ${pos.x - 10} ${shuntY} M ${pos.x - 23} ${shuntY + 4} L ${pos.x - 13} ${shuntY + 4}`);
      marker.setAttribute("stroke", "#b42318");
      marker.setAttribute("stroke-width", "1.5");
      marker.setAttribute("fill", "none");
      marker.addEventListener("mousemove", (event) => showShuntTooltip(event, shunt));
      marker.addEventListener("mouseleave", hideTooltip);
      group.appendChild(marker);
    });

    networkGraph.appendChild(group);
  });
}

function showBusTooltip(event, node) {
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.innerHTML = `
    <strong>${node.label}</strong><br>
    Area: ${node.area}<br>
    Type: ${node.ac_dc}, bus type ${node.bus_type}<br>
    Voltage: ${fmt(node.voltage, 4)} p.u.<br>
    Apparatus: ${node.apparatus?.map((item) => `${item.name} (type ${item.type})`).join("<br>") || "None"}<br>
    Shunts/loads: ${node.shunts?.length || 0}
  `;
}

function showApparatusTooltip(event, apparatus, node) {
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.innerHTML = `
    <strong>${apparatus.name}</strong><br>
    Type: ${apparatus.type}<br>
    Connected bus node: ${node.id}<br>
    Apparatus buses: ${apparatus.buses.join("-")}
  `;
}

function showShuntTooltip(event, shunt) {
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.innerHTML = `
    <strong>Bus ${shunt.bus} shunt/load</strong><br>
    Source: ${shunt.source}<br>
    Type: ${shunt.ac_dc}<br>
    R: ${fmt(shunt.r, 5)}<br>
    X/wL: ${fmt(shunt.x, 5)}<br>
    B/wC: ${fmt(shunt.b, 5)}<br>
    G: ${fmt(shunt.g, 5)}
  `;
}

function addNetworkText(x, y, value) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("fill", "#667085");
  text.textContent = value;
  networkGraph.appendChild(text);
}

function resetGreybox() {
  greyboxSummary.textContent = "No greybox data in this result file.";
  admittanceMeta.textContent = "";
  admittanceChannel.innerHTML = "";
  admittancePlot.innerHTML = "";
  admittanceView = null;
  admittanceDragging = null;
  greyboxLayers.innerHTML = "";
}

function renderGreybox(data) {
  casePath.textContent = data.case?.path || "Loaded greybox results";
  const admittance = data.whole_system_impedance || data.whole_system_admittance;
  const transferName = data.whole_system_impedance ? "impedance" : "admittance";
  stats.innerHTML = [
    stat("Buses", data.case?.bus_count ?? "n/a"),
    stat("Apparatus", data.case?.apparatus_count ?? "n/a"),
    stat("Frequencies", admittance?.frequencies_hz?.length ?? 0),
    stat("Layers", data.config?.layers?.join(", ") || "admittance-only"),
  ].join("");
  plot.innerHTML = "";
  modesDiv.innerHTML = "<span class=\"muted\">State-space modal results are not part of greybox output.</span>";
  participationDiv.innerHTML = "<span class=\"muted\">Greybox results are shown in the Greybox Analysis section.</span>";
  networkGraph.innerHTML = "";
  addNetworkText(320, 260, "Network diagram is available in modal result files.");
  matrixMeta.textContent = "System A matrix is available in modal result files.";
  matrixHeatmap.innerHTML = "";
  if (!admittance) {
    resetGreybox();
    return;
  }
  greyboxSummary.textContent = `Greybox layers: ${data.config?.layers?.join(", ") || "admittance-only"}. Showing whole-system ${transferName}.`;
  populateAdmittanceChannels(admittance);
  renderAdmittancePlot(admittance);
  renderGreyboxLayers(data);
}

function populateAdmittanceChannels(admittance) {
  const channels = admittance.channels || [];
  const outputs = admittance.outputs || [];
  const inputs = admittance.inputs || [];
  const options = channels.length
    ? channels.map((channel) => `<option value="${channel.row},${channel.col}">${channel.label}</option>`)
    : outputs.flatMap((output, row) => inputs.map((input, col) => `<option value="${row},${col}">${output} / ${input}</option>`));
  admittanceChannel.innerHTML = options.join("");
  admittanceChannel.onchange = () => {
    admittanceView = null;
    renderAdmittancePlot(admittance);
  };
}

function renderAdmittancePlot(admittance) {
  const frequencies = admittance.frequencies_hz || [];
  const values = admittance.values || [];
  admittancePlot.innerHTML = "";
  admittancePlot.setAttribute("viewBox", "0 0 900 420");
  if (!frequencies.length || !values.length) {
    admittanceMeta.textContent = "No whole-system transfer samples are available.";
    return;
  }
  const [row, col] = (admittanceChannel.value || "0,0").split(",").map(Number);
  const magnitudes = values.map((matrix) => {
    const item = matrix?.[row]?.[col] || { real: 0, imag: 0 };
    return Math.hypot(item.real || 0, item.imag || 0);
  });
  const width = admittancePlot.clientWidth || 900;
  const height = 420;
  const pad = 78;
  const minF = Math.min(...frequencies.filter((value) => value > 0));
  const maxF = Math.max(...frequencies);
  const maxMag = Math.max(...magnitudes, 1e-12);
  if (!admittanceView) {
    admittanceView = { minLogF: Math.log10(minF), maxLogF: Math.log10(maxF), maxMag };
  }
  const { minLogF, maxLogF } = admittanceView;
  const viewMaxMag = admittanceView.maxMag || maxMag;
  const xScale = (f) => pad + ((Math.log10(Math.max(f, minF)) - minLogF) / (maxLogF - minLogF || 1)) * (width - 2 * pad);
  const yScale = (m) => height - pad - (m / viewMaxMag) * (height - 2 * pad);
  addAdmittanceLine(pad, height - pad, width - pad, height - pad, "#d0d5dd");
  addAdmittanceLine(pad, pad, pad, height - pad, "#d0d5dd");
  addAdmittanceTicks(pad, width, height, viewMaxMag, yScale);
  addFrequencyTicks(pad, width, height, minLogF, maxLogF, xScale);
  const visible = frequencies.map((freq, index) => ({ freq, mag: magnitudes[index], value: values[index]?.[row]?.[col] || { real: 0, imag: 0 } }))
    .filter((item) => {
      const logF = Math.log10(Math.max(item.freq, minF));
      return logF >= minLogF && logF <= maxLogF;
    });
  const points = visible.map((item) => `${xScale(item.freq)},${yScale(item.mag)}`).join(" ");
  const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  polyline.setAttribute("points", points);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#4f46e5");
  polyline.setAttribute("stroke-width", "2");
  admittancePlot.appendChild(polyline);
  visible.forEach((item) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", xScale(item.freq));
    circle.setAttribute("cy", yScale(item.mag));
    circle.setAttribute("r", "3");
    circle.setAttribute("fill", "#0891b2");
    circle.addEventListener("mousemove", (event) => showAdmittanceTooltip(event, admittance, row, col, item));
    circle.addEventListener("mouseleave", hideTooltip);
    admittancePlot.appendChild(circle);
  });
  addAdmittanceText(pad, 24, currentData?.whole_system_impedance ? "|Zsys(jw)|" : "|Ysys(jw)|");
  addAdmittanceText(12, pad - 18, "Magnitude");
  addAdmittanceText(width / 2 - 42, height - 12, "Frequency (Hz)");
  addAdmittanceText(width - 220, height - 14, `View: ${fmt(10 ** minLogF, 2)} to ${fmt(10 ** maxLogF, 2)} Hz`);
  const channel = channelDetails(admittance, row, col);
  const quantity = currentData?.whole_system_impedance ? "impedance" : "admittance";
  admittanceMeta.textContent = `${channel.output} (${channel.outputComponent}) / ${channel.input} (${channel.inputComponent}), max ${quantity} magnitude ${fmtCompact(maxMag, 6)}.`;
}

function zoomAdmittance(factor) {
  if (!admittanceView || currentData?.analysis_type !== "greybox") return;
  const center = (admittanceView.minLogF + admittanceView.maxLogF) / 2;
  const half = (admittanceView.maxLogF - admittanceView.minLogF) * factor / 2;
  admittanceView = { ...admittanceView, minLogF: center - half, maxLogF: center + half, maxMag: admittanceView.maxMag * factor };
  renderAdmittancePlot(currentGreyboxTransfer());
}

function channelDetails(admittance, row, col) {
  const channel = admittance.channels?.find((item) => item.row === row && item.col === col);
  return {
    output: channel?.output || admittance.outputs?.[row] || `output ${row}`,
    input: channel?.input || admittance.inputs?.[col] || `input ${col}`,
    outputComponent: channel?.output_component || admittance.output_components?.[row] || "Unknown component",
    inputComponent: channel?.input_component || admittance.input_components?.[col] || "Unknown component",
  };
}

function showAdmittanceTooltip(event, admittance, row, col, item) {
  const channel = channelDetails(admittance, row, col);
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.innerHTML = `
    <strong>${channel.output} / ${channel.input}</strong><br>
    Output: ${channel.outputComponent}<br>
    Input: ${channel.inputComponent}<br>
    Frequency: ${fmtCompact(item.freq, 4)} Hz<br>
    Magnitude: ${fmtCompact(item.mag, 6)}<br>
    Real: ${fmtCompact(item.value.real, 6)}<br>
    Imag: ${fmtCompact(item.value.imag, 6)}
  `;
}

function addAdmittanceTicks(pad, width, height, maxMag, yScale) {
  const tickCount = 4;
  for (let idx = 0; idx <= tickCount; idx += 1) {
    const value = maxMag * idx / tickCount;
    const y = yScale(value);
    addAdmittanceLine(pad, y, width - pad, y, idx === 0 ? "#d0d5dd" : "#edf0f5");
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", pad - 12);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("fill", "#667085");
    label.setAttribute("font-size", "12");
    label.textContent = fmtCompact(value, 3);
    admittancePlot.appendChild(label);
  }
}

function addFrequencyTicks(pad, width, height, minLogF, maxLogF, xScale) {
  const start = Math.ceil(minLogF);
  const stop = Math.floor(maxLogF);
  const ticks = [];
  for (let exponent = start; exponent <= stop; exponent += 1) {
    ticks.push(10 ** exponent);
  }
  if (!ticks.length) {
    const min = 10 ** minLogF;
    const max = 10 ** maxLogF;
    ticks.push(min, (min + max) / 2, max);
  }
  ticks.forEach((value) => {
    const x = xScale(value);
    addAdmittanceLine(x, height - pad, x, height - pad + 6, "#98a2b3");
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x);
    label.setAttribute("y", height - pad + 22);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", "#667085");
    label.setAttribute("font-size", "12");
    label.textContent = fmtCompact(value, 3);
    admittancePlot.appendChild(label);
  });
}

function renderGreyboxLayers(data) {
  const modeSections = (data.modes || []).map((mode) => `
    <div class="panel" style="margin-top:12px;">
      <h2>Greybox Mode ${mode.index}</h2>
      ${renderLayerTable("Apparatus Layer 1", mode.layer1, ["label", "value", "normalized"])}
      ${renderLayerTable("Apparatus Layer 2", mode.layer2, ["label", "real", "imag", "real_normalized", "imag_normalized"])}
      ${renderLayerTable("Apparatus Layer 3", mode.layer3, ["label", "parameter", "d_lambda_hz", "d_lambda_pu_hz"])}
    </div>
  `).join("");
  const sensitivitySections = (data.sensitivity || []).map((item) => `
    <div class="panel" style="margin-top:12px;">
      <h2>Sensitivity Mode ${item.mode_index}</h2>
      ${renderLayerTable("Sensitivity Layer 1/2", item.layer12, ["component", "kind", "layer1_normalized", "layer2_real_normalized", "layer2_imag_normalized"])}
      ${renderLayerTable("Sensitivity Layer 3", item.layer3, ["component", "parameter", "d_lambda_rad", "d_lambda_pu_hz"])}
    </div>
  `).join("");
  greyboxLayers.innerHTML = modeSections || sensitivitySections ? modeSections + sensitivitySections : "<p class=\"muted\">No optional greybox layer results were included.</p>";
}

function renderLayerTable(title, rows, columns) {
  if (!rows?.length) return `<p class="muted">${title}: not included.</p>`;
  return `
    <h2>${title}</h2>
    <table>
      <thead><tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `<tr>${columns.map((column) => `<td>${formatCell(row[column])}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table>
  `;
}

function formatCell(value) {
  if (value && typeof value === "object" && "real" in value && "imag" in value) {
    return `${fmt(value.real, 4)} ${value.imag >= 0 ? "+" : "-"} ${fmt(Math.abs(value.imag), 4)}j`;
  }
  if (typeof value === "number") return fmt(value, 5);
  return value ?? "n/a";
}

function addAdmittanceLine(x1, y1, x2, y2, stroke) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", stroke);
  admittancePlot.appendChild(line);
}

function addAdmittanceText(x, y, textValue) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("fill", "#667085");
  text.setAttribute("font-size", "12");
  text.textContent = textValue;
  admittancePlot.appendChild(text);
}
