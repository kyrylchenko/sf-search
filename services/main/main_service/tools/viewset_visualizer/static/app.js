const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const select = document.getElementById("viewsetSelect");
const summary = document.getElementById("summary");
const viewMatrixEl = document.getElementById("viewMatrix");
const selectedViewEl = document.getElementById("selectedView");
const showAllBtn = document.getElementById("showAll");
const hideAllBtn = document.getElementById("hideAll");
const openModeInput = document.getElementById("openMode");

let state = null;
let panoImage = null;
let currentViewset = null;
let currentViewsetName = null;
let visibleViewIds = new Set();
let selectedViewId = null;
let hoveredViewId = null;

async function loadState() {
  const response = await fetch("/api/state");
  state = await response.json();
  panoImage = new Image();
  panoImage.onload = () => {
    canvas.width = state.pano.width;
    canvas.height = state.pano.height;
    populateViewsets();
    draw();
  };
  panoImage.src = state.pano.url;
}

function populateViewsets() {
  select.innerHTML = "";
  state.viewsets.forEach((viewset, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${viewset.name} (${viewset.views.length})`;
    select.appendChild(option);
  });
  select.addEventListener("change", () => {
    currentViewsetName = null;
    draw();
  });
  showAllBtn.addEventListener("click", () => {
    if (!currentViewset) return;
    visibleViewIds = new Set(currentViewset.views.map((view) => view.id));
    draw();
  });
  hideAllBtn.addEventListener("click", () => {
    visibleViewIds = new Set();
    draw();
  });
  openModeInput.addEventListener("change", () => {
    if (currentViewset) {
      renderSelectedView(currentViewset);
    }
  });
  viewMatrixEl.addEventListener("click", handleMatrixClick, true);
}

function draw() {
  if (!state || !panoImage) return;
  const viewset = state.viewsets[Number(select.value || 0)];
  currentViewset = viewset;
  if (currentViewsetName !== viewset.name) {
    currentViewsetName = viewset.name;
    visibleViewIds = defaultVisibleViews(viewset);
    selectedViewId = viewset.views[0]?.id || null;
  }
  drawCanvas(viewset);
  renderSidebar(viewset);
}

function drawCanvas(viewset) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(panoImage, 0, 0, canvas.width, canvas.height);
  drawOverlays(viewset.views.filter((view) => visibleViewIds.has(view.id)));
  const hoveredView = viewset.views.find((view) => view.id === hoveredViewId);
  if (hoveredView) {
    drawHoverOverlay(hoveredView);
  }
}

function defaultVisibleViews(viewset) {
  if (viewset.views.length > 20) {
    return new Set();
  }
  return new Set(viewset.views.map((view) => view.id));
}

function drawOverlays(views) {
  context.lineWidth = Math.max(2, canvas.width / 900);
  views.forEach((view, index) => {
    const hue = (index * 37) % 360;
    context.strokeStyle = `hsla(${hue}, 95%, 65%, 0.9)`;
    view.polygons.forEach((polygon) => drawPolygon(polygon));
  });
}

function drawPolygon(polygon) {
  if (polygon.length === 0) return;
  polygon.forEach((point, index) => {
    const next = polygon[(index + 1) % polygon.length];
    if (Math.abs(next.x - point.x) > 0.5) {
      return;
    }
    context.beginPath();
    context.moveTo(point.x * canvas.width, point.y * canvas.height);
    context.lineTo(next.x * canvas.width, next.y * canvas.height);
    context.stroke();
  });
}

function drawHoverOverlay(view) {
  const previousLineWidth = context.lineWidth;
  const previousStrokeStyle = context.strokeStyle;
  const previousShadowBlur = context.shadowBlur;
  const previousShadowColor = context.shadowColor;
  context.lineWidth = Math.max(3, canvas.width / 650);
  context.strokeStyle = "rgba(250, 204, 21, 0.98)";
  context.shadowBlur = Math.max(4, canvas.width / 900);
  context.shadowColor = "rgba(2, 6, 23, 0.8)";
  view.polygons.forEach((polygon) => drawPolygon(polygon));
  context.lineWidth = previousLineWidth;
  context.strokeStyle = previousStrokeStyle;
  context.shadowBlur = previousShadowBlur;
  context.shadowColor = previousShadowColor;
}

function renderSidebar(viewset) {
  summary.textContent = `${viewset.description || "No description"} · ${state.pano.filename} · ${state.pano.width}x${state.pano.height}`;
  renderMatrix(viewset);
  renderSelectedView(viewset);
}

function renderMatrix(viewset) {
  viewMatrixEl.innerHTML = "";
  const spatialViews = viewset.views.map((view) => ({ view, center: viewCenter(view) }));
  const rows = groupSpatialRows(spatialViews);
  rows.forEach((rowViews) => {
    const row = document.createElement("div");
    row.className = "matrixRow";
    const label = document.createElement("div");
    label.className = "rowLabel";
    label.textContent = `${Math.round(rowViews[0].center.y * 100)}%`;
    const grid = document.createElement("div");
    grid.className = "seatGrid";
    rowViews
      .sort((a, b) => a.center.x - b.center.x)
      .forEach(({ view }) => {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "seat";
        checkbox.dataset.viewId = view.id;
        checkbox.title = `${view.id} · heading ${formatNumber(view.relative_heading)} · pitch ${formatNumber(view.pitch)} · hover previews`;
        checkbox.checked = visibleViewIds.has(view.id);
        checkbox.addEventListener("change", (event) => {
          selectedViewId = view.id;
          if (openModeInput.checked) {
            event.preventDefault();
            checkbox.checked = visibleViewIds.has(view.id);
            renderSelectedView(viewset);
            return;
          }
          if (checkbox.checked) {
            visibleViewIds.add(view.id);
          } else {
            visibleViewIds.delete(view.id);
          }
          draw();
        });
        checkbox.addEventListener("mouseenter", () => {
          previewMatrixView(viewset, view);
        });
        checkbox.addEventListener("focus", () => {
          previewMatrixView(viewset, view);
        });
        checkbox.addEventListener("mouseleave", () => {
          clearMatrixPreview(viewset, view.id);
        });
        checkbox.addEventListener("blur", () => {
          clearMatrixPreview(viewset, view.id);
        });
        grid.appendChild(checkbox);
      });
    row.appendChild(label);
    row.appendChild(grid);
    viewMatrixEl.appendChild(row);
  });
}

function handleMatrixClick(event) {
  const checkbox = event.target.closest?.(".seat");
  if (!checkbox || !openModeInput.checked || !currentViewset) {
    return;
  }
  const view = findCurrentViewById(checkbox.dataset.viewId);
  if (!view) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  selectedViewId = view.id;
  checkbox.checked = visibleViewIds.has(view.id);
  renderSelectedView(currentViewset);
  openViewImage(currentViewset, view);
}

function previewMatrixView(viewset, view) {
  selectedViewId = view.id;
  hoveredViewId = view.id;
  drawCanvas(viewset);
  renderSelectedView(viewset);
}

function clearMatrixPreview(viewset, viewId) {
  if (hoveredViewId !== viewId) {
    return;
  }
  hoveredViewId = null;
  drawCanvas(viewset);
}

function findCurrentViewById(viewId) {
  if (!currentViewset || !viewId) {
    return null;
  }
  return currentViewset.views.find((view) => view.id === viewId) || null;
}

function groupSpatialRows(spatialViews) {
  const sorted = [...spatialViews].sort((a, b) => a.center.y - b.center.y);
  const rows = [];
  sorted.forEach((item) => {
    const existing = rows.find((row) => Math.abs(row[0].center.y - item.center.y) < 0.025);
    if (existing) {
      existing.push(item);
    } else {
      rows.push([item]);
    }
  });
  return rows;
}

function viewCenter(view) {
  let best = null;
  view.polygons.forEach((polygon) => {
    const visiblePoints = polygon.filter((point) => point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1);
    if (visiblePoints.length === 0) {
      return;
    }
    if (best === null || visiblePoints.length > best.length) {
      best = visiblePoints;
    }
  });
  const points = best || view.polygons.flat();
  if (points.length === 0) {
    return { x: 0.5, y: 0.5 };
  }
  return {
    x: clamp01(points.reduce((sum, point) => sum + point.x, 0) / points.length),
    y: clamp01(points.reduce((sum, point) => sum + point.y, 0) / points.length),
  };
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function renderSelectedView(viewset) {
  const view = viewset.views.find((candidate) => candidate.id === selectedViewId) || viewset.views[0];
  if (!view) {
    selectedViewEl.textContent = "";
    return;
  }
  selectedViewEl.innerHTML = `
    <strong>${escapeHtml(view.id)}</strong>
    <span>${escapeHtml(view.view_kind)}</span><br>
    <span>heading ${formatNumber(view.relative_heading)} · pitch ${formatNumber(view.pitch)} · fov ${formatNumber(view.fov)}</span><br>
    <span>${view.output_width}x${view.output_height} · ${view.polygons.length} polygon${view.polygons.length === 1 ? "" : "s"} · ${openModeInput.checked ? "open mode" : "toggle mode"}</span>
    <button type="button">Open selected view</button>
  `;
  selectedViewEl.querySelector("button").addEventListener("click", () => openViewImage(viewset, view));
}

function openViewImage(viewset, view) {
  const params = new URLSearchParams({ viewset: viewset.name, view: view.id });
  window.open(`/view?${params.toString()}`, "_blank", "noopener");
}

function formatNumber(value) {
  return Number(value).toFixed(1).replace(/\.0$/, "");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadState().catch((error) => {
  summary.textContent = error instanceof Error ? error.message : String(error);
});
