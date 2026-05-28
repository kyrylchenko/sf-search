const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const select = document.getElementById("viewsetSelect");
const summary = document.getElementById("summary");
const viewMatrixEl = document.getElementById("viewMatrix");
const selectedViewEl = document.getElementById("selectedView");
const showAllBtn = document.getElementById("showAll");
const hideAllBtn = document.getElementById("hideAll");

let state = null;
let panoImage = null;
let currentViewset = null;
let currentViewsetName = null;
let visibleViewIds = new Set();
let selectedViewId = null;

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
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(panoImage, 0, 0, canvas.width, canvas.height);
  drawOverlays(viewset.views.filter((view) => visibleViewIds.has(view.id)));
  renderSidebar(viewset);
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

function renderSidebar(viewset) {
  summary.textContent = `${viewset.description || "No description"} · ${state.pano.filename} · ${state.pano.width}x${state.pano.height}`;
  renderMatrix(viewset);
  renderSelectedView(viewset);
}

function renderMatrix(viewset) {
  viewMatrixEl.innerHTML = "";
  const pitches = [...new Set(viewset.views.map((view) => Number(view.pitch)))].sort((a, b) => b - a);
  const headings = [...new Set(viewset.views.map((view) => Number(view.relative_heading)))].sort((a, b) => a - b);
  pitches.forEach((pitch) => {
    const row = document.createElement("div");
    row.className = "matrixRow";
    const label = document.createElement("div");
    label.className = "rowLabel";
    label.textContent = `${formatNumber(pitch)}°`;
    const grid = document.createElement("div");
    grid.className = "seatGrid";
    headings.forEach((heading) => {
      const view = findView(viewset, pitch, heading);
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "seat";
      if (view) {
        checkbox.title = `${view.id} · heading ${formatNumber(view.relative_heading)} · pitch ${formatNumber(view.pitch)}`;
        checkbox.checked = visibleViewIds.has(view.id);
        checkbox.addEventListener("change", () => {
          selectedViewId = view.id;
          if (checkbox.checked) {
            visibleViewIds.add(view.id);
          } else {
            visibleViewIds.delete(view.id);
          }
          draw();
        });
        checkbox.addEventListener("mouseenter", () => {
          selectedViewId = view.id;
          renderSelectedView(viewset);
        });
      } else {
        checkbox.disabled = true;
        checkbox.title = `No view at heading ${formatNumber(heading)}, pitch ${formatNumber(pitch)}`;
      }
      grid.appendChild(checkbox);
    });
    row.appendChild(label);
    row.appendChild(grid);
    viewMatrixEl.appendChild(row);
  });
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
    <span>${view.output_width}x${view.output_height} · ${view.polygons.length} polygon${view.polygons.length === 1 ? "" : "s"}</span>
    <button type="button">Open selected view</button>
  `;
  selectedViewEl.querySelector("button").addEventListener("click", () => openViewImage(viewset, view));
}

function findView(viewset, pitch, heading) {
  return viewset.views.find(
    (view) => Number(view.pitch) === pitch && Number(view.relative_heading) === heading
  );
}

canvas.addEventListener("click", (event) => {
  if (!currentViewset) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  const view = [...currentViewset.views]
    .filter((candidate) => visibleViewIds.has(candidate.id))
    .reverse()
    .find((candidate) => candidate.polygons.some((polygon) => pointInPolygon(x, y, polygon)));
  if (view) {
    openViewImage(currentViewset, view);
  }
});

function openViewImage(viewset, view) {
  const params = new URLSearchParams({ viewset: viewset.name, view: view.id });
  window.open(`/view?${params.toString()}`, "_blank", "noopener");
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
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
