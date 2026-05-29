const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const select = document.getElementById("viewsetSelect");
const summary = document.getElementById("summary");
const viewMatrixEl = document.getElementById("viewMatrix");
const selectedViewEl = document.getElementById("selectedView");
const showAllBtn = document.getElementById("showAll");
const hideAllBtn = document.getElementById("hideAll");
const prevPanoBtn = document.getElementById("prevPano");
const nextPanoBtn = document.getElementById("nextPano");
const panoLabelEl = document.getElementById("panoLabel");

let state = null;
let panoImage = null;
let currentViewset = null;
let currentViewsetName = null;
let visibleViewIds = new Set();
let selectedViewId = null;
let controlsInitialized = false;

async function loadState(panoIndex = null) {
  const url = panoIndex === null ? "/api/state" : `/api/state?pano=${encodeURIComponent(panoIndex)}`;
  const response = await fetch(url);
  state = await response.json();
  panoImage = new Image();
  panoImage.onload = () => {
    canvas.width = state.pano.width;
    canvas.height = state.pano.height;
    if (!controlsInitialized) {
      initializeControls();
      controlsInitialized = true;
    }
    syncViewsetOptions();
    syncPanoControls();
    draw();
  };
  panoImage.src = state.pano.url;
}

function initializeControls() {
  select.addEventListener("change", () => {
    currentViewsetName = null;
    draw();
  });
  prevPanoBtn.addEventListener("click", () => {
    loadState(state.pano.previous_index);
  });
  nextPanoBtn.addEventListener("click", () => {
    loadState(state.pano.next_index);
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

function syncViewsetOptions() {
  const selectedValue = select.value;
  select.innerHTML = "";
  state.viewsets.forEach((viewset, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${viewset.name} (${viewset.views.length})`;
    select.appendChild(option);
  });
  if (selectedValue && Number(selectedValue) < state.viewsets.length) {
    select.value = selectedValue;
  }
}

function syncPanoControls() {
  const count = state.pano.count || 1;
  const index = state.pano.index || 0;
  prevPanoBtn.disabled = count <= 1;
  nextPanoBtn.disabled = count <= 1;
  panoLabelEl.textContent = `Pano ${index + 1} of ${count}: ${state.pano.filename}`;
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
        grid.appendChild(checkbox);
      });
    row.appendChild(label);
    row.appendChild(grid);
    viewMatrixEl.appendChild(row);
  });
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
    <span>${view.output_width}x${view.output_height} · ${view.polygons.length} polygon${view.polygons.length === 1 ? "" : "s"}</span>
    <button type="button">Open selected view</button>
    <div class="viewSearch">
      <input id="viewSearchInput" type="search" placeholder="Search views by id, kind, heading, pitch, fov" autocomplete="off">
      <div id="viewSearchResults" class="searchResults"></div>
    </div>
  `;
  selectedViewEl.querySelector("button").addEventListener("click", () => openViewImage(viewset, view));
  wireViewSearch(viewset);
}

function wireViewSearch(viewset) {
  const input = document.getElementById("viewSearchInput");
  const results = document.getElementById("viewSearchResults");
  if (!input || !results) return;
  const render = () => {
    const matches = fuzzyViewMatches(viewset.views, input.value).slice(0, 8);
    results.innerHTML = "";
    matches.forEach((view) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "searchResult";
      button.innerHTML = `
        <strong>${escapeHtml(view.id)}</strong>
        <span>${escapeHtml(view.view_kind)} · heading ${formatNumber(view.relative_heading)} · pitch ${formatNumber(view.pitch)} · fov ${formatNumber(view.fov)}</span>
      `;
      button.addEventListener("click", () => {
        selectedViewId = view.id;
        renderSelectedView(viewset);
        openViewImage(viewset, view);
      });
      results.appendChild(button);
    });
  };
  input.addEventListener("input", render);
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    const [first] = fuzzyViewMatches(viewset.views, input.value);
    if (!first) return;
    event.preventDefault();
    selectedViewId = first.id;
    renderSelectedView(viewset);
    openViewImage(viewset, first);
  });
  render();
}

function fuzzyViewMatches(views, query) {
  const normalizedQuery = normalizeSearch(query);
  return views
    .map((view, index) => ({ view, index, score: fuzzyScore(viewSearchText(view), normalizedQuery) }))
    .filter((item) => item.score !== null)
    .sort((a, b) => a.score - b.score || a.index - b.index)
    .map((item) => item.view);
}

function fuzzyScore(value, query) {
  if (query === "") return 1000;
  let score = 0;
  let position = 0;
  const normalizedValue = normalizeSearch(value);
  for (const character of query) {
    const found = normalizedValue.indexOf(character, position);
    if (found === -1) return null;
    score += found - position;
    position = found + 1;
  }
  return score + normalizedValue.length / 1000;
}

function viewSearchText(view) {
  return [
    view.id,
    view.view_kind,
    `heading ${formatNumber(view.relative_heading)}`,
    `pitch ${formatNumber(view.pitch)}`,
    `fov ${formatNumber(view.fov)}`,
  ].join(" ");
}

function normalizeSearch(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9.-]+/g, "");
}

function openViewImage(viewset, view) {
  const params = new URLSearchParams({
    viewset: viewset.name,
    view: view.id,
    pano: String(state.pano.index || 0),
  });
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
