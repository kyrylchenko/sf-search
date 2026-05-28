const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const select = document.getElementById("viewsetSelect");
const summary = document.getElementById("summary");
const viewsEl = document.getElementById("views");

let state = null;
let panoImage = null;
let currentViewset = null;

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
  select.addEventListener("change", draw);
}

function draw() {
  if (!state || !panoImage) return;
  const viewset = state.viewsets[Number(select.value || 0)];
  currentViewset = viewset;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(panoImage, 0, 0, canvas.width, canvas.height);
  drawOverlays(viewset.views);
  renderSidebar(viewset);
}

function drawOverlays(views) {
  context.lineWidth = Math.max(2, canvas.width / 900);
  views.forEach((view, index) => {
    const hue = (index * 37) % 360;
    context.fillStyle = `hsla(${hue}, 95%, 55%, 0.28)`;
    context.strokeStyle = `hsla(${hue}, 95%, 65%, 0.9)`;
    view.polygons.forEach((polygon) => drawPolygon(polygon));
  });
}

function drawPolygon(polygon) {
  if (polygon.length === 0) return;
  context.beginPath();
  context.moveTo(polygon[0].x * canvas.width, polygon[0].y * canvas.height);
  polygon.slice(1).forEach((point) => {
    context.lineTo(point.x * canvas.width, point.y * canvas.height);
  });
  context.closePath();
  context.fill();
  context.stroke();
}

function renderSidebar(viewset) {
  summary.textContent = `${viewset.description || "No description"} · ${state.pano.filename} · ${state.pano.width}x${state.pano.height}`;
  viewsEl.innerHTML = "";
  viewset.views.forEach((view) => {
    const item = document.createElement("div");
    item.className = "view";
    item.tabIndex = 0;
    item.innerHTML = `
      <strong>${escapeHtml(view.id)}</strong>
      <span>${escapeHtml(view.view_kind)}</span>
      <span>heading ${formatNumber(view.relative_heading)} · pitch ${formatNumber(view.pitch)} · fov ${formatNumber(view.fov)}</span>
      <span>${view.output_width}x${view.output_height} · ${view.polygons.length} polygon${view.polygons.length === 1 ? "" : "s"}</span>
    `;
    item.addEventListener("click", () => openViewImage(viewset, view));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openViewImage(viewset, view);
      }
    });
    viewsEl.appendChild(item);
  });
}

canvas.addEventListener("click", (event) => {
  if (!currentViewset) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  const view = [...currentViewset.views]
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
