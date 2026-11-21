const svgNamespace = "http://www.w3.org/2000/svg";
const number = new Intl.NumberFormat("en-CA", { maximumFractionDigits: 1 });
const $ = (id) => document.getElementById(id);
const cellPaths = new Map();
let grid;
let forecast;
let selectedId;

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(svgNamespace, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Forecast request failed");
  return response.json();
}

function drawMap() {
  $("canada-map").setAttribute("viewBox", grid.viewBox.join(" "));
  $("map-outline").append(svg("path", { d: grid.outline, "fill-rule": "evenodd" }));
  $("map-provinces").append(svg("path", { d: grid.provinces }));
  const shapes = document.createDocumentFragment();
  for (const cell of grid.cells) {
    const path = svg("path", {
      class: "map-cell",
      d: cell.path,
      "fill-rule": "evenodd",
      "data-cell": cell.id,
      role: "button",
      tabindex: "-1",
      "aria-pressed": "false",
    });
    path.append(svg("title"));
    cellPaths.set(cell.id, path);
    shapes.append(path);
  }
  $("map-cells").append(shapes);
  for (const city of grid.cities) {
    const [x, y] = city.point;
    const left = city.name === "Montréal" || city.name === "Halifax";
    $("map-cities").append(
      svg("circle", { cx: x, cy: y, r: 3.5 }),
      svg("text", { x: x + (left ? -7 : 7), y: y + 4, "text-anchor": left ? "end" : "start" }, city.name),
    );
  }
}

function paintEnergy() {
  const max = forecast.cells[forecast.highest_cell].energy_kwh;
  let positive = 0;
  for (const cell of grid.cells) {
    const path = cellPaths.get(cell.id);
    const value = forecast.cells[cell.id]?.energy_kwh;
    path.dataset.forecast = value === undefined ? "unavailable" : "available";
    if (value > 0) {
      path.dataset.band = Math.min(4, Math.floor((value / max) * 5));
      positive++;
    } else {
      delete path.dataset.band;
    }
    const label = cell.region + " · " + (value === undefined ? "Forecast unavailable" : formatReading(value) + " kWh");
    path.firstElementChild.textContent = label;
    path.setAttribute("aria-label", label);
  }
  const forecastCount = Object.keys(forecast.cells).length;
  $("legend-max").textContent = "Up to " + formatReading(max) + " kWh";
  $("legend-energy").hidden = max <= 0;
  $("legend-missing").hidden = forecastCount === grid.cells.length;
  $("legend").hidden = false;
  $("coverage").textContent = positive + " of " + forecastCount + " forecast cells above 0 kWh";
  $("highest").disabled = max <= 0;
}

function formatReading(value) {
  return value > 0 && value < 0.05 ? "<0.1" : number.format(value);
}

function localFormat(zone, options = {}) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: zone, hour: "2-digit", minute: "2-digit", hourCycle: "h23", ...options,
  });
}

function drawChart(times, power, peakIndex, clock) {
  const chart = $("power-chart");
  const xStart = 45, xEnd = 339, yTop = 18, yBottom = 142;
  const maximum = Math.max(1, power[peakIndex]);
  const rawStep = maximum / 4;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 2.5, 5, 10].map((value) => value * magnitude).find((value) => value >= rawStep);
  const ticks = Math.ceil(maximum / step);
  const limit = ticks * step;
  const point = (value, index) => [
    xStart + (index / (power.length - 1)) * (xEnd - xStart),
    yBottom - (value / limit) * (yBottom - yTop),
  ];
  const points = power.map(point);
  const children = [];
  for (let tick = 0; tick <= ticks; tick++) {
    const y = yBottom - (tick / ticks) * (yBottom - yTop);
    children.push(
      svg("line", { x1: xStart, x2: xEnd, y1: y, y2: y, class: "chart-grid" }),
      svg("text", { x: xStart - 6, y: y + 4, "text-anchor": "end", class: "chart-tick" }, number.format(tick * step)),
    );
  }
  for (const index of [0, 6, 12, 18, 23]) {
    children.push(svg("text", {
      x: points[index][0], y: 163, "text-anchor": "middle", class: "chart-tick",
    }, clock.format(times[index] * 1000)));
  }
  const line = "M" + points.map((pair) => pair.join(",")).join("L");
  children.push(
    svg("path", { d: line + "L" + xEnd + "," + yBottom + "L" + xStart + "," + yBottom + "Z", class: "chart-area" }),
    svg("path", { d: line, class: "chart-line" }),
  );
  const [peakX, peakY] = points[peakIndex];
  const dot = svg("circle", { cx: peakX, cy: peakY, r: 4.5, class: "chart-peak" });
  dot.append(svg("title", {}, formatReading(power[peakIndex]) + " kW at " + clock.format(times[peakIndex] * 1000)));
  children.push(dot);
  chart.replaceChildren(...children);
}

function selectCell(cellId, moveFocus = false) {
  const cell = grid.cells.find((item) => item.id === cellId);
  if (!cell) return;
  if (selectedId) {
    cellPaths.get(selectedId).setAttribute("aria-pressed", "false");
    cellPaths.get(selectedId).setAttribute("tabindex", "-1");
  }
  selectedId = cellId;
  const target = cellPaths.get(cellId);
  target.setAttribute("aria-pressed", "true");
  target.setAttribute("tabindex", "0");
  if (moveFocus) target.focus({ preventScroll: true });

  const [cx, cy] = cell.center;
  $("map-selection").replaceChildren(
    svg("path", { d: cell.path, class: "selected-outline" }),
    svg("circle", { cx, cy, r: 5, class: "selected-dot" }),
  );
  $("location-name").textContent = cell.region;
  $("coordinates").textContent = cell.latitude.toFixed(2) + "° N, " + Math.abs(cell.longitude).toFixed(2) + "° W";
  const result = forecast?.cells[cellId];
  $("top-label").hidden = !(result?.energy_kwh > 0 && cellId === forecast.highest_cell);
  $("site-forecast").hidden = !result;
  if (!result) {
    $("detail-status").hidden = false;
    $("detail-status").textContent = forecast ? "No forecast is available for this cell." : "Fetching weather for the grid…";
    return;
  }

  $("detail-status").hidden = true;
  const power = result.hourly_kw;
  const times = forecast.hour_ends;
  const peak = Math.max(...power);
  const peakIndex = power.indexOf(peak);
  const clock = localFormat(cell.timezone);
  const dateAndTime = localFormat(cell.timezone, { month: "short", day: "numeric" });
  $("total").textContent = formatReading(result.energy_kwh);
  $("peak").textContent = formatReading(peak);
  $("peak-time").textContent = peak > 0 ? dateAndTime.format(times[peakIndex] * 1000) : "Zero predicted output";
  $("window-label").textContent = dateAndTime.formatRange(new Date(forecast.window_start), new Date(times.at(-1) * 1000));
  $("model-note").hidden = !result.daylight_temperature_outside_training_range;
  drawChart(times, power, peakIndex, clock);
  const rows = document.createDocumentFragment();
  for (const [index, timestamp] of times.entries()) {
    const row = document.createElement("tr");
    row.insertCell().textContent = dateAndTime.format(timestamp * 1000);
    row.insertCell().textContent = formatReading(power[index]);
    rows.append(row);
  }
  $("hourly-rows").replaceChildren(rows);
}

function moveSelection(key) {
  const directions = { ArrowUp: [0, -1], ArrowRight: [1, 0], ArrowDown: [0, 1], ArrowLeft: [-1, 0] };
  const direction = directions[key];
  if (!direction) return;
  const [dx, dy] = direction;
  const current = grid.cells.find((cell) => cell.id === selectedId);
  const [x, y] = current.center;
  let next;
  let nearest = Infinity;
  for (const cell of grid.cells) {
    const [cx, cy] = cell.center;
    const along = (cx - x) * dx + (cy - y) * dy;
    const across = Math.abs((cx - x) * dy - (cy - y) * dx);
    const distance = along + 2.5 * across;
    if (along > 1 && distance < nearest) {
      nearest = distance;
      next = cell.id;
    }
  }
  if (next) selectCell(next, true);
}

async function loadForecast() {
  $("retry").hidden = true;
  $("map-loading").hidden = false;
  $("map-loading").textContent = "Loading Canada forecast…";
  $("detail-status").hidden = false;
  $("detail-status").textContent = "Fetching weather for the grid…";
  try {
    if (!grid) {
      grid = await getJson("/assets/canada-grid.json");
      drawMap();
      selectCell(grid.default);
    }
    forecast = await getJson("/forecast/canada");
    paintEnergy();
    selectCell(selectedId);
    $("map-loading").hidden = true;
    const stamp = new Intl.DateTimeFormat("en-CA", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
    $("updated").textContent = (forecast.stale ? "Previous forecast · " : "Weather fetched ") + stamp.format(new Date(forecast.updated_at));
  } catch (error) {
    $("map-loading").textContent = "Forecast unavailable";
    $("detail-status").hidden = false;
    $("detail-status").textContent = "The forecast could not be loaded. Please try again.";
    $("retry").hidden = false;
    console.error(error);
  }
}

$("map-cells").addEventListener("click", (event) => {
  const path = event.target.closest("[data-cell]");
  if (path) selectCell(path.dataset.cell);
});
$("map-cells").addEventListener("keydown", (event) => {
  if (["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"].includes(event.key)) {
    event.preventDefault();
    moveSelection(event.key);
  } else if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    selectCell(event.target.dataset.cell, true);
  }
});
$("highest").addEventListener("click", () => selectCell(forecast.highest_cell, true));
$("retry").addEventListener("click", loadForecast);
loadForecast();
