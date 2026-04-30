// src/server/static/js/zones_editor.js

const canvas = document.getElementById("cv");
const ctx = canvas.getContext("2d");

let bgImg = new Image();
let points = [];     // Stored in ORIGINAL pixel coords (matching bgImg.naturalWidth)
let closed = false;

let zones = [];      // DB zones polygons in ORIGINAL pixel coords
let selectedZoneId = null;
let modeInfos = new Map(); // camera_id -> { editable, reason, default_clip_id, clips[] }
let modeEditable = true;

// --- undo/redo history ---
let history = [];
let redoStack = [];
const HISTORY_LIMIT = 50;
const FALLBACK_CAMERAS = ["cam1", "cam1_hot"];

// Visual constraint (CSS), not logic constraint
const MAX_DISPLAY_WIDTH = 960; 

// ---------- helpers ----------
function $(id) { return document.getElementById(id); }
function cam() { return $("cameraId").value.trim(); }
function clip() { return $("clipId").value.trim(); }

function setStatus(msg, type = "info") {
  const el = $("zoneStatus");
  if (!el) return;
  const colors = {
    info: "#64748b",
    warn: "#b45309",
    ok: "#15803d",
    err: "#b91c1c",
  };
  el.style.color = colors[type] || colors.info;
  el.textContent = msg || "";
}

function setBusy(isBusy) {
  const ids = [
    "loadFrameBtn","refreshZonesBtn","saveBtn","deleteZoneBtn",
    "exportYamlBtn","importYamlBtn","downloadSnapshotBtn",
    "undoBtn","redoBtn","clearBtn","closeBtn","recomputeBtn"
  ];
  ids.forEach(i => {
    const el = $(i);
    if (el) el.disabled = !!isBusy;
  });
  if (!isBusy) applyModeUI();
}

async function fetchText(url, opts = {}) {
  const res = await fetch(url, { cache: "no-store", ...opts });
  const text = await res.text().catch(() => "");
  if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
  return text;
}

async function fetchJSON(url, opts = {}) {
  const text = await fetchText(url, opts);
  try { return JSON.parse(text); } catch { return text; }
}

function downloadText(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function snapshotState(reason = "") {
  const s = {
    reason,
    points: points.map(p => ({ x: p.x, y: p.y })),
    closed,
    selectedZoneId,
    zoneName: $("zoneName") ? $("zoneName").value : "",
    zoneType: $("zoneType") ? $("zoneType").value : "other",
    cameraId: $("cameraId") ? $("cameraId").value : "",
    clipId: $("clipId") ? $("clipId").value : "",
    canvasW: canvas.width,
    canvasH: canvas.height,
  };
  history.push(s);
  if (history.length > HISTORY_LIMIT) history.shift();
  redoStack = [];
  updateUndoRedoButtons();
}

function restoreState(s) {
  points = (s.points || []).map(p => ({ x: p.x, y: p.y }));
  closed = !!s.closed;
  selectedZoneId = s.selectedZoneId ?? null;

  if ($("zoneName")) $("zoneName").value = s.zoneName ?? "";
  if ($("zoneType")) $("zoneType").value = s.zoneType ?? "other";
  if ($("cameraId")) $("cameraId").value = s.cameraId ?? $("cameraId").value;
  if ($("clipId") && s.clipId) $("clipId").value = s.clipId;

  // Restore canvas dimensions (Important: we rely on canvas width matching image width)
  if (s.canvasW && s.canvasH) {
    canvas.width = s.canvasW;
    canvas.height = s.canvasH;
  }

  renderZonesList();
  draw();
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  const u = $("undoBtn");
  const r = $("redoBtn");
  if (u) u.disabled = (history.length <= 1);
  if (r) r.disabled = (redoStack.length === 0);
}

function activeModeInfo() {
  return modeInfos.get(cam()) || {
    camera_id: cam(),
    editable: true,
    reason: null,
    default_clip_id: null,
    clips: []
  };
}

function applyModeUI() {
  const info = activeModeInfo();
  modeEditable = !!info.editable;
  updateCameraAliasHint();

  const notice = $("modeNotice");
  if (notice) {
    if (modeEditable) {
      notice.style.color = "#15803d";
      notice.textContent = "Zone editing enabled for this source.";
    } else {
      notice.style.color = "#b45309";
      notice.textContent = info.reason || "This mode is not zone-editable.";
    }
  }

  const clipSel = $("clipId");
  if (clipSel) clipSel.disabled = !modeEditable;

  const editIds = ["loadFrameBtn", "saveBtn", "deleteZoneBtn", "closeBtn", "clearBtn", "undoBtn", "redoBtn"];
  editIds.forEach(id => {
    const el = $(id);
    if (el) el.disabled = !modeEditable;
  });
}

function updateCameraAliasHint() {
  const el = $("cameraAliasHint");
  if (!el) return;
  const c = cam();
  if (c === "cam1_hot") {
    el.textContent = "CAM1 note: cam1_hot uses the same source clip/CSV as cam1 (retail-shop_CAM1).";
    el.style.color = "#1e40af";
    return;
  }
  if (c === "cam1") {
    el.textContent = "CAM1 note: cam1 and cam1_hot are mapped to the same source clip/CSV (retail-shop_CAM1).";
    el.style.color = "#1e40af";
    return;
  }
  el.textContent = "";
}

// ---------- canvas rendering ----------
function zoneVisualStyle(zone) {
  const zid = String(zone?.zone_id || "").toLowerCase();
  const zt = String(zone?.zone_type || "").toLowerCase();

  if (zt.includes("entrance") || zid.includes("entrance")) {
    return { stroke: "#16a34a", fill: "rgba(22,163,74,0.22)", labelBg: "#166534", labelBorder: "#14532d" };
  }
  if (zt.includes("walkway") || zt.includes("aisle") || zid.includes("walkway") || zid.includes("aisle") || zid.includes("decision")) {
    return { stroke: "#06b6d4", fill: "rgba(6,182,212,0.20)", labelBg: "#0e7490", labelBorder: "#155e75" };
  }
  if (
    zt.includes("promo") || zt.includes("hot") || zt.includes("engage") || zt.includes("brows") ||
    zid.includes("promo") || zid.includes("hot") || zid.includes("engage") || zid.includes("brows")
  ) {
    return { stroke: "#f97316", fill: "rgba(249,115,22,0.22)", labelBg: "#c2410c", labelBorder: "#9a3412" };
  }
  if (zt.includes("cashier") || zt.includes("checkout") || zid.includes("cashier") || zid.includes("checkout") || zid.includes("counter")) {
    return { stroke: "#a855f7", fill: "rgba(168,85,247,0.22)", labelBg: "#7e22ce", labelBorder: "#581c87" };
  }
  return { stroke: "#2563eb", fill: "rgba(37,99,235,0.18)", labelBg: "#1d4ed8", labelBorder: "#1e3a8a" };
}

function zoneLabelAnchor(poly) {
  if (!poly.length) return { x: 8, y: 24 };
  let minY = Infinity;
  let maxY = -Infinity;
  let sumX = 0;
  for (const p of poly) {
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
    sumX += p.x;
  }
  const cx = sumX / poly.length;
  let y = minY - 10; // prefer outside/above polygon
  if (y < 24) y = Math.min(canvas.height - 20, maxY + 20); // fallback below when too close to top
  return { x: cx, y };
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function drawLabel(text, x, y, bg = "#1d4ed8", border = "#1e3a8a") {
  ctx.save();
  ctx.font = "700 14px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  const padX = 8, padY = 5;
  const w = ctx.measureText(text).width + padX * 2;
  const h = 18 + padY * 2;
  const rx = Math.max(4, Math.min(x - w / 2, canvas.width - w - 4));
  const ry = Math.max(4, Math.min(y - h, canvas.height - h - 4));

  ctx.shadowColor = "rgba(15,23,42,0.45)";
  ctx.shadowBlur = 8;
  ctx.shadowOffsetY = 2;
  ctx.fillStyle = bg;
  ctx.strokeStyle = border;
  ctx.lineWidth = 2;
  roundRect(ctx, rx, ry, w, h, 6);
  ctx.fill();
  ctx.stroke();

  ctx.shadowColor = "transparent";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, rx + padX, ry + h - padY - 3);
  ctx.restore();
}

function draw() {
  // 1. Clear Canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 2. Draw Background Image
  if (bgImg.complete && bgImg.naturalWidth) {
    // Since canvas.width matches bgImg.naturalWidth, we draw 1:1
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto, Arial";
    ctx.fillText("Click “Load Frame” to start.", 20, 30);
  }

  // 3. Draw DB Zones
  // DB zones are in Original coords. Since canvas is Original size, draw directly.
  zones.forEach(z => {
    const poly = (z.polygon || []).map(p => ({ x: Number(p[0]), y: Number(p[1]) }));
    if (poly.length < 3) return;
    const style = zoneVisualStyle(z);
    const active = (z.zone_id === selectedZoneId);

    ctx.save();
    ctx.beginPath();
    poly.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.closePath();

    ctx.lineWidth = active ? 5 : 4;
    ctx.strokeStyle = active ? "#ffffff" : style.stroke;
    ctx.fillStyle = style.fill;
    ctx.stroke();
    ctx.fill();

    if (active) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = style.stroke;
      ctx.stroke();
    }

    const label = z.zone_id || z.name || "zone";
    const anchor = zoneLabelAnchor(poly);
    drawLabel(label, anchor.x, anchor.y, style.labelBg, style.labelBorder);
    ctx.restore();
  });

  // 4. Draw Current Polygon (in progress)
  if (points.length > 0) {
    ctx.save();
    ctx.beginPath();
    points.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    if (closed) ctx.closePath();

    ctx.lineWidth = 4;
    ctx.strokeStyle = "#fb923c";
    ctx.stroke();

    if (closed) {
      ctx.fillStyle = "rgba(251,146,60,0.22)";
      ctx.fill();
    }

    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#fb923c";
      ctx.fill();
      ctx.strokeStyle = "rgba(15,23,42,0.15)";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    const last = points[points.length - 1];
    drawLabel(
      closed ? "Closed polygon" : `Points: ${points.length}`,
      last.x + 8,
      last.y - 8,
      "#9a3412",
      "#7c2d12"
    );
    ctx.restore();
  }
}

// ---------- overlap warning ----------
function polyBBox(poly) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x,y] of poly) {
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
  }
  return {minX, minY, maxX, maxY};
}

function checkOverlapWarnings() {
  if (!zones.length) { setStatus("Ready.", "info"); return; }

  const bboxes = zones
    .filter(z => Array.isArray(z.polygon) && z.polygon.length >= 3)
    .map(z => ({ id: z.zone_id, bb: polyBBox(z.polygon) }));

  const overlaps = [];
  for (let i = 0; i < bboxes.length; i++) {
    for (let j = i + 1; j < bboxes.length; j++) {
      const a = bboxes[i], b = bboxes[j];
      const sep =
        (a.bb.maxX < b.bb.minX) || (b.bb.maxX < a.bb.minX) ||
        (a.bb.maxY < b.bb.minY) || (b.bb.maxY < a.bb.minY);

      if (!sep) overlaps.push([a.id, b.id]);
    }
  }

  if (overlaps.length) {
    const msg = `⚠️ Potential overlap: ${overlaps.slice(0,3).map(p => `${p[0]} ↔ ${p[1]}`).join(", ")}${overlaps.length > 3 ? " …" : ""}`;
    setStatus(msg, "warn");
  } else {
    setStatus("✓ Zones loaded. No overlaps.", "ok");
  }
}

// ---------- API ----------
async function loadClips() {
  const sel = $("clipId");
  sel.innerHTML = "";
  if (!cam()) {
    const camSel = $("cameraId");
    if (camSel && camSel.options && camSel.options.length > 0) {
      camSel.value = camSel.options[0].value;
    }
  }

  const data = await fetchJSON(`/kpi/clips?camera_id=${encodeURIComponent(cam())}&for_zone_editor=1`);
  const clips = Array.isArray(data) ? data : (data.clips || []);
  const editable = Array.isArray(data) ? true : (data.editable !== false);
  const reason = Array.isArray(data) ? null : (data.reason || null);
  const defaultClip = Array.isArray(data) ? null : (data.default_clip_id || null);

  modeInfos.set(cam(), {
    camera_id: cam(),
    editable,
    reason,
    default_clip_id: defaultClip,
    clips
  });

  if (!editable) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(mode has no reference frame)";
    sel.appendChild(opt);
    applyModeUI();
    return;
  }

  if (!clips.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no editable raw clip configured)";
    sel.appendChild(opt);
    applyModeUI();
    return;
  }

  clips.forEach(cid => {
    const opt = document.createElement("option");
    opt.value = cid;
    opt.textContent = cid;
    sel.appendChild(opt);
  });

  if (defaultClip && clips.includes(defaultClip)) {
    sel.value = defaultClip;
  }
  applyModeUI();
}

async function loadCameraModes(preferredCameraId = "") {
  const sel = $("cameraId");
  if (!sel) return;

  const data = await fetchJSON("/meta/cameras?for_zone_editor=1");
  const camerasRaw = Array.isArray(data?.cameras) && data.cameras.length
    ? data.cameras
    : FALLBACK_CAMERAS;
  const cameras = camerasRaw
    .map(c => (typeof c === "string" ? c : String(c?.camera_id || c?.id || "")))
    .map(c => c.trim())
    .filter(Boolean);
  const cameraModes = Array.isArray(data?.camera_modes) ? data.camera_modes : [];

  modeInfos = new Map();
  cameraModes.forEach(m => {
    if (m?.camera_id) modeInfos.set(String(m.camera_id), m);
  });

  sel.innerHTML = "";
  for (const c of cameras) {
    const info = modeInfos.get(c);
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = info?.editable === false ? `${c} (analytics-only)` : c;
    sel.appendChild(opt);
  }

  const existing = preferredCameraId || cam();
  if (existing && cameras.includes(existing)) {
    sel.value = existing;
  } else if (cameras.includes("cam1")) {
    sel.value = "cam1";
  } else if (cameras.length > 0) {
    sel.value = cameras[0];
  }
  applyModeUI();
}

async function loadFrame() {
  if (!modeEditable) {
    setStatus(activeModeInfo().reason || "This mode has no single reference frame.", "warn");
    return;
  }

  const cid = clip();
  if (!cid) {
    setStatus("Pick a valid raw clip first.", "warn");
    return;
  }

  setBusy(true);
  points = [];
  closed = false;
  selectedZoneId = null;

  bgImg = new Image();
  
  bgImg.onload = () => {
    // CRITICAL FIX: Set Canvas Internal Resolution to Match Image Resolution exactly.
    // This ensures 1 Canvas Pixel = 1 Image Pixel.
    canvas.width = bgImg.naturalWidth;
    canvas.height = bgImg.naturalHeight;

    // Visual sizing is handled by CSS (max-width: 100%), so no JS math needed for view.
    
    snapshotState("loadFrame");
    draw();
    setBusy(false);
    setStatus("Frame loaded. Draw polygon points.", "info");
  };

  bgImg.onerror = () => {
    setBusy(false);
    alert("Failed to load frame image.");
  };

  bgImg.src = `/zones/frame?clip_id=${encodeURIComponent(cid)}&frame_idx=0&_t=${Date.now()}`;
}

async function refreshZones() {
  setBusy(true);
  try {
    const data = await fetchJSON(`/zones?camera_id=${encodeURIComponent(cam())}`);
    zones = Array.isArray(data) ? data : [];
    renderZonesList();
    draw();
    checkOverlapWarnings();
  } catch (e) {
    alert("Refresh zones failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

function renderZonesList() {
  const box = $("zonesList");
  box.innerHTML = "";

  if (!zones.length) {
    const empty = document.createElement("div");
    empty.style.padding = "10px";
    empty.style.color = "#64748b";
    empty.textContent = "No zones in DB.";
    box.appendChild(empty);
    return;
  }

  zones.forEach(z => {
    const div = document.createElement("div");
    div.className = "zoneItem" + (z.zone_id === selectedZoneId ? " active" : "");

    const left = document.createElement("div");
    left.textContent = z.zone_id || z.name || "zone";

    const right = document.createElement("div");
    right.style.color = "#64748b";
    right.style.fontSize = "12px";
    right.style.fontWeight = "800";
    right.textContent = (z.zone_type || "other");

    div.appendChild(left);
    div.appendChild(right);

    div.onclick = () => {
      snapshotState("selectZone");
      selectedZoneId = z.zone_id;
      $("zoneName").value = z.name || z.zone_id || "";
      $("zoneType").value = z.zone_type || "other";

      points = (z.polygon || []).map(p => ({ x: Number(p[0]), y: Number(p[1]) }));
      closed = true;

      renderZonesList();
      draw();
    };

    box.appendChild(div);
  });
}

async function saveZone() {
  const name = $("zoneName").value.trim();
  const ztype = $("zoneType").value;

  if (!name) return alert("Zone name required.");
  if (points.length < 3) return alert("Need at least 3 points.");
  if (!closed) return alert("Close polygon first.");

  const zoneIdToUse = selectedZoneId || name;

  const payload = {
    camera_id: cam(),
    zone_id: zoneIdToUse,
    name: name,
    zone_type: ztype,
    polygon: points.map(p => [p.x, p.y]) // 1:1 Canvas Coords = Image Coords
  };

  setBusy(true);
  try {
    await fetchText(`/zones`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    snapshotState("saveZone");
    points = [];
    closed = false;
    selectedZoneId = null;

    await refreshZones();
    setStatus("✓ Zone saved. Click 'Apply Zones to Analytics' to refresh dashboard KPIs.", "ok");
  } catch (e) {
    alert("Save failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

async function deleteSelected() {
  if (!selectedZoneId) return alert("Select a zone first.");
  if (!confirm(`Delete zone "${selectedZoneId}"?`)) return;

  setBusy(true);
  try {
    await fetchText(
      `/zones/${encodeURIComponent(selectedZoneId)}?camera_id=${encodeURIComponent(cam())}`,
      { method: "DELETE" }
    );

    snapshotState("deleteZone");
    selectedZoneId = null;
    points = [];
    closed = false;

    await refreshZones();
    setStatus("✓ Zone deleted.", "ok");
  } catch (e) {
    alert("Delete failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

async function exportYaml() {
  setBusy(true);
  try {
    const y = await fetchText(`/zones/export-yaml?camera_id=${encodeURIComponent(cam())}`);
    $("yamlBox").value = y;
    setStatus("✓ YAML exported.", "ok");
  } catch (e) {
    alert("Export failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

async function importYaml() {
  const txt = $("yamlBox").value;
  if (!txt.trim()) return alert("Paste YAML first.");

  setBusy(true);
  try {
    const responseText = await fetchText(`/zones/import-yaml`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_id: cam(),
        yaml_text: txt,
        replace_existing: true
      })
    });

    // Parse the response to see how many were imported vs skipped
    const data = JSON.parse(responseText);

    if (data.imported === 0 && data.skipped > 0) {
      // Show error if everything was skipped
      alert(`Import Failed: 0 zones imported, ${data.skipped} skipped.\n\nDetails: ${data.skipped_details.join(", ")}`);
    } else {
      // Success path
      snapshotState("importYaml");
      
      // 1. Refresh the list and canvas
      await refreshZones();
      
      // 2. Refresh the YAML text box to show the clean state from DB
      await exportYaml();

      const deleted = Number(data?.deleted_existing_zones || 0);
      alert(`Imported ${data.imported} zones successfully.${deleted ? ` Replaced ${deleted} previous zones.` : ""}`);
      setStatus("YAML imported. Click 'Apply Zones to Analytics' to refresh dashboard KPIs.", "info");
    }
  } catch (e) {
    alert("Import failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

async function recomputeAnalytics() {
  const cameraId = cam();
  const clipId = clip();
  if (!cameraId) {
    setStatus("Select a camera first.", "warn");
    return;
  }
  if (!clipId) {
    setStatus("Select a clip before applying zones to analytics.", "warn");
    return;
  }

  setBusy(true);
  setStatus("Recomputing analytics from current zones ...", "info");
  try {
    const data = await fetchJSON("/zones/recompute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_id: cameraId,
        clip_id: clipId || null
      })
    });

    const rows = Number(data?.rows_scanned || 0);
    const prevZoned = Number(data?.rows_previously_zoned || 0);
    const zoned = Number(data?.rows_zoned || 0);
    const deletedEvents = Number(data?.events_deleted || 0);
    const rebuiltEvents = Number(data?.events_rebuilt || data?.events_after_rebuild || 0);
    const tgtCam = String(data?.target_camera_id || cameraId);
    const tgtClip = data?.target_clip_id ? ` clip=${data.target_clip_id}` : " all clips";
    const requestedClip = String(data?.requested_clip_id || clipId || "").trim();
    const clipResolved = !!data?.clip_resolved;
    const zoneCam = String(data?.zone_camera_id || cameraId);
    const loadedZones = Number(data?.zones_loaded_from_db || 0);
    const zonesUsed = Array.isArray(data?.zones_used) ? data.zones_used : [];
    const zoneList = zonesUsed
      .map(z => {
        const zid = String(z?.zone_id || "").trim();
        const zt = String(z?.zone_type || "").trim();
        if (!zid) return "";
        return zt ? `${zid}(${zt})` : zid;
      })
      .filter(Boolean);
    const zoneLabel = zoneList.length ? zoneList.join(", ") : "(none)";
    const dashboardLink = $("dashboardLink");
    if (dashboardLink && data?.target_clip_id) {
      dashboardLink.href = `/ui?camera_id=${encodeURIComponent(tgtCam)}&clip_id=${encodeURIComponent(String(data.target_clip_id))}`;
    }
    setStatus(
      `✓ Analytics rebuilt for ${tgtCam}${tgtClip}${clipResolved && requestedClip ? ` (requested ${requestedClip})` : ""}. Loaded ${loadedZones} current zones from DB for ${zoneCam}. Using zones: ${zoneLabel}. tracks=${rows}, zoned(before→after)=${prevZoned}→${zoned}, events(deleted→rebuilt)=${deletedEvents}→${rebuiltEvents}. Refresh dashboard.`,
      "ok"
    );
  } catch (e) {
    setStatus(`Recompute failed: ${e.message}`, "err");
    alert("Recompute failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

async function downloadSnapshot() {
  setBusy(true);
  try {
    const camera = cam();
    const yamlText = await fetchText(`/zones/export-yaml?camera_id=${encodeURIComponent(camera)}`);
    downloadText(`zones_${camera}.yaml`, yamlText, "text/yaml");

    const jsonText = JSON.stringify({ camera_id: camera, zones }, null, 2);
    downloadText(`zones_${camera}.json`, jsonText, "application/json");

    setStatus("✓ Snapshot downloaded.", "ok");
  } catch (e) {
    alert("Snapshot failed: " + e.message);
  } finally {
    setBusy(false);
  }
}

// ---------- undo/redo ----------
function doUndo() {
  if (history.length <= 1) return;
  const current = history.pop();
  redoStack.push(current);
  const prev = history[history.length - 1];
  restoreState(prev);
}

function doRedo() {
  if (!redoStack.length) return;
  const s = redoStack.pop();
  history.push(s);
  restoreState(s);
}

// ---------- input events ----------
canvas.addEventListener("click", (e) => {
  if (!modeEditable) return;
  if (closed) return;

  const rect = canvas.getBoundingClientRect();
  
  // Map Visual DOM Coordinates -> Canvas Internal Coordinates
  // This handles CSS resizing (max-width) automatically
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);

  snapshotState("addPoint");
  points.push({ x, y });
  draw();
});

canvas.addEventListener("dblclick", () => {
  if (!modeEditable) return;
  if (points.length >= 3) {
    snapshotState("closePolygon");
    closed = true;
    draw();
  }
});

document.addEventListener("keydown", (e) => {
  if (!modeEditable) return;
  if (e.ctrlKey || e.metaKey) return;
  if (e.key === "Escape") {
    snapshotState("clear");
    points = [];
    closed = false;
    selectedZoneId = null;
    renderZonesList();
    draw();
  } else if (e.key.toLowerCase() === "u") doUndo();
  else if (e.key.toLowerCase() === "r") doRedo();
  else if (e.key.toLowerCase() === "z") {
    if (points.length >= 3) {
      snapshotState("closePolygon");
      closed = true;
      draw();
    }
  }
});

 $("loadFrameBtn").onclick = loadFrame;
 $("refreshZonesBtn").onclick = refreshZones;
 $("saveBtn").onclick = saveZone;
 $("deleteZoneBtn").onclick = deleteSelected;
 $("exportYamlBtn").onclick = exportYaml;
 $("importYamlBtn").onclick = importYaml;
const snapBtn = $("downloadSnapshotBtn");
if (snapBtn) snapBtn.onclick = downloadSnapshot;
const recomputeBtn = $("recomputeBtn");
if (recomputeBtn) recomputeBtn.onclick = recomputeAnalytics;

 $("undoBtn").onclick = doUndo;
const redoBtn = $("redoBtn");
if (redoBtn) redoBtn.onclick = doRedo;

 $("clearBtn").onclick = () => {
  snapshotState("clearBtn");
  points = [];
  closed = false;
  selectedZoneId = null;
  draw();
  renderZonesList();
};

 $("closeBtn").onclick = () => {
  if (points.length >= 3) {
    snapshotState("closeBtn");
    closed = true;
    draw();
  }
};

 $("cameraId").addEventListener("change", async () => {
  try {
    setBusy(true);
    snapshotState("cameraChange");
    await loadClips();
    await refreshZones();
    applyModeUI();
  } finally {
    setBusy(false);
  }
});

(async () => {
  try {
    setBusy(true);
    
    // --- ADD THIS BLOCK ---
    const params = new URLSearchParams(window.location.search);
    const camParam = params.get("camera_id");
    await loadCameraModes(camParam || "");
    // ---------------------

    await loadClips();
    await refreshZones();
    applyModeUI();
    snapshotState("init");
    draw();
  } catch (e) {
    console.error(e);
    alert("Init failed: " + e.message);
  } finally {
    setBusy(false);
  }
})();
