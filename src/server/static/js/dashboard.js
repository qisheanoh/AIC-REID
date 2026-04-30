async function fetchText(url, opts = {}) {
  const res = await fetch(url, { cache: "no-store", ...opts });
  const text = await res.text().catch(() => "");
  if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
  return text;
}

async function fetchJSON(url, opts = {}) {
  const text = await fetchText(url, opts);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function withParams(base, params) {
  const u = new URL(base, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") {
      u.searchParams.set(k, v);
    }
  });
  return u.toString();
}

function toNum(x, fallback = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
}

function setError(msg) {
  const el = document.getElementById("dashboard-error");
  if (el) el.textContent = msg || "";
}

function setFreshnessNotice(level, msg) {
  const el = document.getElementById("freshnessNotice");
  if (!el) return;
  el.className = "";
  if (!msg || !level || level === "ok") {
    el.style.display = "none";
    el.textContent = "";
    return;
  }
  el.className = String(level);
  el.style.display = "block";
  el.textContent = msg;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

function fmtPct(p) {
  if (p === null || p === undefined) return "N/A";
  const n = Number(p);
  if (!Number.isFinite(n)) return "N/A";
  return `${n.toFixed(1)}%`;
}

function fmtDeltaPct(p) {
  if (p === null || p === undefined) return "N/A";
  const n = Number(p);
  if (!Number.isFinite(n)) return "N/A";
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

function setBusy(isBusy) {
  const ids = ["cameraSelect", "groupSelect", "clipSelect", "btnAB", "btnInsights"];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = !!isBusy;
  });
}

function renderEmptyRow(tbody, colspan, msg) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td colspan="${colspan}" style="color:#64748b; padding:12px;">${msg}</td>`;
  tbody.appendChild(tr);
}

function deltaClass(val) {
  const s = String(val || "").trim();
  if (s.startsWith("+")) return "pos";
  if (s.startsWith("-")) return "neg";
  return "neu";
}

const PRIMARY_MODES = [
  { id: "cam1", label: "cam1", desc: "baseline entrance flow" },
  { id: "cam1_hot", label: "cam1_hot", desc: "hot/promo engagement" },
  { id: "cross_cam", label: "cross_cam", desc: "cross-camera continuity" }
];
const INITIAL_QS = new URLSearchParams(window.location.search);
const INITIAL_CAMERA = INITIAL_QS.get("camera_id") || "";
const INITIAL_CLIP = INITIAL_QS.get("clip_id") || "";
let initialCameraPending = true;
let initialClipPending = true;

function modeMeta(modeId) {
  return PRIMARY_MODES.find(m => m.id === modeId) || { id: modeId, label: modeId, desc: "custom mode" };
}

function getComparisonPair() {
  return {
    camA: "cam1",
    camB: "cam1_hot"
  };
}

function currentCamera() {
  return document.getElementById("cameraSelect")?.value || "cam1";
}

function isCrossCamMode() {
  return currentCamera() === "cross_cam";
}

function setSummaryCards(cards) {
  for (let i = 0; i < 4; i++) {
    const c = cards[i] || {
      label: `Metric ${i + 1}`,
      value: "–",
      hint: "-"
    };

    setText(`card-label-${i + 1}`, c.label || `Metric ${i + 1}`);
    setText(`card-value-${i + 1}`, c.value || "–");
    setText(`card-hint-${i + 1}`, c.hint || "-");
  }
}

function updateZoneLink() {
  const cam = document.getElementById("cameraSelect")?.value || "cam1";
  const link = document.getElementById("zoneLink");
  if (link) {
    link.href = `/ui/zones?camera_id=${encodeURIComponent(cam)}`;
  }
}

function updateStaticTexts() {
  const cam = currentCamera();
  const meta = modeMeta(cam);
  const pair = getComparisonPair();
  const metaA = modeMeta(pair.camA);
  const metaB = modeMeta(pair.camB);

  const abTitle = document.getElementById("comparison-title");
  const abSub = document.getElementById("comparison-sub");
  const abHeadA = document.getElementById("ab-head-a");
  const abHeadB = document.getElementById("ab-head-b");
  const abBtn = document.getElementById("btnAB");
  const insightsSub = document.getElementById("insights-sub");
  const zoneSub = document.getElementById("zone-kpi-sub");
  const modeSub = document.getElementById("mode-sub");

  if (abTitle) {
    abTitle.textContent = `Mode Comparison (${metaA.label} vs ${metaB.label})`;
  }
  if (abSub) {
    abSub.textContent =
      "Exploratory readout only. Compare operational modes, not causal treatment effects.";
  }
  if (abHeadA) {
    abHeadA.textContent = `A (${metaA.label})`;
  }
  if (abHeadB) {
    abHeadB.textContent = `B (${metaB.label})`;
  }
  if (abBtn) {
    abBtn.textContent = "Generate Comparison Insights";
  }
  if (insightsSub) {
    insightsSub.textContent = "Privacy-first, edge-friendly insights from ordinary CCTV using zone-semantic retail analytics.";
  }
  if (zoneSub) {
    zoneSub.textContent = "Raw visits count all entries. Qualified visits require zone-specific dwell thresholds.";
  }
  if (modeSub) {
    modeSub.textContent = `Current mode: ${meta.label} (${meta.desc}).`;
  }
}

function updateModePanels() {
  const abTitle = document.getElementById("comparison-title");
  const abSection = abTitle ? abTitle.closest(".section") : null;
  const abBtn = document.getElementById("btnAB");
  const abTbody = document.querySelector("#ab-table tbody");
  const abBox = document.getElementById("abInsightsBox");

  if (!abSection) return;

  if (isCrossCamMode()) {
    abSection.style.display = "none";
    if (abBtn) abBtn.disabled = true;
    if (abTbody) abTbody.innerHTML = "";
    if (abBox) abBox.innerHTML = "";
    return;
  }

  abSection.style.display = "";
  if (abBtn) abBtn.disabled = false;
}

function getParams() {
  const cam = document.getElementById("cameraSelect")?.value || "cam1";
  const group = document.getElementById("groupSelect")?.value || "";
  const clip = document.getElementById("clipSelect")?.value || "";

  const params = { camera_id: cam };
  if (clip) params.clip_id = clip;
  else if (group) params.group = group;

  return params;
}

async function loadCameras() {
  const sel = document.getElementById("cameraSelect");
  if (!sel) return;

  const current = sel.value;
  const data = await fetchJSON("/meta/cameras");
  const cams = (data && data.cameras) ? data.cameras : [];

  sel.innerHTML = "";

  const ordered = [];
  const seen = new Set();
  for (const p of PRIMARY_MODES.map(m => m.id)) {
    if (!seen.has(p)) {
      ordered.push(p);
      seen.add(p);
    }
  }
  for (const c of cams) {
    if (!seen.has(c)) {
      ordered.push(c);
      seen.add(c);
    }
  }

  for (const c of ordered) {
    const m = modeMeta(c);
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = m.desc === "custom mode" ? m.label : `${m.label} - ${m.desc}`;
    sel.appendChild(opt);
  }

  if (initialCameraPending && INITIAL_CAMERA && ordered.includes(INITIAL_CAMERA)) {
    sel.value = INITIAL_CAMERA;
  } else if (current && ordered.includes(current)) {
    sel.value = current;
  } else if (ordered.length > 0) {
    sel.value = ordered[0];
  }
  initialCameraPending = false;

  updateZoneLink();
  updateStaticTexts();
  updateModePanels();
}

async function loadGroups() {
  const cam = document.getElementById("cameraSelect")?.value || "cam1";
  const sel = document.getElementById("groupSelect");
  if (!sel) return;

  const current = sel.value;
  const data = await fetchJSON(withParams("/meta/groups", { camera_id: cam }));
  const groups = (data && data.groups) ? data.groups : [];

  sel.innerHTML = `<option value="">All</option>`;
  for (const g of groups) {
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    sel.appendChild(opt);
  }

  if (current && groups.includes(current)) {
    sel.value = current;
  } else {
    sel.value = "";
  }
}

async function loadClips() {
  const cam = document.getElementById("cameraSelect")?.value || "cam1";
  const sel = document.getElementById("clipSelect");
  if (!sel) return;

  const current = sel.value;
  const data = await fetchJSON(withParams("/kpi/clips", { camera_id: cam }));
  const clips = (data && data.clips) ? data.clips : [];

  sel.innerHTML = `<option value="">All clips</option>`;

  if (!clips.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no clips)";
    sel.appendChild(opt);
    return;
  }

  for (const cid of clips) {
    const opt = document.createElement("option");
    opt.value = cid;
    opt.textContent = cid;
    sel.appendChild(opt);
  }

  if (initialClipPending && INITIAL_CLIP && clips.includes(INITIAL_CLIP)) {
    sel.value = INITIAL_CLIP;
  } else if (current && clips.includes(current)) {
    sel.value = current;
  } else {
    sel.value = "";
  }
  initialClipPending = false;

  const btn  = document.getElementById("btnCheckQuality");
  const btn2 = document.getElementById("btnRunPipeline");
  if (btn)  btn.disabled  = !sel.value;
  if (btn2) btn2.disabled = !sel.value;
}

function buildZoneMetrics(zoneStats) {
  const visits = toNum(zoneStats?.visits, 0);
  const dwell = toNum(zoneStats?.dwell_s, 0);
  const avg = toNum(zoneStats?.avg_dwell_s, visits > 0 ? dwell / visits : 0);

  let qualified = zoneStats?.qualified_visits;
  if (qualified === undefined || qualified === null || qualified === "") {
    qualified = visits;
  }
  qualified = toNum(qualified, 0);

  let transit = zoneStats?.transit_visits;
  if (transit === undefined || transit === null || transit === "") {
    transit = Math.max(0, visits - qualified);
  }
  transit = toNum(transit, 0);

  let rate = zoneStats?.qualification_rate;
  if (rate === undefined || rate === null || rate === "") {
    rate = visits > 0 ? (qualified / visits) * 100 : 0;
  } else {
    rate = Number(rate);
    if (Number.isFinite(rate) && rate <= 1) {
      rate = rate * 100;
    }
  }
  rate = Number.isFinite(rate) ? rate : 0;

  return {
    visits,
    qualified,
    transit,
    rate,
    dwell,
    avg
  };
}

async function loadTables() {
  const params = getParams();
  const cameraId = String(params.camera_id || "cam1");

  const [summaryData, zoneData, personData, zonesDef] = await Promise.all([
    fetchJSON(withParams("/kpi/camera-summary", params)),
    fetchJSON(withParams("/kpi/zone-summary", params)),
    fetchJSON(withParams("/kpi/person-summary", params)),
    fetchJSON(withParams("/zones", { camera_id: cameraId })),
  ]);

  setSummaryCards(summaryData?.cards || []);

  const zoneTbody = document.querySelector("#zone-table tbody");
  if (zoneTbody) {
    zoneTbody.innerHTML = "";

    const perZone = zoneData?.per_zone || {};
    const zoneDefs = Array.isArray(zonesDef) ? zonesDef : [];
    for (const z of zoneDefs) {
      const zid = String(z?.zone_id || "").trim();
      if (!zid) continue;
      if (!Object.prototype.hasOwnProperty.call(perZone, zid)) {
        perZone[zid] = {
          zone_id: zid,
          visits: 0,
          qualified_visits: 0,
          transit_visits: 0,
          qualification_rate: 0,
          dwell_s: 0,
          avg_dwell_s: 0
        };
      }
    }
    const zoneEntries = Object.entries(perZone).sort((a, b) => a[0].localeCompare(b[0]));

    if (!zoneEntries.length) {
      const _zoneEmptyMsg = currentCamera() === "uploaded"
        ? "No zones configured yet — per-person track data is shown in the table below. Draw zones in the Zone Editor to add zone-level analytics."
        : "No zone events found. Assign zones and ingest KPIs first.";
      renderEmptyRow(zoneTbody, 6, _zoneEmptyMsg);
    } else {
      for (const [zid, s] of zoneEntries) {
        const m = buildZoneMetrics(s);

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${zid}</td>
          <td>${m.visits}</td>
          <td>${m.qualified}</td>
          <td>${m.transit}</td>
          <td>${fmtPct(m.rate)}</td>
          <td>${m.avg.toFixed(2)}</td>
        `;
        zoneTbody.appendChild(tr);
      }
    }
  }

  const personTbody = document.querySelector("#person-table tbody");
  if (personTbody) {
    personTbody.innerHTML = "";

    const personEntries = Object.entries(personData || {}).sort(
      (a, b) => toNum(a[0], 0) - toNum(b[0], 0)
    );

    if (!personEntries.length) {
      const _personEmptyMsg = currentCamera() === "uploaded"
        ? "No track data available yet — run the pipeline first."
        : "No per-person events found for this selection.";
      renderEmptyRow(personTbody, 4, _personEmptyMsg);
    } else {
      for (const [pid, s] of personEntries) {
        const visits = toNum(s?.visits, 0);
        const dwell = toNum(s?.total_dwell, 0);
        const frames = toNum(s?.total_frames, 0);

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${pid}</td>
          <td>${visits}</td>
          <td>${dwell.toFixed(2)}</td>
          <td>${frames}</td>
        `;
        personTbody.appendChild(tr);
      }
    }
  }
}

async function loadInsights() {
  const box = document.getElementById("insightsBox");
  if (!box) return;

  const params = getParams();
  const data = await fetchJSON(withParams("/kpi/insights", params));

  box.innerHTML = "";
  const items = data?.insights || [];

  if (!items.length) {
    box.innerHTML = `<div class="insight info"><b>[INFO] No insights</b><div class="msg">No insight items returned.</div></div>`;
    return;
  }

  for (const it of items) {
    const sev = String(it.severity || "info").toLowerCase();
    const div = document.createElement("div");
    div.className = `insight ${sev}`;
    div.innerHTML = `<b>[${sev.toUpperCase()}] ${it.title}</b><div class="msg">${it.message}</div>`;
    box.appendChild(div);
  }
}

async function loadFreshness() {
  const cam = document.getElementById("cameraSelect")?.value || "cam1";
  const [camData, allData] = await Promise.all([
    fetchJSON(withParams("/health/data-freshness", { camera_id: cam })),
    fetchJSON("/health/data-freshness")
  ]);

  const c = Array.isArray(camData?.checks) ? camData.checks[0] : null;
  const overall = String(allData?.status || "ok").toLowerCase();

  if (!c && overall === "ok") {
    setFreshnessNotice("ok", "");
    return;
  }

  const level = (c?.status || overall || "ok").toLowerCase();
  const msgs = Array.isArray(c?.messages) ? c.messages : [];
  const actions = Array.isArray(c?.actions) ? c.actions : [];

  let text = "";
  if (level === "error") {
    text = `Data freshness error (${cam}). `;
  } else if (level === "warn") {
    text = `Data freshness warning (${cam}). `;
  }

  if (msgs.length) text += msgs.join(" ");
  if (actions.length) text += ` Action: ${actions[0]}`;

  setFreshnessNotice(level, text.trim());
}

async function loadABInsights() {
  const tbody = document.querySelector("#ab-table tbody");
  const box = document.getElementById("abInsightsBox");
  if (!tbody || !box) throw new Error("Comparison HTML missing.");

  const pair = getComparisonPair();
  const data = await fetchJSON(withParams("/kpi/ab-insights", {
    cam_a: pair.camA,
    cam_b: pair.camB
  }));

  tbody.innerHTML = "";
  const summary = data?.summary;

  if (!summary?.A || !summary?.B) {
    renderEmptyRow(tbody, 4, "No comparison summary available.");
    return;
  }

  const A = summary.A;
  const B = summary.B;

  const rows = [
    ["Persons", A.n_persons, B.n_persons, "—"],
    [
      "Visits",
      A.visits,
      B.visits,
      fmtDeltaPct(((B.visits - A.visits) / (A.visits || 1)) * 100)
    ],
    [
      "Total dwell (s)",
      Number(A.dwell_s || 0).toFixed(2),
      Number(B.dwell_s || 0).toFixed(2),
      fmtDeltaPct(((B.dwell_s - A.dwell_s) / (A.dwell_s || 1)) * 100)
    ],
    [
      "Avg dwell (s)",
      Number(A.avg_dwell_s || 0).toFixed(2),
      Number(B.avg_dwell_s || 0).toFixed(2),
      fmtDeltaPct(((B.avg_dwell_s - A.avg_dwell_s) / (A.avg_dwell_s || 1)) * 100)
    ],
  ];

  for (const r of rows) {
    const cls = deltaClass(r[3]);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="${cls}">${r[3]}</td>`;
    tbody.appendChild(tr);
  }

  box.innerHTML = "";
  const items = data?.insights || [];

  if (!items.length) {
    box.innerHTML = `<div class="insight info"><b>[INFO] No comparison insights</b><div class="msg">No comparison insight items returned.</div></div>`;
    return;
  }

  for (const it of items) {
    const sev = String(it.severity || "info").toLowerCase();
    const div = document.createElement("div");
    div.className = `insight ${sev}`;
    div.innerHTML = `<b>[${sev.toUpperCase()}] ${it.title}</b><div class="msg">${it.message}</div>`;
    box.appendChild(div);
  }
}

function _humanCondition(key) {
  // Keys are produced by quality_summary() via reason.split("=")[0]
  const map = {
    white_ratio:   "washed out",
    mean_gray:     "overexposed",
    std_gray:      "low texture",
    null_or_empty: "blank frame",
  };
  return map[key] || String(key).replace(/_/g, " ");
}

function _renderPreprocessBanner(data, clip) {
  const banner = document.getElementById("preprocessBanner");
  if (!banner) return;

  banner.style.display = "block";

  if (!data?.found || !data?.report) {
    banner.className = "not-found";
    banner.innerHTML =
      `<div class="pp-label">Stage 1 · Preprocessing</div>` +
      `No quality report for <strong>${clip}</strong> yet — run Check Quality first.`;
    return;
  }

  const r = data.report;
  const q = r.quality || {};
  const [w, h] = Array.isArray(r.resolution) ? r.resolution : [null, null];
  const res   = (w && h) ? `${w}×${h}` : "–";
  const fps   = Number.isFinite(Number(r.fps)) ? Number(r.fps).toFixed(2) : "–";
  const dur   = Number.isFinite(Number(r.duration_s)) ? Number(r.duration_s).toFixed(1) : "–";
  const total = q.total_frames ?? r.frame_count ?? "–";
  const bad   = q.bad_frames  ?? 0;
  const pct   = q.bad_pct     ?? 0;

  const reasonEntries = Object.entries(q.reasons || {}).sort((a, b) => b[1] - a[1]);
  const dominant = reasonEntries.length ? reasonEntries[0][0] : null;
  const condStr  = dominant ? ` · ${_humanCondition(dominant)}` : "";

  const badClass  = Number(pct) === 0 ? "pp-bad clean" : "pp-bad";
  const pctFmt    = Number(pct) === 0 ? "0%" : `${pct}%`;
  const badStr    = `${bad} / ${total} (${pctFmt})${condStr}`;

  banner.className = "found";
  banner.innerHTML =
    `<div class="pp-label">Stage 1 · Preprocessing</div>` +
    `<div class="pp-row">` +
      `<span class="pp-item"><span>${r.video || clip}</span></span>` +
      `<span class="pp-item">Resolution: <span>${res}</span></span>` +
      `<span class="pp-item">FPS: <span>${fps}</span></span>` +
      `<span class="pp-item">Frames: <span>${total}</span></span>` +
      `<span class="pp-item">Duration: <span>${dur} s</span></span>` +
      `<span class="${badClass}">Bad frames: ${badStr}</span>` +
    `</div>`;
}

async function loadPreprocessReport() {
  const banner = document.getElementById("preprocessBanner");
  if (!banner) return;

  const clip = document.getElementById("clipSelect")?.value || "";
  if (!clip) {
    banner.style.display = "none";
    banner.className = "";
    banner.innerHTML = "";
    return;
  }

  let data;
  try {
    data = await fetchJSON(withParams("/meta/preprocess-report", { clip_id: clip }));
  } catch {
    banner.style.display = "none";
    return;
  }

  _renderPreprocessBanner(data, clip);
}

async function runPreprocessCheck() {
  const banner = document.getElementById("preprocessBanner");
  const btn    = document.getElementById("btnCheckQuality");
  const clip   = document.getElementById("clipSelect")?.value || "";

  if (!clip) {
    if (banner) {
      banner.style.display = "block";
      banner.className = "not-found";
      banner.innerHTML = `<div class="pp-label">Stage 1 · Preprocessing</div>Select a clip first.`;
    }
    return;
  }

  if (btn) { btn.disabled = true; btn.textContent = "Scanning…"; }
  if (banner) {
    banner.style.display = "block";
    banner.className = "";
    banner.innerHTML = `<div class="pp-label">Stage 1 · Preprocessing</div>Scanning <strong>${clip}</strong>…`;
  }

  try {
    const data = await fetchJSON("/pipeline/preprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clip }),
    });
    _renderPreprocessBanner(data, clip);
  } catch (e) {
    if (banner) {
      banner.style.display = "block";
      banner.className = "not-found";
      banner.innerHTML =
        `<div class="pp-label">Stage 1 · Preprocessing</div>` +
        `Scan failed for <strong>${clip}</strong>: ${e.message || "unexpected error"}.`;
    }
  } finally {
    const curClip = document.getElementById("clipSelect")?.value || "";
    if (btn) { btn.disabled = !curClip; btn.textContent = "Check Quality"; }
  }
}

let _pipelineJobId = null;
let _pipelinePollTimer = null;

function _escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function _cleanLogTail(raw) {
  if (!raw) return "";
  return raw.split("\n")
    .filter(l => {
      const t = l.trim();
      return t && !t.startsWith("warnings.warn(") && !t.includes("UserWarning");
    })
    .slice(-12)
    .join("\n");
}

function _renderPipelineBanner(state, msg, logTail) {
  const banner = document.getElementById("pipelineBanner");
  if (!banner) return;
  banner.style.display = "block";
  banner.className = state;
  let html = `<div class="pl-label">Stages 2–4 · Pipeline Run</div>${msg}`;
  if (logTail) {
    html += `<div class="pl-log">${_escHtml(logTail)}</div>`;
  }
  banner.innerHTML = html;
}

async function _pollJobStatus(jobId, clipId) {
  const btn = document.getElementById("btnRunPipeline");
  try {
    const data = await fetchJSON(`/pipeline/status/${encodeURIComponent(jobId)}`);
    const status  = data?.status || "unknown";
    const elapsed = data?.elapsed_s ?? "…";
    // log_tail is a string from the API; guard against null/undefined
    const tail    = _cleanLogTail(data?.log_tail || "");

    if (status === "running") {
      _renderPipelineBanner(
        "running",
        `Running pipeline for <strong>${clipId}</strong>… (${elapsed} s elapsed)`,
        tail || null
      );
      _pipelinePollTimer = setTimeout(() => _pollJobStatus(jobId, clipId), 3000);
    } else if (status === "done") {
      _pipelineJobId = null;
      const _doneCam = currentCamera();
      const _doneUrl = `/ui?camera_id=${encodeURIComponent(_doneCam)}&clip_id=${encodeURIComponent(clipId)}`;
      _renderPipelineBanner(
        "done",
        `Pipeline complete · <strong>${clipId}</strong> · ${elapsed} s · ` +
          `<a href="${_doneUrl}" ` +
          `style="color:#1e40af;font-weight:700;">View Results →</a>`,
        null  // log tail not useful once done; KPI tables refresh below
      );
      if (btn) { btn.disabled = false; btn.textContent = "Run Pipeline"; }
      await refreshAll();
    } else {
      _pipelineJobId = null;
      const errDetail = data?.error ? `: ${_escHtml(data.error)}` : "";
      _renderPipelineBanner(
        "error",
        `Pipeline failed for <strong>${clipId}</strong>${errDetail}`,
        tail || null  // keep log tail on error so user can diagnose
      );
      if (btn) { btn.disabled = false; btn.textContent = "Run Pipeline"; }
    }
  } catch {
    _pipelinePollTimer = setTimeout(() => _pollJobStatus(jobId, clipId), 5000);
  }
}

async function runPipeline() {
  const banner = document.getElementById("pipelineBanner");
  const btn    = document.getElementById("btnRunPipeline");
  const clip   = document.getElementById("clipSelect")?.value || "";

  if (!clip) {
    if (banner) {
      banner.style.display = "block";
      banner.className = "error";
      banner.innerHTML = `<div class="pl-label">Stages 2–4 · Pipeline Run</div>Select a clip first.`;
    }
    return;
  }

  if (_pipelinePollTimer) { clearTimeout(_pipelinePollTimer); _pipelinePollTimer = null; }

  if (btn) { btn.disabled = true; btn.textContent = "Running…"; }
  _renderPipelineBanner("running", `Starting pipeline for <strong>${clip}</strong>…`, null);

  try {
    const data = await fetchJSON("/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clip }),
    });

    _pipelineJobId = data.job_id;
    _pollJobStatus(data.job_id, clip);
  } catch (e) {
    _renderPipelineBanner(
      "error",
      `Could not start pipeline for <strong>${clip}</strong>: ${_escHtml(e.message || "unexpected error")}.`,
      null
    );
    const curClip = document.getElementById("clipSelect")?.value || "";
    if (btn) { btn.disabled = !curClip; btn.textContent = "Run Pipeline"; }
  }
}

async function refreshAll() {
  try {
    setBusy(true);
    setError("");
    updateStaticTexts();
    updateModePanels();

    await loadGroups();
    await loadClips();
    await loadTables();
    await loadInsights();
    await loadFreshness();
    await loadPreprocessReport();
  } catch (e) {
    console.error("Dashboard error:", e);
    setError("Failed to load KPI data. Check API.");
  } finally {
    setBusy(false);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadCameras();
  await refreshAll();

  document.getElementById("cameraSelect")?.addEventListener("change", async () => {
    updateZoneLink();
    const groupSel = document.getElementById("groupSelect");
    const clipSel  = document.getElementById("clipSelect");
    if (groupSel) groupSel.value = "";
    if (clipSel)  clipSel.value  = "";
    const btn  = document.getElementById("btnCheckQuality");
    const btn2 = document.getElementById("btnRunPipeline");
    if (btn)  btn.disabled  = true;
    if (btn2) btn2.disabled = true;
    updateStaticTexts();
    updateModePanels();
    await refreshAll();
  });

  document.getElementById("groupSelect")?.addEventListener("change", async () => {
    const clipSel = document.getElementById("clipSelect");
    if (clipSel) clipSel.value = "";
    await loadTables();
    await loadInsights();
  });

  document.getElementById("clipSelect")?.addEventListener("change", async () => {
    const clip = document.getElementById("clipSelect")?.value || "";
    const btn  = document.getElementById("btnCheckQuality");
    const btn2 = document.getElementById("btnRunPipeline");
    if (btn)  btn.disabled  = !clip;
    if (btn2) btn2.disabled = !clip;
    await loadTables();
    await loadInsights();
    await loadPreprocessReport();
  });

  document.getElementById("btnCheckQuality")?.addEventListener("click", async () => {
    await runPreprocessCheck();
  });

  document.getElementById("btnRunPipeline")?.addEventListener("click", async () => {
    await runPipeline();
  });

  document.getElementById("btnInsights")?.addEventListener("click", async () => {
    try {
      setBusy(true);
      await loadInsights();
    } finally {
      setBusy(false);
    }
  });

  document.getElementById("btnAB")?.addEventListener("click", async () => {
    try {
      setBusy(true);
      await loadABInsights();
    } catch (e) {
      console.error(e);
      setError("Failed to load comparison insights.");
    } finally {
      setBusy(false);
    }
  });
});
