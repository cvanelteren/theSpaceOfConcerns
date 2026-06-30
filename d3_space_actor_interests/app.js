async function main() {
  const [spaceData, ribbonData] = await Promise.all([
    d3.json("./data/space_actor_interests.json"),
    d3.json("../d3_ribbon_plot/data/fig2_ribbon_data.json"),
  ]);

  const STORAGE_KEY = "ats-space-actor-interests-ui";
  const MORPH_DURATION = 720;

  const svg = d3.select("#chart");
  const appLayout = document.getElementById("app-layout");
  const graphPanel = document.getElementById("graph-panel");
  const viewModeRoot = document.getElementById("view-mode");
  const viewModeButtons = Array.from(viewModeRoot.querySelectorAll("[data-view-mode]"));

  const actorOverlayCard = document.getElementById("actor-overlay-card");
  const selectedActorCard = document.getElementById("selected-actor-card");
  const ribbonControlsCard = document.getElementById("ribbon-controls-card");
  const layoutDisplayCard = document.getElementById("layout-display-card");

  const layoutModeSelect = document.getElementById("layout-mode");
  const resetLayoutButton = document.getElementById("reset-layout");
  const nodeSizeScaleInput = document.getElementById("node-size-scale");
  const nodeSizeScaleValue = document.getElementById("node-size-scale-value");
  const layoutSpreadInput = document.getElementById("layout-spread");
  const layoutSpreadValue = document.getElementById("layout-spread-value");
  const edgePercentileInput = document.getElementById("edge-percentile");
  const edgePercentileValue = document.getElementById("edge-percentile-value");
  const edgeThresholdNote = document.getElementById("edge-threshold-note");

  const actorSearch = document.getElementById("actor-search");
  const actorSelect = document.getElementById("actor-select");
  const prevActorButton = document.getElementById("prev-actor");
  const nextActorButton = document.getElementById("next-actor");
  const thresholdInput = document.getElementById("rpa-threshold");
  const thresholdValue = document.getElementById("rpa-threshold-value");
  const showAllLabelsInput = document.getElementById("show-all-labels");
  const showSelectedLabelsInput = document.getElementById("show-selected-labels");
  const showBaseSupportInput = document.getElementById("show-base-support");

  const ribbonThemeSelect = document.getElementById("ribbon-theme-select");
  const ribbonCountrySelect = document.getElementById("ribbon-country-select");
  const ribbonMinFlowInput = document.getElementById("ribbon-min-flow");
  const ribbonMinFlowValue = document.getElementById("ribbon-min-flow-value");
  const ribbonShowLabelsInput = document.getElementById("ribbon-show-labels");
  const ribbonResetButton = document.getElementById("ribbon-reset");
  const ribbonStatus = document.getElementById("ribbon-status");

  const actorCard = document.getElementById("actor-card");
  const topicCard = document.getElementById("topic-card");
  const legend = document.getElementById("legend");
  const meta = document.getElementById("meta");

  const themeOrder = Object.keys(spaceData.theme_colors);
  const supportMax = d3.max(spaceData.nodes, (d) => d.support_count) || 1;
  const degreeMax = d3.max(spaceData.nodes, (d) => d.weighted_degree || d.degree || 1) || 1;
  const strongEdgeWeights = spaceData.links
    .filter((d) => d.kind !== "mst")
    .map((d) => +d.weight)
    .sort(d3.ascending);

  const spaceActorById = new Map(spaceData.actors.map((actor) => [actor.id, actor]));
  const allActors = [{ id: "__all__", label: "All actors (aggregate support)" }].concat(
    spaceData.actors.map((actor) => ({ id: actor.id, label: actor.id }))
  );
  allActors.forEach((actor) => {
    const option = document.createElement("option");
    option.value = actor.id;
    option.textContent = actor.label;
    actorSelect.appendChild(option);
  });
  actorSelect.value = "__all__";

  legend.innerHTML = "";
  Object.entries(spaceData.theme_colors).forEach(([theme, color]) => {
    const row = document.createElement("div");
    row.className = "legend-item";
    row.innerHTML = `<span class="swatch" style="background:${color}"></span><span>${theme}</span>`;
    legend.appendChild(row);
  });

  const ribbonPeriods = ribbonData.periods;
  const ribbonTopics = ribbonData.topics_order;
  const ribbonThemes = ribbonData.themes_order;
  const ribbonThemeColors = ribbonData.theme_colors;
  const ribbonNodes = ribbonData.nodes.map((d) => ({ ...d }));
  const ribbonLinks = ribbonData.links.map((d) => ({ ...d }));
  const ribbonNodeById = new Map(ribbonNodes.map((d) => [d.id, d]));
  const ribbonTopicMeta = new Map();
  ribbonNodes.forEach((node) => {
    if (!ribbonTopicMeta.has(node.topic)) {
      ribbonTopicMeta.set(node.topic, { theme: node.theme, color: node.color });
    }
  });
  const ribbonTopicIndex = new Map(ribbonTopics.map((topic, idx) => [topic, idx]));
  const maxRibbonNodeCount = d3.max(ribbonNodes, (d) => d.count) || 1;
  const maxRibbonLinkValue = d3.max(ribbonLinks, (d) => d.value) || 1;
  const ribbonCountWidth = ribbonData.meta?.edge_width_metric === "actor_count";
  const fmtFlow = ribbonCountWidth ? d3.format(",d") : d3.format(".2f");
  const fmtSupport = d3.format(".2f");
  const fmtInt = d3.format(",d");

  const ribbonCountryPaths = ribbonData.country_paths || {};
  const ribbonNodeCountries = new Map(
    ribbonNodes.map((node) => [node.id, Array.from(new Set(node.countries || [])).sort()])
  );
  if ([...ribbonNodeCountries.values()].every((arr) => arr.length === 0)) {
    Object.entries(ribbonCountryPaths).forEach(([country, path]) => {
      path.forEach((step) => {
        const id = `${step.period}::${step.topic}`;
        if (ribbonNodeCountries.has(id)) ribbonNodeCountries.get(id).push(country);
      });
    });
    for (const [id, arr] of ribbonNodeCountries.entries()) {
      ribbonNodeCountries.set(id, Array.from(new Set(arr)).sort());
    }
  }

  const ribbonCountryList = Object.keys(ribbonCountryPaths).sort((a, b) => a.localeCompare(b));
  ribbonThemeSelect.innerHTML = "";
  ["All themes", ...ribbonThemes].forEach((theme) => {
    const option = document.createElement("option");
    option.value = theme;
    option.textContent = theme;
    ribbonThemeSelect.appendChild(option);
  });
  ribbonCountrySelect.innerHTML = "";
  ["All countries", ...ribbonCountryList].forEach((country) => {
    const option = document.createElement("option");
    option.value = country;
    option.textContent = country;
    ribbonCountrySelect.appendChild(option);
  });
  const ribbonFlowStep = ribbonCountWidth
    ? 1
    : maxRibbonLinkValue <= 5
      ? 0.05
      : maxRibbonLinkValue <= 20
        ? 0.1
        : 0.5;
  ribbonMinFlowInput.min = "0";
  ribbonMinFlowInput.max = String(maxRibbonLinkValue);
  ribbonMinFlowInput.step = String(ribbonFlowStep);
  ribbonMinFlowInput.value = "0";
  ribbonMinFlowValue.textContent = ribbonCountWidth ? "0" : "0.00";

  const state = {
    actorId: "__all__",
    threshold: Number(thresholdInput.value),
    showAllLabels: showAllLabelsInput.checked,
    showSelectedLabels: showSelectedLabelsInput.checked,
    showBaseSupport: showBaseSupportInput.checked,
    viewMode: "space",
    layoutMode: "space",
    nodeSizeScale: 1.35,
    layoutSpread: 1.0,
    edgePercentile: 95,
    ribbon: {
      theme: "All themes",
      country: "All countries",
      minFlow: 0,
      showLabels: true,
      hoverTopic: null,
      lockedTopics: new Set(),
    },
  };

  function readUiState() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) || {} : {};
    } catch (_) {
      return {};
    }
  }

  function persistUiState() {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          viewMode: state.viewMode,
          layoutMode: state.layoutMode,
          nodeSizeScale: state.nodeSizeScale,
          layoutSpread: state.layoutSpread,
          edgePercentile: state.edgePercentile,
        })
      );
    } catch (_) {
      // Ignore storage issues.
    }
  }

  function applyPersistedUiState() {
    const saved = readUiState();
    if (saved.viewMode === "space" || saved.viewMode === "ribbon") state.viewMode = saved.viewMode;
    if (["space", "columns", "ring"].includes(saved.layoutMode)) state.layoutMode = saved.layoutMode;
    if (Number.isFinite(saved.nodeSizeScale) && saved.nodeSizeScale >= 0.7 && saved.nodeSizeScale <= 2.4) {
      state.nodeSizeScale = saved.nodeSizeScale;
    }
    if (Number.isFinite(saved.layoutSpread) && saved.layoutSpread >= 0.7 && saved.layoutSpread <= 1.8) {
      state.layoutSpread = saved.layoutSpread;
    }
    if (Number.isFinite(saved.edgePercentile) && saved.edgePercentile >= 0 && saved.edgePercentile <= 100) {
      state.edgePercentile = saved.edgePercentile;
    }

    layoutModeSelect.value = state.layoutMode;
    nodeSizeScaleInput.value = state.nodeSizeScale.toFixed(2);
    layoutSpreadInput.value = state.layoutSpread.toFixed(2);
    edgePercentileInput.value = String(Math.round(state.edgePercentile));
  }

  function extraEdgeThreshold(percentile) {
    if (!strongEdgeWeights.length) return 0;
    const q = Math.max(0, Math.min(1, percentile / 100));
    return d3.quantileSorted(strongEdgeWeights, q) ?? strongEdgeWeights[0];
  }

  function updateSpaceControlReadout() {
    nodeSizeScaleValue.textContent = `${state.nodeSizeScale.toFixed(2)}x`;
    layoutSpreadValue.textContent = `${state.layoutSpread.toFixed(2)}x`;
    edgePercentileValue.textContent = `${Math.round(state.edgePercentile)}`;
    const threshold = extraEdgeThreshold(state.edgePercentile);
    edgeThresholdNote.textContent = `Strong edges shown from the ${Math.round(state.edgePercentile)}th percentile upward (>= ${threshold.toFixed(2)}); MST scaffold retained.`;
  }

  function setDefaultTopicCard() {
    topicCard.className = "topic-card muted";
    topicCard.innerHTML =
      state.viewMode === "space"
        ? "Hover a topic to inspect support and top actors."
        : "Hover ribbon bars, ribbons, or topic labels to inspect specialized support flows.";
  }

  function setTopicCard(html, muted = false) {
    topicCard.className = muted ? "topic-card muted" : "topic-card";
    topicCard.innerHTML = html;
  }

  function setRibbonStatus(text) {
    ribbonStatus.textContent = text;
  }

  function updateMeta() {
    if (state.viewMode === "space") {
      meta.textContent = `${spaceData.meta.n_topics} topics | ${spaceData.meta.n_links} links | ${spaceData.meta.n_actors} actors | default threshold ${spaceData.meta.default_rpa_threshold.toFixed(1)}`;
    } else {
      meta.textContent = `${ribbonData.meta.n_topics} topics | ${ribbonData.nodes.length} topic-period bars | ${ribbonData.links.length} ribbon links | RCA>${ribbonData.meta.rca_threshold}`;
    }
  }

  function applyViewChrome() {
    const showSpace = state.viewMode === "space";
    graphPanel.classList.toggle("ribbon-view", !showSpace);
    appLayout.classList.toggle("ribbon-mode", !showSpace);
    actorOverlayCard.hidden = !showSpace;
    selectedActorCard.hidden = !showSpace;
    layoutDisplayCard.hidden = !showSpace;
    ribbonControlsCard.hidden = showSpace;
    viewModeButtons.forEach((button) => {
      const active = button.dataset.viewMode === state.viewMode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    updateMeta();
    if (topicCard.classList.contains("muted")) setDefaultTopicCard();
  }

  const defs = svg.append("defs");
  defs
    .append("pattern")
    .attr("id", "ribbon-hatch-start")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
    .append("path")
    .attr("d", "M-1,1 l2,-2 M0,5 l5,-5 M4,6 l2,-2")
    .attr("stroke", "#111")
    .attr("stroke-width", 1.15)
    .attr("stroke-linecap", "round");
  defs
    .append("pattern")
    .attr("id", "ribbon-hatch-end")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
    .append("path")
    .attr("d", "M-1,4 l2,2 M0,0 l5,5 M4,-1 l2,2")
    .attr("stroke", "#111")
    .attr("stroke-width", 1.15)
    .attr("stroke-linecap", "round");

  const mainRoot = svg.append("g");
  const spaceLayers = {
    links: mainRoot.append("g").attr("class", "space-links"),
    supportLinks: mainRoot.append("g").attr("class", "space-support-links"),
    baseHalos: mainRoot.append("g").attr("class", "space-base-halos"),
    actorHalos: mainRoot.append("g").attr("class", "space-actor-halos"),
    nodes: mainRoot.append("g").attr("class", "space-nodes"),
    labels: mainRoot.append("g").attr("class", "space-labels"),
  };
  const ribbonRoot = mainRoot.append("g").attr("class", "ribbon-root");
  const ribbonTop = ribbonRoot.append("g").attr("class", "ribbon-top");
  const ribbonLayers = {
    periodLabels: ribbonTop.append("g").attr("class", "ribbon-period-labels"),
    guides: ribbonTop.append("g").attr("class", "ribbon-guides"),
    links: ribbonTop.append("g").attr("class", "ribbon-links"),
    nodes: ribbonTop.append("g").attr("class", "ribbon-nodes"),
    countryPath: ribbonTop.append("g").attr("class", "ribbon-country-path"),
    labels: ribbonTop.append("g").attr("class", "ribbon-topic-labels"),
  };

  let currentTransform = d3.zoomIdentity;
  let spaceLayout = null;
  let spaceById = new Map();
  let ribbonLayout = null;
  const ribbonGeomCache = { key: null, linkGeom: null };
  let ribbonInteractionRefresh = null;
  let ribbonInteractionRaf = null;

  function queueRibbonInteractionRefresh() {
    if (!ribbonInteractionRefresh || state.viewMode !== "ribbon") return;
    if (ribbonInteractionRaf != null) return;
    ribbonInteractionRaf = requestAnimationFrame(() => {
      ribbonInteractionRaf = null;
      if (ribbonInteractionRefresh && state.viewMode === "ribbon") {
        ribbonInteractionRefresh();
      }
    });
  }

  function clampX(width, x) {
    return Math.max(40, Math.min(width - 40, x));
  }

  function clampY(height, y) {
    return Math.max(36, Math.min(height - 40, y));
  }

  function applyLayoutSpread(nodes, width, height) {
    if (!nodes.length || state.layoutSpread === 1) return nodes;
    const xs = nodes.map((node) => node.sx);
    const ys = nodes.map((node) => node.sy);
    const cx = (d3.min(xs) + d3.max(xs)) / 2;
    const cy = (d3.min(ys) + d3.max(ys)) / 2;
    return nodes.map((node) => ({
      ...node,
      sx: clampX(width, cx + (node.sx - cx) * state.layoutSpread),
      sy: clampY(height, cy + (node.sy - cy) * state.layoutSpread),
    }));
  }

  function computeSpaceLayout(width, height) {
    const padding = { top: 36, right: 40, bottom: 40, left: 40 };
    const xExtent = d3.extent(spaceData.nodes, (d) => d.x);
    const yExtent = d3.extent(spaceData.nodes, (d) => d.y);
    const xScale = d3.scaleLinear().domain(xExtent).range([padding.left, width - padding.right]);
    const yScale = d3.scaleLinear().domain(yExtent).range([height - padding.bottom, padding.top]);

    let nodes;
    if (state.layoutMode === "space") {
      nodes = spaceData.nodes.map((node) => ({ ...node, sx: xScale(node.x), sy: yScale(node.y) }));
    } else if (state.layoutMode === "columns") {
      const xByTheme = d3.scalePoint().domain(themeOrder).range([padding.left + 34, width - padding.right - 34]);
      nodes = [];
      themeOrder.forEach((theme) => {
        const members = spaceData.nodes
          .filter((node) => node.theme === theme)
          .sort((a, b) => d3.descending(a.weighted_degree || a.degree || 0, b.weighted_degree || b.degree || 0) || d3.ascending(a.id, b.id));
        const yByIndex = d3.scalePoint().domain(d3.range(members.length)).range([padding.top + 36, height - padding.bottom - 36]);
        members.forEach((node, idx) => {
          nodes.push({ ...node, sx: xByTheme(theme), sy: yByIndex(idx) });
        });
      });
    } else {
      const cx = width / 2;
      const cy = height / 2;
      const outerRadius = Math.min(width, height) * 0.33;
      nodes = [];
      themeOrder.forEach((theme, themeIdx) => {
        const members = spaceData.nodes.filter((node) => node.theme === theme).sort((a, b) => d3.ascending(a.id, b.id));
        const angle = -Math.PI / 2 + (2 * Math.PI * themeIdx) / themeOrder.length;
        const centerX = cx + outerRadius * Math.cos(angle);
        const centerY = cy + outerRadius * Math.sin(angle);
        const innerRadius = Math.max(26, 10 + members.length * 2.2);
        members.forEach((node, idx) => {
          const localAngle = members.length === 1 ? 0 : -Math.PI / 2 + (2 * Math.PI * idx) / members.length;
          nodes.push({
            ...node,
            sx: clampX(width, centerX + innerRadius * Math.cos(localAngle)),
            sy: clampY(height, centerY + innerRadius * Math.sin(localAngle)),
          });
        });
      });
    }

    nodes = applyLayoutSpread(nodes, width, height);
    return { width, height, nodes, byId: new Map(nodes.map((node) => [node.id, node])) };
  }

  function ribbonPath(x0, y0, x1, y1, thickness, curvature = 0.5) {
    const dx = Math.max(x1 - x0, 1e-6);
    const cx0 = x0 + dx * curvature;
    const cx1 = x1 - dx * curvature;
    const top0 = y0 + thickness / 2;
    const bot0 = y0 - thickness / 2;
    const top1 = y1 + thickness / 2;
    const bot1 = y1 - thickness / 2;
    return [
      `M${x0},${top0}`,
      `C${cx0},${top0} ${cx1},${top1} ${x1},${top1}`,
      `L${x1},${bot1}`,
      `C${cx1},${bot1} ${cx0},${bot0} ${x0},${bot0}`,
      "Z",
    ].join(" ");
  }

  function nodeCornerRadius(height, width) {
    return Math.max(0.8, Math.min(1.8, height * 0.16, width * 0.12));
  }

  function makeRibbonLinkGeometry(xMap, yMap, nodeWidth, rowGap) {
    const linksByPair = d3.group(ribbonLinks, (d) => `${d.period0}||${d.period1}`);
    const out = [];

    linksByPair.forEach((pairLinks) => {
      const totalsBySrc = new Map();
      const totalsByTgt = new Map();
      pairLinks.forEach((link) => {
        totalsBySrc.set(link.source, (totalsBySrc.get(link.source) || 0) + link.value);
        totalsByTgt.set(link.target, (totalsByTgt.get(link.target) || 0) + link.value);
      });
      const maxTotal = d3.max([...totalsBySrc.values()]) || 1;
      const scale = (rowGap * 2.8 * 0.95) / maxTotal;

      const srcOffsets = new Map();
      const tgtOffsets = new Map();
      totalsBySrc.forEach((total, sourceId) => {
        srcOffsets.set(sourceId, yMap.get(ribbonNodeById.get(sourceId).topic) - (total * scale) / 2);
      });
      totalsByTgt.forEach((total, targetId) => {
        tgtOffsets.set(targetId, yMap.get(ribbonNodeById.get(targetId).topic) - (total * scale) / 2);
      });

      const sorted = [...pairLinks].sort((a, b) => {
        const sa = ribbonTopicIndex.get(a.source_topic);
        const sb = ribbonTopicIndex.get(b.source_topic);
        if (sa !== sb) return sa - sb;
        return ribbonTopicIndex.get(a.target_topic) - ribbonTopicIndex.get(b.target_topic);
      });

      const srcPos = new Map();
      sorted.forEach((link) => {
        const thick = link.value * scale;
        const start = srcOffsets.get(link.source) || 0;
        srcPos.set(link.id, { y: start + thick / 2, thickness: thick });
        srcOffsets.set(link.source, start + thick);
      });

      const sortedByTarget = [...sorted].sort((a, b) => {
        const ta = ribbonTopicIndex.get(a.target_topic);
        const tb = ribbonTopicIndex.get(b.target_topic);
        if (ta !== tb) return ta - tb;
        return ribbonTopicIndex.get(a.source_topic) - ribbonTopicIndex.get(b.source_topic);
      });
      const tgtPos = new Map();
      sortedByTarget.forEach((link) => {
        const thick = link.value * scale;
        const start = tgtOffsets.get(link.target) || 0;
        tgtPos.set(link.id, { y: start + thick / 2, thickness: thick });
        tgtOffsets.set(link.target, start + thick);
      });

      sorted.forEach((link) => {
        const srcNode = ribbonNodeById.get(link.source);
        const tgtNode = ribbonNodeById.get(link.target);
        const sx = xMap.get(srcNode.period) + nodeWidth / 2;
        const tx = xMap.get(tgtNode.period) - nodeWidth / 2;
        const sy = srcPos.get(link.id).y;
        const ty = tgtPos.get(link.id).y;
        const thickness = srcPos.get(link.id).thickness;
        out.push({
          ...link,
          sx,
          tx,
          sy,
          ty,
          thickness,
          path: ribbonPath(sx, sy, tx, ty, thickness, 0.5),
        });
      });
    });

    return out;
  }

  function computeRibbonLayout(width, height) {
    const margin = { top: 26, right: 250, bottom: 28, left: 92 };
    const topHeight = height - margin.bottom;
    const rightNodePad = 17;
    const x = d3.scalePoint().domain(ribbonPeriods).range([margin.left, width - margin.right - rightNodePad]);
    const y = d3.scalePoint().domain(ribbonTopics).range([margin.top + 26, topHeight - 10]);
    const rowGap = ribbonTopics.length > 1 ? Math.abs(y(ribbonTopics[1]) - y(ribbonTopics[0])) : topHeight * 0.5;
    const nodeWidth = Math.max(12, Math.min(20, (width - margin.left - margin.right) / (ribbonPeriods.length * 4.6)));
    const nodeScale = (rowGap * 2.8) / maxRibbonNodeCount;
    const minNodeHeight = 2.4;
    const rightLabelX = width - margin.right + 12;
    const anchorX = rightLabelX - 18;

    const cacheKey = [width, height, nodeWidth, rowGap].join("|");
    let linkGeom = ribbonGeomCache.linkGeom;
    if (!linkGeom || ribbonGeomCache.key !== cacheKey) {
      linkGeom = makeRibbonLinkGeometry(
        new Map(ribbonPeriods.map((period) => [period, x(period)])),
        new Map(ribbonTopics.map((topic) => [topic, y(topic)])),
        nodeWidth,
        rowGap
      );
      ribbonGeomCache.key = cacheKey;
      ribbonGeomCache.linkGeom = linkGeom;
    }

    return {
      width,
      height,
      margin,
      topHeight,
      rightLabelX,
      anchorX,
      x,
      y,
      rowGap,
      nodeWidth,
      nodeScale,
      minNodeHeight,
      linkGeom,
    };
  }

  function getCurrentActor() {
    return state.actorId === "__all__" ? null : spaceActorById.get(state.actorId);
  }

  function getCurrentPlacements() {
    const actor = getCurrentActor();
    if (!actor) return [];
    return actor.topics.filter((topic) => topic.rpa >= state.threshold);
  }

  function renderActorCard(actor, placements) {
    if (!actor) {
      actorCard.innerHTML = `
        <div class="muted">Showing aggregate specialized support density.</div>
        <div style="margin-top:8px">Outer rings indicate how many actors exceed RPA &gt; 1 in each topic.</div>
      `;
      return;
    }

    const topicItems = placements
      .slice()
      .sort((a, b) => d3.descending(a.rpa, b.rpa))
      .slice(0, 8)
      .map(
        (rec) => `
          <div class="topic-item">
            <strong>${rec.topic}</strong>
            <span>${rec.theme}</span><br />
            <span>RPA ${rec.rpa.toFixed(2)}</span>
          </div>
        `
      )
      .join("");

    const icon = actor.icon_path ? `<img src="${actor.icon_path}" alt="${actor.id}" />` : "<div></div>";
    actorCard.innerHTML = `
      <div class="actor-head">
        ${icon}
        <div>
          <div class="actor-name">${actor.id}</div>
          <div class="actor-kind">${actor.kind}${actor.source_type ? ` | ${actor.source_type}` : ""}</div>
        </div>
      </div>
      <div class="metric-grid">
        <div class="metric"><div class="metric-label">Support size</div><div class="metric-value">${placements.length}</div></div>
        <div class="metric"><div class="metric-label">Dominant theme</div><div class="metric-value">${actor.dominant_theme}</div></div>
        <div class="metric"><div class="metric-label">Mean RPA</div><div class="metric-value">${actor.mean_rpa.toFixed(2)}</div></div>
        <div class="metric"><div class="metric-label">Max RPA</div><div class="metric-value">${actor.max_rpa.toFixed(2)}</div></div>
      </div>
      <div style="margin-bottom:8px"><strong>Centroid</strong>: (${actor.centroid.x.toFixed(1)}, ${actor.centroid.y.toFixed(1)})</div>
      <div><strong>Top specialized topics</strong></div>
      <div class="topic-list">${topicItems || '<div class="muted">No topics at this threshold.</div>'}</div>
    `;
  }

  function renderSpaceTopicCard(node) {
    const topActors = (node.top_actors || [])
      .slice(0, 5)
      .map((rec) => `<div>${rec.actor}: <strong>${rec.rpa.toFixed(2)}</strong></div>`)
      .join("");
    setTopicCard(`
      <div><strong>${node.id}</strong></div>
      <div class="muted">${node.theme}</div>
      <div>Actors with RPA &gt; 1: <strong>${node.support_count}</strong></div>
      <div>Mean specialized support: <strong>${node.mean_support_rpa.toFixed(2)}</strong></div>
      <div style="margin-top:8px"><strong>Top actors</strong></div>
      ${topActors || '<div class="muted">No actors above threshold.</div>'}
    `);
  }

  function renderRibbonLinkCard(link) {
    const sample = (link.countries || []).slice(0, 8).join(", ");
    setTopicCard(
      `<strong>${link.source_topic}</strong> → <strong>${link.target_topic}</strong><br>` +
        `Period: ${link.period0} → ${link.period1}<br>` +
        `Actors (edge width): ${fmtInt(link.actor_count ?? Math.round(link.value || 0))}<br>` +
        `Weighted support: ${fmtSupport(link.support_value ?? link.value)}<br>` +
        `Theme: ${link.source_theme}${link.source_theme === link.target_theme ? "" : ` → ${link.target_theme}`}<br>` +
        `Countries: ${sample}${(link.countries || []).length > 8 ? "…" : ""}`
    );
  }

  function renderRibbonNodeCard(node) {
    const countries = ribbonNodeCountries.get(node.id) || [];
    const sample = countries.slice(0, 10).join(", ");
    setTopicCard(
      `<strong>${node.topic}</strong><br>` +
        `Period: ${node.period}<br>` +
        `Actors with RCA≥1 in topic: ${fmtInt(node.count)}<br>` +
        `Theme: ${node.theme}<br>` +
        `Countries: ${sample}${countries.length > 10 ? "…" : ""}`
    );
  }

  function renderRibbonLabelCard(topic) {
    const metaForTopic = ribbonTopicMeta.get(topic) || { theme: "Unknown" };
    setTopicCard(
      `<strong>${topic}</strong><br>` +
        `Theme: ${metaForTopic.theme}<br>` +
        `Hover to preview trace, click to lock/unlock.`
    );
  }

  function transitionFor(animate) {
    return animate
      ? svg.transition().duration(MORPH_DURATION).ease(d3.easeCubicInOut)
      : null;
  }

  function withTransition(selection, transition, delay = 0) {
    return transition ? selection.transition(transition).delay(delay) : selection;
  }

  function currentRibbonInteraction() {
    const topicSelected = (link) => {
      if (state.ribbon.hoverTopic && (link.source_topic === state.ribbon.hoverTopic || link.target_topic === state.ribbon.hoverTopic)) {
        return true;
      }
      if (state.ribbon.lockedTopics.size > 0) {
        return state.ribbon.lockedTopics.has(link.source_topic) || state.ribbon.lockedTopics.has(link.target_topic);
      }
      return false;
    };

    const visibleLinks = ribbonLayout.linkGeom.filter((link) => {
      if (topicSelected(link)) return true;
      if (link.value < state.ribbon.minFlow) return false;
      if (state.ribbon.theme !== "All themes") {
        if (link.source_theme !== state.ribbon.theme && link.target_theme !== state.ribbon.theme) return false;
      }
      return true;
    });

    const visibleLinkIds = new Set(visibleLinks.map((link) => link.id));
    const activeNodeIds = new Set();
    const activeTopics = new Set();
    visibleLinks.forEach((link) => {
      activeNodeIds.add(link.source);
      activeNodeIds.add(link.target);
      activeTopics.add(link.source_topic);
      activeTopics.add(link.target_topic);
    });

    const highlightTopics = new Set([...state.ribbon.lockedTopics]);
    if (state.ribbon.hoverTopic) highlightTopics.add(state.ribbon.hoverTopic);
    const hasHighlight = highlightTopics.size > 0;
    const highlightLinkIds = new Set();
    const highlightNodeIds = new Set();
    if (hasHighlight) {
      visibleLinks.forEach((link) => {
        if (highlightTopics.has(link.source_topic) || highlightTopics.has(link.target_topic)) {
          highlightLinkIds.add(link.id);
          highlightNodeIds.add(link.source);
          highlightNodeIds.add(link.target);
        }
      });
      ribbonNodes.forEach((node) => {
        if (highlightTopics.has(node.topic)) highlightNodeIds.add(node.id);
      });
    }

    const countryNodeIds = new Set();
    const countryLinkIds = new Set();
    const countryPathPoints = [];
    const countryTopics = new Set();
    if (state.ribbon.country !== "All countries" && ribbonCountryPaths[state.ribbon.country]) {
      const path = [...ribbonCountryPaths[state.ribbon.country]].sort((a, b) => a.period_order - b.period_order);
      path.forEach((step) => {
        const id = `${step.period}::${step.topic}`;
        countryNodeIds.add(id);
        countryTopics.add(step.topic);
        if (ribbonNodeById.has(id)) {
          countryPathPoints.push({
            id,
            x: ribbonLayout.x(step.period),
            y: ribbonLayout.y(step.topic),
            rca: step.rca,
          });
        }
      });
      ribbonLinks.forEach((link) => {
        if ((link.countries || []).includes(state.ribbon.country)) countryLinkIds.add(link.id);
      });
    }

    return {
      visibleLinks,
      visibleLinkIds,
      activeNodeIds,
      activeTopics,
      highlightTopics,
      hasHighlight,
      highlightLinkIds,
      highlightNodeIds,
      countryNodeIds,
      countryLinkIds,
      countryPathPoints,
      countryTopics,
    };
  }

  function ribbonNodeVisible(node, interaction) {
    if (interaction.hasHighlight && interaction.highlightNodeIds.has(node.id)) return true;
    if (state.ribbon.theme !== "All themes" && node.theme !== state.ribbon.theme && !interaction.activeNodeIds.has(node.id)) {
      return false;
    }
    return true;
  }

  function renderSpaceScene(animate = false) {
    const transition = transitionFor(animate);
    const showSpace = state.viewMode === "space";
    const exitDelay = showSpace ? 0 : 180;
    const actor = getCurrentActor();
    const placements = getCurrentPlacements();
    const selectedIds = new Set(placements.map((rec) => rec.topic));
    const edgeThreshold = extraEdgeThreshold(state.edgePercentile);
    const visibleBaseLinks = spaceData.links.filter((link) => link.kind === "mst" || link.weight >= edgeThreshold);

    const anchorByTopic = new Map(
      ribbonTopics.map((topic) => [topic, { sx: ribbonLayout.anchorX, sy: ribbonLayout.y(topic) }])
    );
    const currentNodeTarget = (topicId) => (showSpace ? spaceById.get(topicId) : anchorByTopic.get(topicId));
    const nodeRadius = (node) => showSpace
      ? state.nodeSizeScale * (4 + 8 * ((node.weighted_degree || node.degree || 1) / degreeMax))
      : Math.max(2.6, state.nodeSizeScale * 2.8);

    const spaceLinkSel = spaceLayers.links
      .selectAll("line")
      .data(visibleBaseLinks, (d) => `${d.source}__${d.target}`)
      .join("line")
      .attr("stroke", (d) => (d.kind === "mst" ? "#536271" : "#98a9ba"))
      .attr("stroke-linecap", "round");
    withTransition(spaceLinkSel, transition)
      .attr("x1", (d) => spaceById.get(d.source).sx)
      .attr("y1", (d) => spaceById.get(d.source).sy)
      .attr("x2", (d) => spaceById.get(d.target).sx)
      .attr("y2", (d) => spaceById.get(d.target).sy)
      .attr("stroke-opacity", (d) => (showSpace ? (d.kind === "mst" ? 0.82 : 0.48) : 0))
      .attr("stroke-width", (d) => (d.kind === "mst" ? 1.7 : 1.0) + 2.5 * d.weight);

    const baseHaloSel = spaceLayers.baseHalos
      .selectAll("circle")
      .data(spaceLayout.nodes, (d) => d.id)
      .join("circle")
      .attr("fill", "#0f766e")
      .attr("stroke", "#0f766e")
      .attr("stroke-width", 1);
    withTransition(baseHaloSel, transition, exitDelay)
      .attr("cx", (d) => currentNodeTarget(d.id).sx)
      .attr("cy", (d) => currentNodeTarget(d.id).sy)
      .attr("r", (d) => state.nodeSizeScale * (3 + 14 * (d.support_count / supportMax)))
      .attr("fill-opacity", showSpace && !actor && state.showBaseSupport ? 0.08 : 0)
      .attr("stroke-opacity", showSpace && !actor && state.showBaseSupport ? 0.22 : 0);

    const nodeSel = spaceLayers.nodes
      .selectAll("circle")
      .data(spaceLayout.nodes, (d) => d.id)
      .join("circle")
      .attr("fill", (d) => d.color)
      .attr("stroke", "white")
      .on("mouseenter", (_, d) => {
        if (state.viewMode !== "space") return;
        renderSpaceTopicCard(d);
      })
      .on("mouseleave", () => {
        if (state.viewMode !== "space") return;
        setDefaultTopicCard();
      })
      .call(
        d3.drag()
          .on("start", (event) => {
            if (state.viewMode !== "space") return;
            event.sourceEvent.stopPropagation();
          })
          .on("drag", (event, d) => {
            if (state.viewMode !== "space") return;
            const scale = currentTransform?.k || 1;
            d.sx = clampX(spaceLayout.width, d.sx + event.dx / scale);
            d.sy = clampY(spaceLayout.height, d.sy + event.dy / scale);
            spaceById.set(d.id, d);
            renderSpaceScene(false);
          })
          .on("end", (event) => {
            if (state.viewMode !== "space") return;
            event.sourceEvent.stopPropagation();
          })
      );
    withTransition(nodeSel, transition, exitDelay)
      .attr("cx", (d) => currentNodeTarget(d.id).sx)
      .attr("cy", (d) => currentNodeTarget(d.id).sy)
      .attr("r", (d) => nodeRadius(d))
      .attr("stroke-width", showSpace ? 1.2 : 0.8)
      .attr("opacity", showSpace ? 1 : 0)
      .style("pointer-events", showSpace ? "auto" : "none")
      .style("cursor", showSpace ? "move" : "default");

    const supportLinks = actor
      ? spaceData.links.filter((link) => selectedIds.has(link.source) && selectedIds.has(link.target))
      : [];
    const supportSel = spaceLayers.supportLinks
      .selectAll("line")
      .data(supportLinks, (d) => `${d.source}__${d.target}`)
      .join("line")
      .attr("stroke", "#0f172a")
      .attr("stroke-linecap", "round");
    withTransition(supportSel, transition, exitDelay)
      .attr("x1", (d) => spaceById.get(d.source).sx)
      .attr("y1", (d) => spaceById.get(d.source).sy)
      .attr("x2", (d) => spaceById.get(d.target).sx)
      .attr("y2", (d) => spaceById.get(d.target).sy)
      .attr("stroke-opacity", showSpace && actor ? 0.22 : 0)
      .attr("stroke-width", (d) => 1.0 + 2.6 * d.weight);

    const rpaMax = d3.max(placements, (d) => d.rpa) || 1;
    const actorHaloSel = spaceLayers.actorHalos
      .selectAll("circle")
      .data(placements, (d) => d.topic)
      .join("circle")
      .attr("fill", (d) => d.color)
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", 2.2);
    withTransition(actorHaloSel, transition, exitDelay)
      .attr("cx", (d) => currentNodeTarget(d.topic).sx)
      .attr("cy", (d) => currentNodeTarget(d.topic).sy)
      .attr("r", (d) => state.nodeSizeScale * (8 + 18 * (d.rpa / rpaMax)))
      .attr("fill-opacity", showSpace && actor ? 0.14 : 0)
      .attr("stroke-opacity", showSpace && actor ? 0.95 : 0);

    const labels = spaceLayout.nodes.filter((node) => {
      if (!showSpace) return false;
      if (state.showAllLabels) return true;
      if (state.showSelectedLabels && selectedIds.has(node.id)) return true;
      return false;
    });
    const labelSel = spaceLayers.labels
      .selectAll("text")
      .data(labels, (d) => d.id)
      .join("text")
      .attr("class", "topic-label")
      .text((d) => d.id);
    withTransition(labelSel, transition)
      .attr("x", (d) => currentNodeTarget(d.id).sx + 8)
      .attr("y", (d) => currentNodeTarget(d.id).sy - 8)
      .attr("opacity", showSpace ? 1 : 0);

    renderActorCard(actor, placements);
  }

  function renderRibbonScene(animate = false) {
    const transition = transitionFor(animate);
    const showRibbon = state.viewMode === "ribbon";
    const initialInteraction = currentRibbonInteraction();

    ribbonRoot.style("pointer-events", showRibbon ? "auto" : "none");

    const periodSel = ribbonLayers.periodLabels
      .selectAll("text")
      .data(ribbonPeriods, (d) => d)
      .join("text")
      .attr("class", "period-label")
      .attr("text-anchor", "middle")
      .text((d) => d);
    withTransition(periodSel, transition, showRibbon ? 90 : 0)
      .attr("x", (d) => ribbonLayout.x(d))
      .attr("y", ribbonLayout.margin.top - 5)
      .attr("fill", "#111827")
      .attr("font-size", 15)
      .attr("font-weight", 600)
      .attr("opacity", showRibbon ? 1 : 0);

    const guideSel = ribbonLayers.guides
      .selectAll("line")
      .data(ribbonTopics, (d) => d)
      .join("line")
      .attr("stroke", "#6b7280")
      .attr("stroke-dasharray", "4 4");
    withTransition(guideSel, transition)
      .attr("x1", ribbonLayout.margin.left + ribbonLayout.nodeWidth * 0.35)
      .attr("x2", ribbonLayout.width - ribbonLayout.margin.right - 17 - ribbonLayout.nodeWidth * 0.35)
      .attr("y1", (d) => ribbonLayout.y(d))
      .attr("y2", (d) => ribbonLayout.y(d))
      .attr("stroke-width", (d) => (initialInteraction.hasHighlight && initialInteraction.highlightTopics.has(d) ? 1.2 : 0.9))
      .attr("opacity", showRibbon ? 0.22 : 0);

    const ribbonLinkSel = ribbonLayers.links
      .selectAll("path")
      .data(showRibbon ? ribbonLayout.linkGeom : [], (d) => d.id)
      .join("path")
      .attr("class", "link")
      .attr("d", (d) => d.path)
      .attr("fill", (d) => ribbonNodeById.get(d.source)?.color || "#999")
      .on("mouseenter", (_, d) => {
        if (state.viewMode !== "ribbon") return;
        state.ribbon.hoverTopic = null;
        renderRibbonLinkCard(d);
        queueRibbonInteractionRefresh();
      })
      .on("mouseleave", () => {
        if (state.viewMode !== "ribbon") return;
        setDefaultTopicCard();
        queueRibbonInteractionRefresh();
      });
    withTransition(ribbonLinkSel, transition, showRibbon ? 160 : 0)
      .attr("display", (d) => (showRibbon && initialInteraction.visibleLinkIds.has(d.id) ? null : "none"))
      .attr("opacity", (d) => {
        if (!showRibbon || !initialInteraction.visibleLinkIds.has(d.id)) return 0;
        const base = 0.10 + 0.70 * (d.value / maxRibbonLinkValue);
        if (state.ribbon.country !== "All countries") {
          const value = initialInteraction.countryLinkIds.has(d.id) ? Math.min(0.95, base + 0.2) : base * 0.10;
          return initialInteraction.hasHighlight
            ? initialInteraction.highlightLinkIds.has(d.id)
              ? Math.min(1, value + 0.25)
              : value * 0.08
            : value;
        }
        return initialInteraction.hasHighlight
          ? initialInteraction.highlightLinkIds.has(d.id)
            ? Math.min(1, base + 0.25)
            : base * 0.08
          : base;
      })
      .attr("stroke", (d) => (showRibbon && initialInteraction.hasHighlight && initialInteraction.highlightLinkIds.has(d.id) ? "#111111b3" : "none"))
      .attr("stroke-width", (d) => (showRibbon && initialInteraction.hasHighlight && initialInteraction.highlightLinkIds.has(d.id) ? 0.6 : 0));

    const barGroup = ribbonLayers.nodes
      .selectAll("g.ribbon-node")
      .data(showRibbon ? ribbonNodes : [], (d) => d.id)
      .join("g")
      .attr("class", "ribbon-node");

    const ribbonBaseSel = barGroup
      .selectAll("rect.base")
      .data((d) => [d])
      .join("rect")
      .attr("class", "base")
      .attr("fill", (d) => d.color)
      .on("mouseenter", (_, d) => {
        if (state.viewMode !== "ribbon") return;
        renderRibbonNodeCard(d);
      })
      .on("mouseleave", () => {
        if (state.viewMode !== "ribbon") return;
        setDefaultTopicCard();
      });

    withTransition(ribbonBaseSel, transition, showRibbon ? 130 : 0)
      .attr("x", (d) => showRibbon ? ribbonLayout.x(d.period) - ribbonLayout.nodeWidth / 2 : ribbonLayout.anchorX)
      .attr("y", (d) => showRibbon ? ribbonLayout.y(d.topic) - (d.count * ribbonLayout.nodeScale) / 2 : ribbonLayout.y(d.topic))
      .attr("width", showRibbon ? ribbonLayout.nodeWidth : 0.8)
      .attr("height", (d) => showRibbon ? Math.max(ribbonLayout.minNodeHeight, d.count * ribbonLayout.nodeScale) : 0.8)
      .attr("rx", (d) => nodeCornerRadius(showRibbon ? Math.max(ribbonLayout.minNodeHeight, d.count * ribbonLayout.nodeScale) : 1, showRibbon ? ribbonLayout.nodeWidth : 1))
      .attr("ry", (d) => nodeCornerRadius(showRibbon ? Math.max(ribbonLayout.minNodeHeight, d.count * ribbonLayout.nodeScale) : 1, showRibbon ? ribbonLayout.nodeWidth : 1))
      .attr("stroke", (d) => {
        if (!showRibbon) return "none";
        if (state.ribbon.country !== "All countries" && initialInteraction.countryNodeIds.has(d.id)) return "#000";
        if (initialInteraction.hasHighlight && initialInteraction.highlightNodeIds.has(d.id)) return "#111";
        return d.is_start || d.is_end ? "#111" : "none";
      })
      .attr("stroke-width", (d) => {
        if (!showRibbon) return 0;
        if (state.ribbon.country !== "All countries" && initialInteraction.countryNodeIds.has(d.id)) return 1.6;
        if (initialInteraction.hasHighlight && initialInteraction.highlightNodeIds.has(d.id)) return 1.1;
        return d.is_start || d.is_end ? 1.0 : 0;
      })
      .attr("opacity", (d) => {
        if (!showRibbon || !ribbonNodeVisible(d, initialInteraction)) return 0;
        const active = initialInteraction.activeNodeIds.has(d.id);
        let base;
        if (state.ribbon.country !== "All countries") {
          base = initialInteraction.countryNodeIds.has(d.id) ? 1.0 : active ? 0.30 : 0.06;
        } else {
          base = active ? 0.92 : 0.14;
        }
        if (initialInteraction.hasHighlight) {
          return initialInteraction.highlightNodeIds.has(d.id) ? Math.min(1, base + 0.25) : base * 0.12;
        }
        return base;
      })
      .style("pointer-events", showRibbon ? "auto" : "none")
      .attr("display", (d) => (showRibbon && ribbonNodeVisible(d, initialInteraction) ? null : "none"));

    const ribbonHatchSel = barGroup
      .selectAll("rect.hatch")
      .data((d) => [d])
      .join("rect")
      .attr("class", "hatch")
      .attr("pointer-events", "none");
    withTransition(ribbonHatchSel, transition, showRibbon ? 130 : 0)
      .attr("x", (d) => showRibbon ? ribbonLayout.x(d.period) - ribbonLayout.nodeWidth / 2 : ribbonLayout.anchorX)
      .attr("y", (d) => showRibbon ? ribbonLayout.y(d.topic) - (d.count * ribbonLayout.nodeScale) / 2 : ribbonLayout.y(d.topic))
      .attr("width", showRibbon ? ribbonLayout.nodeWidth : 0.8)
      .attr("height", (d) => showRibbon ? Math.max(ribbonLayout.minNodeHeight, d.count * ribbonLayout.nodeScale) : 0.8)
      .attr("rx", (d) => nodeCornerRadius(showRibbon ? Math.max(ribbonLayout.minNodeHeight, d.count * ribbonLayout.nodeScale) : 1, showRibbon ? ribbonLayout.nodeWidth : 1))
      .attr("ry", (d) => nodeCornerRadius(showRibbon ? Math.max(ribbonLayout.minNodeHeight, d.count * ribbonLayout.nodeScale) : 1, showRibbon ? ribbonLayout.nodeWidth : 1))
      .attr("fill", (d) => (d.is_start ? "url(#ribbon-hatch-start)" : d.is_end ? "url(#ribbon-hatch-end)" : "none"))
      .attr("opacity", (d) => (showRibbon && (d.is_start || d.is_end) && ribbonNodeVisible(d, initialInteraction) ? 0.72 : 0))
      .attr("display", (d) => (showRibbon && (d.is_start || d.is_end) && ribbonNodeVisible(d, initialInteraction) ? null : "none"));

    const ribbonLabelData = ribbonTopics.map((topic) => ({
      topic,
      theme: ribbonTopicMeta.get(topic)?.theme || "Unknown",
      color: ribbonTopicMeta.get(topic)?.color || "#666",
    }));
    const ribbonLabelSel = ribbonLayers.labels
      .selectAll("text")
      .data(showRibbon ? ribbonLabelData : [], (d) => d.topic)
      .join("text")
      .attr("class", "topic-label")
      .attr("dominant-baseline", "middle")
      .text((d) => d.topic)
      .on("mouseenter", (_, d) => {
        if (state.viewMode !== "ribbon") return;
        if (state.ribbon.hoverTopic === d.topic) return;
        state.ribbon.hoverTopic = d.topic;
        queueRibbonInteractionRefresh();
        renderRibbonLabelCard(d.topic);
      })
      .on("mouseleave", () => {
        if (state.viewMode !== "ribbon") return;
        if (state.ribbon.hoverTopic == null) return;
        state.ribbon.hoverTopic = null;
        queueRibbonInteractionRefresh();
        setDefaultTopicCard();
      })
      .on("click", (event, d) => {
        if (state.viewMode !== "ribbon") return;
        event.stopPropagation();
        if (state.ribbon.lockedTopics.has(d.topic)) {
          state.ribbon.lockedTopics.delete(d.topic);
        } else {
          state.ribbon.lockedTopics.add(d.topic);
        }
        queueRibbonInteractionRefresh();
        renderRibbonLabelCard(d.topic);
      });
    withTransition(ribbonLabelSel, transition, showRibbon ? 200 : 0)
      .attr("x", ribbonLayout.rightLabelX)
      .attr("y", (d) => ribbonLayout.y(d.topic))
      .attr("fill", (d) => d.color)
      .attr("font-weight", (d) => {
        if (state.ribbon.lockedTopics.has(d.topic)) return 700;
        if (state.ribbon.hoverTopic === d.topic) return 650;
        return 400;
      })
      .attr("opacity", (d) => {
        if (!showRibbon || !state.ribbon.showLabels) return 0;
        if (initialInteraction.hasHighlight) return initialInteraction.highlightTopics.has(d.topic) ? 1.0 : 0.16;
        if (state.ribbon.country !== "All countries") return initialInteraction.countryTopics.has(d.topic) ? 1.0 : 0.10;
        if (state.ribbon.theme !== "All themes" && d.theme !== state.ribbon.theme) return 0.14;
        return initialInteraction.activeTopics.has(d.topic) ? 0.98 : 0.35;
      })
      .style("pointer-events", showRibbon ? "auto" : "none")
      .attr("display", () => (showRibbon && state.ribbon.showLabels ? null : "none"));

    const countryLine = d3.line().x((d) => d.x).y((d) => d.y).curve(d3.curveMonotoneX);
    const ribbonCountryPathSel = ribbonLayers.countryPath
      .selectAll("path")
      .data(showRibbon && state.ribbon.country !== "All countries" && initialInteraction.countryPathPoints.length >= 2 ? [initialInteraction.countryPathPoints] : [], (d) => d?.[0]?.id ?? "country-path")
      .join("path")
      .attr("fill", "none")
      .attr("stroke", "#111")
      .attr("stroke-width", 2.2)
      .attr("stroke-dasharray", "6 4");
    withTransition(ribbonCountryPathSel, transition, showRibbon ? 180 : 0)
      .attr("d", countryLine)
      .attr("opacity", showRibbon ? 0.85 : 0);

    const ribbonCountryPointSel = ribbonLayers.countryPath
      .selectAll("circle")
      .data(showRibbon && state.ribbon.country !== "All countries" ? initialInteraction.countryPathPoints : [], (d) => d.id)
      .join("circle")
      .attr("fill", "#111")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.1);
    withTransition(ribbonCountryPointSel, transition, showRibbon ? 180 : 0)
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("r", 3.2)
      .attr("opacity", showRibbon ? 0.95 : 0);

    function applyRibbonInteraction() {
      const interaction = currentRibbonInteraction();

      guideSel
        .attr("stroke-width", (d) => (interaction.hasHighlight && interaction.highlightTopics.has(d) ? 1.2 : 0.9))
        .attr("opacity", (d) => {
          if (!showRibbon) return 0;
          if (interaction.hasHighlight) return interaction.highlightTopics.has(d) ? 0.8 : 0.16;
          if (state.ribbon.country !== "All countries") return interaction.countryTopics.has(d) ? 0.72 : 0.08;
          if (state.ribbon.theme !== "All themes") return ribbonTopicMeta.get(d)?.theme === state.ribbon.theme ? 0.45 : 0.12;
          return interaction.activeTopics.has(d) ? 0.42 : 0.22;
        });

      ribbonLinkSel
        .attr("display", (d) => (showRibbon && interaction.visibleLinkIds.has(d.id) ? null : "none"))
        .attr("opacity", (d) => {
          if (!showRibbon || !interaction.visibleLinkIds.has(d.id)) return 0;
          const base = 0.10 + 0.70 * (d.value / maxRibbonLinkValue);
          if (state.ribbon.country !== "All countries") {
            const value = interaction.countryLinkIds.has(d.id) ? Math.min(0.95, base + 0.2) : base * 0.10;
            return interaction.hasHighlight
              ? interaction.highlightLinkIds.has(d.id)
                ? Math.min(1, value + 0.25)
                : value * 0.08
              : value;
          }
          return interaction.hasHighlight
            ? interaction.highlightLinkIds.has(d.id)
              ? Math.min(1, base + 0.25)
              : base * 0.08
            : base;
        })
        .attr("stroke", (d) => (showRibbon && interaction.hasHighlight && interaction.highlightLinkIds.has(d.id) ? "#111111b3" : "none"))
        .attr("stroke-width", (d) => (showRibbon && interaction.hasHighlight && interaction.highlightLinkIds.has(d.id) ? 0.6 : 0));

      ribbonBaseSel
        .attr("stroke", (d) => {
          if (!showRibbon) return "none";
          if (state.ribbon.country !== "All countries" && interaction.countryNodeIds.has(d.id)) return "#000";
          if (interaction.hasHighlight && interaction.highlightNodeIds.has(d.id)) return "#111";
          return d.is_start || d.is_end ? "#111" : "none";
        })
        .attr("stroke-width", (d) => {
          if (!showRibbon) return 0;
          if (state.ribbon.country !== "All countries" && interaction.countryNodeIds.has(d.id)) return 1.6;
          if (interaction.hasHighlight && interaction.highlightNodeIds.has(d.id)) return 1.1;
          return d.is_start || d.is_end ? 1.0 : 0;
        })
        .attr("display", (d) => (showRibbon && ribbonNodeVisible(d, interaction) ? null : "none"))
        .attr("opacity", (d) => {
          if (!showRibbon || !ribbonNodeVisible(d, interaction)) return 0;
          const active = interaction.activeNodeIds.has(d.id);
          let base;
          if (state.ribbon.country !== "All countries") {
            base = interaction.countryNodeIds.has(d.id) ? 1.0 : active ? 0.30 : 0.06;
          } else {
            base = active ? 0.92 : 0.14;
          }
          if (interaction.hasHighlight) return interaction.highlightNodeIds.has(d.id) ? Math.min(1, base + 0.25) : base * 0.12;
          return base;
        });

      ribbonHatchSel
        .attr("display", (d) => (showRibbon && (d.is_start || d.is_end) && ribbonNodeVisible(d, interaction) ? null : "none"))
        .attr("opacity", (d) => (showRibbon && (d.is_start || d.is_end) && ribbonNodeVisible(d, interaction) ? 0.72 : 0));

      ribbonLabelSel
        .attr("display", () => (showRibbon && state.ribbon.showLabels ? null : "none"))
        .attr("font-weight", (d) => {
          if (state.ribbon.lockedTopics.has(d.topic)) return 700;
          if (state.ribbon.hoverTopic === d.topic) return 650;
          return 400;
        })
        .attr("opacity", (d) => {
          if (!showRibbon || !state.ribbon.showLabels) return 0;
          if (interaction.hasHighlight) return interaction.highlightTopics.has(d.topic) ? 1.0 : 0.16;
          if (state.ribbon.country !== "All countries") return interaction.countryTopics.has(d.topic) ? 1.0 : 0.10;
          if (state.ribbon.theme !== "All themes" && d.theme !== state.ribbon.theme) return 0.14;
          return interaction.activeTopics.has(d.topic) ? 0.98 : 0.35;
        });

      ribbonCountryPathSel
        .data(showRibbon && state.ribbon.country !== "All countries" && interaction.countryPathPoints.length >= 2 ? [interaction.countryPathPoints] : [])
        .join("path")
        .attr("fill", "none")
        .attr("stroke", "#111")
        .attr("stroke-width", 2.2)
        .attr("stroke-dasharray", "6 4")
        .attr("d", countryLine)
        .attr("opacity", showRibbon ? 0.85 : 0);

      ribbonCountryPointSel
        .data(showRibbon && state.ribbon.country !== "All countries" ? interaction.countryPathPoints : [], (d) => d.id)
        .join("circle")
        .attr("fill", "#111")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.1)
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y)
        .attr("r", 3.2)
        .attr("opacity", showRibbon ? 0.95 : 0);

      const nShown = interaction.visibleLinks.length;
      const lockSuffix = state.ribbon.lockedTopics.size > 0 ? ` | Topic locks: ${state.ribbon.lockedTopics.size}` : "";
      const countrySuffix = state.ribbon.country !== "All countries" ? ` | Country: ${state.ribbon.country}` : "";
      setRibbonStatus(`Topics: ${ribbonTopics.length} | Bars: ${ribbonNodes.length} | Links shown: ${nShown}/${ribbonLinks.length}${lockSuffix}${countrySuffix}`);
    }

    ribbonInteractionRefresh = applyRibbonInteraction;
    applyRibbonInteraction();
  }

  const zoom = d3.zoom().scaleExtent([0.6, 6]).on("zoom", (event) => {
    if (state.viewMode !== "space") return;
    currentTransform = event.transform;
    mainRoot.attr("transform", currentTransform);
  });
  svg.call(zoom);

  function render(animate = false, full = animate) {
    const width = Math.max(900, graphPanel.clientWidth || 900);
    const height = Math.max(680, graphPanel.clientHeight || 680);
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    if (state.viewMode === "ribbon") {
      currentTransform = d3.zoomIdentity;
      mainRoot.attr("transform", currentTransform);
      svg.call(zoom.transform, d3.zoomIdentity);
    }

    spaceLayout = computeSpaceLayout(width, height);
    spaceById = spaceLayout.byId;
    ribbonLayout = computeRibbonLayout(width, height);

    if (full || state.viewMode === "space") renderSpaceScene(animate);
    if (full || state.viewMode === "ribbon") renderRibbonScene(animate);
    updateSpaceControlReadout();
    applyViewChrome();
  }

  viewModeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const nextView = button.dataset.viewMode;
      if (!nextView || nextView === state.viewMode) return;
      state.viewMode = nextView;
      state.ribbon.hoverTopic = null;
      setDefaultTopicCard();
      persistUiState();
      render(true, true);
    });
  });

  layoutModeSelect.addEventListener("change", () => {
    state.layoutMode = layoutModeSelect.value;
    persistUiState();
    render(false, false);
  });
  resetLayoutButton.addEventListener("click", () => render(false, false));
  nodeSizeScaleInput.addEventListener("input", () => {
    state.nodeSizeScale = +nodeSizeScaleInput.value;
    persistUiState();
    render(false, false);
  });
  layoutSpreadInput.addEventListener("input", () => {
    state.layoutSpread = +layoutSpreadInput.value;
    persistUiState();
    render(false, false);
  });
  edgePercentileInput.addEventListener("input", () => {
    state.edgePercentile = +edgePercentileInput.value;
    persistUiState();
    render(false, false);
  });

  actorSelect.addEventListener("change", () => {
    state.actorId = actorSelect.value;
    render(false, false);
  });
  thresholdInput.addEventListener("input", () => {
    state.threshold = +thresholdInput.value;
    thresholdValue.textContent = state.threshold.toFixed(1);
    render(false, false);
  });
  showAllLabelsInput.addEventListener("change", () => {
    state.showAllLabels = showAllLabelsInput.checked;
    render(false, false);
  });
  showSelectedLabelsInput.addEventListener("change", () => {
    state.showSelectedLabels = showSelectedLabelsInput.checked;
    render(false, false);
  });
  showBaseSupportInput.addEventListener("change", () => {
    state.showBaseSupport = showBaseSupportInput.checked;
    render(false, false);
  });
  prevActorButton.addEventListener("click", () => {
    const options = allActors.map((actor) => actor.id);
    const current = options.indexOf(actorSelect.value);
    const next = Math.max(0, Math.min(options.length - 1, current - 1));
    actorSelect.value = options[next];
    state.actorId = actorSelect.value;
    render(false, false);
  });
  nextActorButton.addEventListener("click", () => {
    const options = allActors.map((actor) => actor.id);
    const current = options.indexOf(actorSelect.value);
    const next = Math.max(0, Math.min(options.length - 1, current + 1));
    actorSelect.value = options[next];
    state.actorId = actorSelect.value;
    render(false, false);
  });
  function runActorSearch() {
    const query = actorSearch.value.trim().toLowerCase();
    if (!query) return;
    const hit = spaceData.actors.find((actor) => actor.id.toLowerCase().includes(query));
    if (!hit) return;
    actorSelect.value = hit.id;
    state.actorId = hit.id;
    render(false, false);
  }
  actorSearch.addEventListener("change", runActorSearch);
  actorSearch.addEventListener("search", runActorSearch);
  actorSearch.addEventListener("keydown", (event) => {
    if (event.key === "Enter") runActorSearch();
  });

  ribbonThemeSelect.addEventListener("change", () => {
    state.ribbon.theme = ribbonThemeSelect.value;
    queueRibbonInteractionRefresh();
  });
  ribbonCountrySelect.addEventListener("change", () => {
    state.ribbon.country = ribbonCountrySelect.value;
    queueRibbonInteractionRefresh();
  });
  ribbonMinFlowInput.addEventListener("input", () => {
    state.ribbon.minFlow = Number(ribbonMinFlowInput.value);
    ribbonMinFlowValue.textContent = fmtFlow(state.ribbon.minFlow);
    queueRibbonInteractionRefresh();
  });
  ribbonShowLabelsInput.addEventListener("change", () => {
    state.ribbon.showLabels = ribbonShowLabelsInput.checked;
    queueRibbonInteractionRefresh();
  });
  ribbonResetButton.addEventListener("click", () => {
    state.ribbon.theme = "All themes";
    state.ribbon.country = "All countries";
    state.ribbon.minFlow = 0;
    state.ribbon.showLabels = true;
    state.ribbon.hoverTopic = null;
    state.ribbon.lockedTopics.clear();
    ribbonThemeSelect.value = state.ribbon.theme;
    ribbonCountrySelect.value = state.ribbon.country;
    ribbonMinFlowInput.value = "0";
    ribbonMinFlowValue.textContent = ribbonCountWidth ? "0" : "0.00";
    ribbonShowLabelsInput.checked = true;
    queueRibbonInteractionRefresh();
  });

  svg.on("click", () => {
    if (state.viewMode !== "ribbon") return;
    if (state.ribbon.lockedTopics.size > 0) {
      state.ribbon.lockedTopics.clear();
      queueRibbonInteractionRefresh();
      setDefaultTopicCard();
    }
  });

  window.addEventListener("resize", () => render(false, true));

  applyPersistedUiState();
  ribbonThemeSelect.value = state.ribbon.theme;
  ribbonCountrySelect.value = state.ribbon.country;
  ribbonShowLabelsInput.checked = state.ribbon.showLabels;
  thresholdValue.textContent = state.threshold.toFixed(1);
  setDefaultTopicCard();
  applyViewChrome();
  render(false, true);
}

main().catch((error) => {
  console.error(error);
  const meta = document.getElementById("meta");
  if (meta) meta.textContent = `Failed to load app: ${error}`;
});
