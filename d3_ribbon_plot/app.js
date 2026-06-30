async function main() {
  const data = await d3.json("./data/fig2_ribbon_data.json");

  const periods = data.periods;
  const topics = data.topics_order;
  const themeColors = data.theme_colors;
  const themes = data.themes_order;
  const nodes = data.nodes.map((d) => ({ ...d }));
  const links = data.links.map((d) => ({ ...d }));
  const countryPaths = data.country_paths || {};
  const themeCounts = data.theme_counts || [];
  const transitionSource = data.meta?.transition_source || "top_rca";
  const edgeWidthMetric = data.meta?.edge_width_metric || "actor_count";

  const svg = d3.select("#chart");
  const panel = document.querySelector(".chart-panel");
  const legendRoot = d3.select("#legend");
  const info = document.getElementById("info");
  const status = document.getElementById("status");

  const themeSelect = document.getElementById("theme-select");
  const countrySelect = document.getElementById("country-select");
  const minFlow = document.getElementById("min-flow");
  const minFlowValue = document.getElementById("min-flow-value");
  const showLabels = document.getElementById("show-labels");
  const resetBtn = document.getElementById("reset-btn");

  const nodeById = new Map(nodes.map((d) => [d.id, d]));
  const topicMeta = new Map();
  nodes.forEach((n) => {
    if (!topicMeta.has(n.topic)) {
      topicMeta.set(n.topic, { theme: n.theme, color: n.color });
    }
  });
  const topicIndex = new Map(topics.map((d, i) => [d, i]));
  const periodIndex = new Map(periods.map((d, i) => [d, i]));
  const maxNodeCount = d3.max(nodes, (d) => d.count) || 1;
  const maxLinkValue = d3.max(links, (d) => d.value) || 1;
  const countWidth = edgeWidthMetric === "actor_count";
  const fmtFlow = countWidth ? d3.format(",d") : d3.format(".2f");
  const fmtSupport = d3.format(".2f");
  const fmtInt = d3.format(",d");

  const nodeCountries = new Map(
    nodes.map((n) => [n.id, Array.from(new Set(n.countries || [])).sort()])
  );
  // Fallback for older exports without node-level country lists.
  if ([...nodeCountries.values()].every((arr) => arr.length === 0)) {
    Object.entries(countryPaths).forEach(([country, path]) => {
      path.forEach((step) => {
        const id = `${step.period}::${step.topic}`;
        if (nodeCountries.has(id)) nodeCountries.get(id).push(country);
      });
    });
    for (const [id, arr] of nodeCountries.entries()) {
      nodeCountries.set(id, Array.from(new Set(arr)).sort());
    }
  }

  const state = {
    theme: "All themes",
    country: "All countries",
    minFlow: 0,
    showLabels: true,
    hoverTopic: null,
    lockedTopics: new Set(),
  };
  let refreshInteraction = null;
  let interactionRaf = null;
  const linkGeomCache = {
    width: -1,
    height: -1,
    nodeWidth: -1,
    rowGap: -1,
    linkGeom: null,
  };

  function idleInfoText() {
    return state.lockedTopics.size > 0
      ? `Locked topics: ${[...state.lockedTopics].join(", ")}`
      : "Hover ribbons, nodes, or right-side topic labels.";
  }

  function queueInteractionRefresh() {
    if (!refreshInteraction) return;
    if (interactionRaf != null) return;
    interactionRaf = requestAnimationFrame(() => {
      interactionRaf = null;
      if (refreshInteraction) refreshInteraction();
    });
  }

  function setInfo(html) {
    info.innerHTML = html;
  }
  function setStatus(text) {
    status.textContent = text;
  }

  const countryList = Object.keys(countryPaths).sort((a, b) => a.localeCompare(b));
  const themeOptions = ["All themes", ...themes];

  themeSelect.innerHTML = "";
  themeOptions.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    themeSelect.appendChild(opt);
  });
  countrySelect.innerHTML = "";
  ["All countries", ...countryList].forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    countrySelect.appendChild(opt);
  });

  const flowStep = countWidth
    ? 1
    : maxLinkValue <= 5
      ? 0.05
      : maxLinkValue <= 20
        ? 0.1
        : 0.5;
  minFlow.min = "0";
  minFlow.max = String(maxLinkValue);
  minFlow.step = String(flowStep);
  minFlow.value = "0";
  minFlowValue.textContent = countWidth ? "0" : "0.00";

  legendRoot.selectAll("*").remove();
  const legendItems = legendRoot
    .selectAll(".legend-item")
    .data(themes)
    .join("div")
    .attr("class", "legend-item")
    .style("cursor", "pointer")
    .on("click", (_, theme) => {
      state.theme = state.theme === theme ? "All themes" : theme;
      themeSelect.value = state.theme;
      render();
    });

  legendItems
    .append("span")
    .attr("class", "swatch")
    .style("background", (d) => themeColors[d] || "#999");
  legendItems.append("span").text((d) => d);

  const defs = svg.append("defs");
  defs
    .append("pattern")
    .attr("id", "hatch-start")
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
    .attr("id", "hatch-end")
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 5)
    .attr("height", 5)
    .append("path")
    .attr("d", "M-1,4 l2,2 M0,0 l5,5 M4,-1 l2,2")
    .attr("stroke", "#111")
    .attr("stroke-width", 1.15)
    .attr("stroke-linecap", "round");

  const gRoot = svg.append("g");
  const gTop = gRoot.append("g").attr("class", "top-panel");
  const gBottom = gRoot.append("g").attr("class", "bottom-panel");
  const gPeriodLabels = gTop.append("g");
  const gTopicGuides = gTop.append("g");
  const gLinks = gTop.append("g");
  const gNodes = gTop.append("g");
  const gCountryPath = gTop.append("g");
  const gTopicLabels = gTop.append("g");
  const gArea = gBottom.append("g");
  const gAreaAxes = gBottom.append("g");

  svg.on("click", () => {
    if (state.lockedTopics.size > 0) {
      state.lockedTopics.clear();
      setInfo("Topic lock cleared.");
      queueInteractionRefresh();
    }
  });

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

  function makeLinkGeometry(xMap, yMap, nodeWidth, rowGap) {
    const linksByPair = d3.group(links, (d) => `${d.period0}||${d.period1}`);
    const out = [];

    linksByPair.forEach((pairLinks) => {
      const totalsBySrc = new Map();
      const totalsByTgt = new Map();
      pairLinks.forEach((l) => {
        totalsBySrc.set(l.source, (totalsBySrc.get(l.source) || 0) + l.value);
        totalsByTgt.set(l.target, (totalsByTgt.get(l.target) || 0) + l.value);
      });
      const maxTotal = d3.max([...totalsBySrc.values()]) || 1;
      // Slightly thicker ribbons for presentation visibility.
      const scale = (rowGap * 2.8 * 0.95) / maxTotal;

      const srcOffsets = new Map();
      const tgtOffsets = new Map();
      totalsBySrc.forEach((total, sourceId) => {
        srcOffsets.set(sourceId, yMap.get(nodeById.get(sourceId).topic) - (total * scale) / 2);
      });
      totalsByTgt.forEach((total, targetId) => {
        tgtOffsets.set(targetId, yMap.get(nodeById.get(targetId).topic) - (total * scale) / 2);
      });

      const sorted = [...pairLinks].sort((a, b) => {
        const sa = topicIndex.get(a.source_topic);
        const sb = topicIndex.get(b.source_topic);
        if (sa !== sb) return sa - sb;
        return topicIndex.get(a.target_topic) - topicIndex.get(b.target_topic);
      });

      const srcPos = new Map();
      sorted.forEach((l) => {
        const thick = l.value * scale;
        const start = srcOffsets.get(l.source) || 0;
        srcPos.set(l.id, { y: start + thick / 2, thickness: thick });
        srcOffsets.set(l.source, start + thick);
      });

      const sortedByTarget = [...sorted].sort((a, b) => {
        const ta = topicIndex.get(a.target_topic);
        const tb = topicIndex.get(b.target_topic);
        if (ta !== tb) return ta - tb;
        return topicIndex.get(a.source_topic) - topicIndex.get(b.source_topic);
      });
      const tgtPos = new Map();
      sortedByTarget.forEach((l) => {
        const thick = l.value * scale;
        const start = tgtOffsets.get(l.target) || 0;
        tgtPos.set(l.id, { y: start + thick / 2, thickness: thick });
        tgtOffsets.set(l.target, start + thick);
      });

      sorted.forEach((l) => {
        const srcNode = nodeById.get(l.source);
        const tgtNode = nodeById.get(l.target);
        const sx = xMap.get(srcNode.period) + nodeWidth / 2;
        const tx = xMap.get(tgtNode.period) - nodeWidth / 2;
        const sy = srcPos.get(l.id).y;
        const ty = tgtPos.get(l.id).y;
        const thickness = srcPos.get(l.id).thickness;
        out.push({
          ...l,
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

  function render() {
    if (interactionRaf != null) {
      cancelAnimationFrame(interactionRaf);
      interactionRaf = null;
    }

    const width = Math.max(900, panel.clientWidth || 900);
    const height = Math.max(620, panel.clientHeight || 620);
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const margin = { top: 26, right: 250, bottom: 20, left: 92 };
    const topHeight = Math.round(height * 0.78);
    const bottomGap = 14;
    const bottomY = topHeight + bottomGap;
    const bottomHeight = height - bottomY - 20;

    gTop.attr("transform", "translate(0,0)");
    gBottom.attr("transform", `translate(0,${bottomY})`);

    const rightNodePad = 17;
    const x = d3
      .scalePoint()
      .domain(periods)
      .range([margin.left, width - margin.right - rightNodePad]);

    const y = d3
      .scalePoint()
      .domain(topics)
      .range([margin.top + 26, topHeight - 26]);

    const rowGap =
      topics.length > 1 ? Math.abs(y(topics[1]) - y(topics[0])) : topHeight * 0.5;
    const nodeWidth = Math.max(
      12,
      Math.min(20, (width - margin.left - margin.right) / (periods.length * 4.6))
    );
    const nodeScale = (rowGap * 2.8) / maxNodeCount;
    const minNodeHeight = 2.4;

    const topicSelected = (l) => {
      if (
        state.hoverTopic &&
        (l.source_topic === state.hoverTopic || l.target_topic === state.hoverTopic)
      ) {
        return true;
      }
      if (state.lockedTopics.size > 0) {
        return (
          state.lockedTopics.has(l.source_topic) ||
          state.lockedTopics.has(l.target_topic)
        );
      }
      return false;
    };

    const visibleLinkFilter = (l) => {
      if (topicSelected(l)) return true;
      if (l.value < state.minFlow) return false;
      if (state.theme !== "All themes") {
        if (l.source_theme !== state.theme && l.target_theme !== state.theme)
          return false;
      }
      return true;
    };

    let linkGeom = linkGeomCache.linkGeom;
    if (
      !linkGeom ||
      linkGeomCache.width !== width ||
      linkGeomCache.height !== height ||
      linkGeomCache.nodeWidth !== nodeWidth ||
      linkGeomCache.rowGap !== rowGap
    ) {
      linkGeom = makeLinkGeometry(
        new Map(periods.map((p) => [p, x(p)])),
        new Map(topics.map((t) => [t, y(t)])),
        nodeWidth,
        rowGap
      );
      linkGeomCache.width = width;
      linkGeomCache.height = height;
      linkGeomCache.nodeWidth = nodeWidth;
      linkGeomCache.rowGap = rowGap;
      linkGeomCache.linkGeom = linkGeom;
    }

    const computeCountryState = () => {
      const countryNodeIds = new Set();
      const countryLinkIds = new Set();
      const countryPathPoints = [];
      const countryTopics = new Set();

      if (state.country !== "All countries" && countryPaths[state.country]) {
        const path = [...countryPaths[state.country]].sort(
          (a, b) => a.period_order - b.period_order
        );
        path.forEach((step) => {
          const id = `${step.period}::${step.topic}`;
          countryNodeIds.add(id);
          countryTopics.add(step.topic);
          if (nodeById.has(id)) {
            const n = nodeById.get(id);
            if (x(n.period) != null && y(n.topic) != null) {
              countryPathPoints.push({
                x: x(n.period),
                y: y(n.topic),
                id,
                rca: step.rca,
              });
            }
          }
        });
        links.forEach((l) => {
          if ((l.countries || []).includes(state.country)) countryLinkIds.add(l.id);
        });
      }

      return { countryNodeIds, countryLinkIds, countryPathPoints, countryTopics };
    };

    const baseLinkOpacity = (d, visibleLinkIds, countryState) => {
      if (!visibleLinkIds.has(d.id)) return 0;
      const base = 0.10 + 0.70 * (d.value / maxLinkValue);
      if (state.country !== "All countries") {
        return countryState.countryLinkIds.has(d.id)
          ? Math.min(0.95, base + 0.2)
          : base * 0.10;
      }
      return base;
    };

    const baseNodeOpacity = (d, activeNodeIds, countryState) => {
      const active = activeNodeIds.has(d.id);
      if (state.country !== "All countries") {
        if (countryState.countryNodeIds.has(d.id)) return 1.0;
        return active ? 0.30 : 0.06;
      }
      return active ? 0.92 : 0.14;
    };

    const baseNodeStroke = (d, countryState) => {
      if (state.country !== "All countries" && countryState.countryNodeIds.has(d.id))
        return "#000";
      if (d.is_start || d.is_end) return "#111";
      return "none";
    };

    const baseNodeStrokeWidth = (d, countryState) => {
      if (state.country !== "All countries" && countryState.countryNodeIds.has(d.id))
        return 1.6;
      return d.is_start || d.is_end ? 1.0 : 0;
    };

    const computeInteractionState = () => {
      const visibleLinks = linkGeom.filter(visibleLinkFilter);
      const visibleLinkIds = new Set(visibleLinks.map((l) => l.id));
      const activeNodeIds = new Set();
      const activeTopics = new Set();
      visibleLinks.forEach((l) => {
        activeNodeIds.add(l.source);
        activeNodeIds.add(l.target);
        activeTopics.add(l.source_topic);
        activeTopics.add(l.target_topic);
      });

      const highlightTopics = new Set([...state.lockedTopics]);
      if (state.hoverTopic) highlightTopics.add(state.hoverTopic);
      const hasHighlight = highlightTopics.size > 0;
      const highlightLinkIds = new Set();
      const highlightNodeIds = new Set();
      if (hasHighlight) {
        visibleLinks.forEach((l) => {
          if (
            highlightTopics.has(l.source_topic) ||
            highlightTopics.has(l.target_topic)
          ) {
            highlightLinkIds.add(l.id);
            highlightNodeIds.add(l.source);
            highlightNodeIds.add(l.target);
          }
        });
        nodes.forEach((n) => {
          if (highlightTopics.has(n.topic)) highlightNodeIds.add(n.id);
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
      };
    };

    let interaction = computeInteractionState();

    const nodeVisible = (d, ctx = interaction) => {
      if (ctx.hasHighlight && ctx.highlightNodeIds.has(d.id)) return true;
      if (
        state.theme !== "All themes" &&
        d.theme !== state.theme &&
        !ctx.activeNodeIds.has(d.id)
      ) {
        return false;
      }
      return true;
    };

    // Period labels
    gPeriodLabels
      .selectAll("text")
      .data(periods)
      .join("text")
      .attr("class", "period-label")
      .attr("x", (d) => x(d))
      .attr("y", margin.top - 5)
      .attr("text-anchor", "middle")
      .text((d) => d);

    // Horizontal topic guide lines (as in the static ribbon figure)
    const guideSel = gTopicGuides
      .selectAll("line")
      .data(topics, (d) => d)
      .join("line")
      .attr("x1", margin.left + nodeWidth * 0.35)
      .attr("x2", width - margin.right - rightNodePad - nodeWidth * 0.35)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d))
      .attr("stroke", "#6b7280")
      .attr("stroke-dasharray", "4 4");

    // Links
    const linkSel = gLinks
      .selectAll("path")
      .data(linkGeom, (d) => d.id)
      .join("path")
      .attr("class", "link")
      .attr("d", (d) => d.path)
      .attr("fill", (d) => nodeById.get(d.source)?.color || "#999")
      .on("mouseenter", function (_, d) {
        const sample = d.countries.slice(0, 8).join(", ");
        setInfo(
          `<strong>${d.source_topic}</strong> → <strong>${d.target_topic}</strong><br>` +
            `Period: ${d.period0} → ${d.period1}<br>` +
            `Actors (edge width): ${fmtInt(d.actor_count ?? Math.round(d.value || 0))}<br>` +
            `Weighted support: ${fmtSupport(d.support_value ?? d.value)}<br>` +
            `Theme: ${d.source_theme}${d.source_theme === d.target_theme ? "" : ` → ${d.target_theme}`}<br>` +
            `Countries: ${sample}${d.countries.length > 8 ? "…" : ""}`
        );
      })
      .on("mouseleave", function () {
        setInfo(idleInfoText());
      });

    // Nodes
    const nodeGroup = gNodes.selectAll("g.node-g").data(nodes, (d) => d.id).join("g").attr("class", "node-g");
    const nodeBaseSel = nodeGroup
      .selectAll("rect.base")
      .data((d) => [d])
      .join("rect")
      .attr("class", "node")
      .attr("x", (d) => x(d.period) - nodeWidth / 2)
      .attr("y", (d) => y(d.topic) - (d.count * nodeScale) / 2)
      .attr("width", nodeWidth)
      .attr("height", (d) => Math.max(minNodeHeight, d.count * nodeScale))
      .attr("rx", (d) => nodeCornerRadius(Math.max(minNodeHeight, d.count * nodeScale), nodeWidth))
      .attr("ry", (d) => nodeCornerRadius(Math.max(minNodeHeight, d.count * nodeScale), nodeWidth))
      .attr("fill", (d) => d.color)
      .on("mouseenter", function (_, d) {
        d3.select(this).attr("opacity", 1.0).raise();
        const countries = nodeCountries.get(d.id) || [];
        const sample = countries.slice(0, 10).join(", ");
        setInfo(
          `<strong>${d.topic}</strong><br>` +
            `Period: ${d.period}<br>` +
            `Actors with RCA≥1 in topic: ${fmtInt(d.count)}<br>` +
            `Theme: ${d.theme}<br>` +
            `Countries: ${sample}${countries.length > 10 ? "…" : ""}`
        );
      })
      .on("mouseleave", function (_, d) {
        const countryState = computeCountryState();
        const base = baseNodeOpacity(d, interaction.activeNodeIds, countryState);
        const o = interaction.hasHighlight
          ? interaction.highlightNodeIds.has(d.id)
            ? Math.min(1, base + 0.25)
            : base * 0.12
          : base;
        d3.select(this).attr("opacity", o);
        setInfo(idleInfoText());
      });

    const hatchSel = nodeGroup
      .selectAll("rect.hatch")
      .data((d) => [d])
      .join("rect")
      .attr("class", "hatch")
      .attr("x", (d) => x(d.period) - nodeWidth / 2)
      .attr("y", (d) => y(d.topic) - (d.count * nodeScale) / 2)
      .attr("width", nodeWidth)
      .attr("height", (d) => Math.max(minNodeHeight, d.count * nodeScale))
      .attr("rx", (d) => nodeCornerRadius(Math.max(minNodeHeight, d.count * nodeScale), nodeWidth))
      .attr("ry", (d) => nodeCornerRadius(Math.max(minNodeHeight, d.count * nodeScale), nodeWidth))
      .attr("fill", (d) => (d.is_start ? "url(#hatch-start)" : d.is_end ? "url(#hatch-end)" : "none"))
      .attr("opacity", 0.72)
      .attr("pointer-events", "none");

    // Country trace
    const line = d3
      .line()
      .x((d) => d.x)
      .y((d) => d.y)
      .curve(d3.curveMonotoneX);

    // Topic labels on the right
    const labelData = topics.map((t) => ({
      topic: t,
      theme: topicMeta.get(t)?.theme || "Unknown",
      color: topicMeta.get(t)?.color || "#666",
    }));
    const labelSel = gTopicLabels
      .selectAll("text")
      .data(labelData, (d) => d.topic)
      .join("text")
      .attr("class", "label")
      .attr("x", width - margin.right + 12)
      .attr("y", (d) => y(d.topic))
      .attr("dominant-baseline", "middle")
      .attr("fill", (d) => d.color)
      .text((d) => d.topic)
      .on("mouseenter", function (_, d) {
        if (state.hoverTopic === d.topic) return;
        state.hoverTopic = d.topic;
        setInfo(
          `<strong>${d.topic}</strong><br>` +
            `Theme: ${d.theme}<br>` +
            `Hover to preview trace, click to lock/unlock.`
        );
        queueInteractionRefresh();
      })
      .on("mouseleave", function () {
        if (state.hoverTopic == null) return;
        state.hoverTopic = null;
        setInfo(idleInfoText());
        queueInteractionRefresh();
      })
      .on("click", function (event, d) {
        event.stopPropagation();
        if (state.lockedTopics.has(d.topic)) {
          state.lockedTopics.delete(d.topic);
        } else {
          state.lockedTopics.add(d.topic);
        }
        setInfo(
          state.lockedTopics.size > 0
            ? `Locked topics: ${[...state.lockedTopics].join(", ")}`
            : "Topic lock cleared."
        );
        queueInteractionRefresh();
      });

    const applyTopicInteraction = () => {
      interaction = computeInteractionState();
      const countryState = computeCountryState();

      gCountryPath
        .selectAll("path")
        .data(
          state.country === "All countries" || countryState.countryPathPoints.length < 2
            ? []
            : [countryState.countryPathPoints]
        )
        .join("path")
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "#111")
        .attr("stroke-width", 2.2)
        .attr("stroke-dasharray", "6 4")
        .attr("opacity", 0.85);

      gCountryPath
        .selectAll("circle")
        .data(
          state.country === "All countries" ? [] : countryState.countryPathPoints,
          (d) => d.id
        )
        .join("circle")
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y)
        .attr("r", 3.2)
        .attr("fill", "#111")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.1)
        .attr("opacity", 0.95);

      guideSel
        .attr("stroke-width", (d) =>
          interaction.hasHighlight && interaction.highlightTopics.has(d) ? 1.2 : 0.9
        )
        .attr("opacity", (d) => {
          if (interaction.hasHighlight) {
            return interaction.highlightTopics.has(d) ? 0.8 : 0.16;
          }
          if (state.country !== "All countries") {
            return countryState.countryTopics.has(d) ? 0.72 : 0.08;
          }
          if (state.theme !== "All themes") {
            return topicMeta.get(d)?.theme === state.theme ? 0.45 : 0.12;
          }
          return interaction.activeTopics.has(d) ? 0.42 : 0.22;
        });

      linkSel
        .attr("display", (d) =>
          interaction.visibleLinkIds.has(d.id) ? null : "none"
        )
        .attr("opacity", (d) => {
          const base = baseLinkOpacity(d, interaction.visibleLinkIds, countryState);
          if (!interaction.hasHighlight) return base;
          return interaction.highlightLinkIds.has(d.id) ? Math.min(1, base + 0.25) : base * 0.08;
        })
        .attr("stroke", (d) =>
          interaction.hasHighlight && interaction.highlightLinkIds.has(d.id)
            ? "#111111b3"
            : "none"
        )
        .attr("stroke-width", (d) =>
          interaction.hasHighlight && interaction.highlightLinkIds.has(d.id) ? 0.6 : 0
        );

      nodeBaseSel
        .attr("stroke", (d) => {
          const base = baseNodeStroke(d, countryState);
          if (!interaction.hasHighlight) return base;
          return interaction.highlightNodeIds.has(d.id)
            ? "#111"
            : base === "none"
              ? "none"
              : base;
        })
        .attr("stroke-width", (d) => {
          const base = baseNodeStrokeWidth(d, countryState);
          if (!interaction.hasHighlight) return base;
          if (interaction.highlightNodeIds.has(d.id)) return Math.max(base, 1.1);
          return base;
        })
        .attr("display", (d) => (nodeVisible(d, interaction) ? null : "none"))
        .attr("opacity", (d) => {
          const base = baseNodeOpacity(d, interaction.activeNodeIds, countryState);
          if (!interaction.hasHighlight) return base;
          return interaction.highlightNodeIds.has(d.id) ? Math.min(1, base + 0.25) : base * 0.12;
        });

      hatchSel.attr("display", (d) => {
        if (!nodeVisible(d, interaction)) return "none";
        if (!d.is_start && !d.is_end) return "none";
        return null;
      });

      labelSel
        .attr("display", () => (state.showLabels ? null : "none"))
        .attr("opacity", (d) => {
          if (!state.showLabels) return 0;
          if (interaction.hasHighlight) {
            return interaction.highlightTopics.has(d.topic) ? 1.0 : 0.16;
          }
          if (state.country !== "All countries") {
            return countryState.countryTopics.has(d.topic) ? 1.0 : 0.10;
          }
          if (state.theme !== "All themes" && d.theme !== state.theme) return 0.14;
          return interaction.activeTopics.has(d.topic) ? 0.98 : 0.35;
        })
        .attr("font-weight", (d) => {
          if (state.lockedTopics.has(d.topic)) return 700;
          if (state.hoverTopic === d.topic) return 650;
          return 400;
        });

      areaSel.attr("opacity", (d) => {
        if (state.theme === "All themes") return 0.84;
        return d.key === state.theme ? 0.92 : 0.15;
      });

      const nShown = interaction.visibleLinks.length;
      const lockSuffix =
        state.lockedTopics.size > 0 ? ` | Topic locks: ${state.lockedTopics.size}` : "";
      const countrySuffix =
        state.country !== "All countries" ? ` | Country: ${state.country}` : "";
      setStatus(
        `Topics: ${topics.length} | Nodes: ${nodes.length} | Links shown: ${nShown}/${links.length}${lockSuffix}${countrySuffix}`
      );
    };
    // Bottom stacked theme chart.
    const bottomMargin = { left: margin.left, right: margin.right, top: 4, bottom: 22 };
    const bInnerW = width - bottomMargin.left - bottomMargin.right;
    const bInnerH = Math.max(50, bottomHeight - bottomMargin.top - bottomMargin.bottom);

    const tCounts = themeCounts.map((row) => ({ ...row }));
    const stack = d3.stack().keys(themes);
    const stacked = stack(tCounts);
    const maxStack = d3.max(stacked, (layer) => d3.max(layer, (d) => d[1])) || 1;
    const xb = d3
      .scalePoint()
      .domain(periods)
      .range([0, bInnerW]);
    const yb = d3
      .scaleLinear()
      .domain([0, maxStack])
      .range([bInnerH, 0]);

    const area = d3
      .area()
      .x((d) => xb(d.data.period))
      .y0((d) => yb(d[0]))
      .y1((d) => yb(d[1]))
      .curve(d3.curveMonotoneX);

    const areaG = gArea.attr("transform", `translate(${bottomMargin.left},${bottomMargin.top})`);
    const areaSel = areaG
      .selectAll("path")
      .data(stacked, (d) => d.key)
      .join("path")
      .attr("d", area)
      .attr("fill", (d) => themeColors[d.key] || "#999")
      .attr("opacity", 0.84);

    refreshInteraction = applyTopicInteraction;
    applyTopicInteraction();

    const axisG = gAreaAxes.attr(
      "transform",
      `translate(${bottomMargin.left},${bottomMargin.top})`
    );
    axisG.selectAll("*").remove();
    axisG
      .append("g")
      .attr("transform", `translate(0,${bInnerH})`)
      .call(d3.axisBottom(xb))
      .selectAll("text")
      .attr("font-size", 12)
      .attr("transform", "rotate(-20)")
      .attr("text-anchor", "end");
    axisG
      .append("g")
      .call(d3.axisLeft(yb).ticks(3))
      .selectAll("text")
      .attr("font-size", 12);
    axisG
      .append("text")
      .attr("class", "theme-axis-label")
      .attr("x", -8)
      .attr("y", -2)
      .text("RCA>1 topic assignments by theme");

  }

  themeSelect.addEventListener("change", () => {
    state.theme = themeSelect.value;
    queueInteractionRefresh();
  });
  countrySelect.addEventListener("change", () => {
    state.country = countrySelect.value;
    queueInteractionRefresh();
  });
  minFlow.addEventListener("input", () => {
    state.minFlow = Number(minFlow.value);
    minFlowValue.textContent = fmtFlow(state.minFlow);
    queueInteractionRefresh();
  });
  showLabels.addEventListener("change", () => {
    state.showLabels = showLabels.checked;
    queueInteractionRefresh();
  });
  resetBtn.addEventListener("click", () => {
    state.theme = "All themes";
    state.country = "All countries";
    state.minFlow = 0;
    state.showLabels = true;
    state.hoverTopic = null;
    state.lockedTopics.clear();
    themeSelect.value = state.theme;
    countrySelect.value = state.country;
    minFlow.value = "0";
    minFlowValue.textContent = countWidth ? "0" : "0.00";
    showLabels.checked = true;
    queueInteractionRefresh();
  });

  window.addEventListener("resize", () => render());

  setInfo(idleInfoText());
  setStatus(
    `Loaded ${transitionSource} transition mode | Edge width: ${edgeWidthMetric} | Topics: ${topics.length} | Nodes: ${nodes.length} | Links: ${links.length}`
  );
  render();
}

main().catch((err) => {
  const status = document.getElementById("status");
  if (status) status.textContent = `Failed to load data: ${err}`;
  console.error(err);
});
