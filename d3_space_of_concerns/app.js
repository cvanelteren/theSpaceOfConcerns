async function main() {
  const data = await d3.json("./data/space_of_concerns_graph.json");
  const nodes = data.nodes.map((d) => ({ ...d }));
  const links = data.links.map((d) => ({ ...d }));

  const svg = d3.select("#chart");
  const panel = document.querySelector(".graph-panel");
  const info = document.getElementById("info");
  const status = document.getElementById("status");
  const showLabels = document.getElementById("labels");
  const chargeInput = document.getElementById("charge");
  const chargeBox = document.getElementById("charge-box");
  const distanceInput = document.getElementById("distance");
  const distanceBox = document.getElementById("distance-box");
  const layoutModeInput = document.getElementById("layout-mode");
  const focusNodeInput = document.getElementById("focus-node");
  const fitToAntarcticaInput = document.getElementById("fit-to-antarctica");
  const edgeScaleInput = document.getElementById("edge-scale");
  const edgeScaleBox = document.getElementById("edge-scale-box");
  const edgeRepelStrengthInput = document.getElementById("edge-repel-strength");
  const edgeRepelStrengthBox = document.getElementById("edge-repel-strength-box");
  const edgeRepelRadiusInput = document.getElementById("edge-repel-radius");
  const edgeRepelRadiusBox = document.getElementById("edge-repel-radius-box");
  const nodeScaleInput = document.getElementById("node-scale");
  const nodeScaleBox = document.getElementById("node-scale-box");
  const uniformNodeSizeInput = document.getElementById("uniform-node-size");
  const pinDraggedInput = document.getElementById("pin-dragged");
  const boxSelectInput = document.getElementById("box-select");
  const saveButton = document.getElementById("save-layout");
  const clearSelectionButton = document.getElementById("clear-selection");
  const clearPinsButton = document.getElementById("clear-pins");
  const resetButton = document.getElementById("reset");

  const legend = d3.select("#legend");
  Object.entries(data.theme_colors).forEach(([theme, color]) => {
    const row = legend.append("div").attr("class", "legend-item");
    row.append("span").attr("class", "swatch").style("background", color);
    row.append("span").text(theme);
  });

  const sortedNodeIds = nodes.map((n) => n.id).sort((a, b) => a.localeCompare(b));
  sortedNodeIds.forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    focusNodeInput.appendChild(opt);
  });
  if (sortedNodeIds.length > 0) focusNodeInput.value = sortedNodeIds[0];

  const neighborMap = new Map(nodes.map((n) => [n.id, new Set()]));
  const mstNeighborMap = new Map(nodes.map((n) => [n.id, new Set()]));
  links.forEach((l) => {
    const a = String(l.source);
    const b = String(l.target);
    if (!neighborMap.has(a)) neighborMap.set(a, new Set());
    if (!neighborMap.has(b)) neighborMap.set(b, new Set());
    neighborMap.get(a).add(b);
    neighborMap.get(b).add(a);
    if (l.kind === "mst") {
      if (!mstNeighborMap.has(a)) mstNeighborMap.set(a, new Set());
      if (!mstNeighborMap.has(b)) mstNeighborMap.set(b, new Set());
      mstNeighborMap.get(a).add(b);
      mstNeighborMap.get(b).add(a);
    }
  });
  const hasMstLinks = links.some((l) => l.kind === "mst");
  const shellNeighborMap = hasMstLinks ? mstNeighborMap : neighborMap;
  const selectedIds = new Set();
  const ANTARCTICA_BG_SCALE = 0.68;
  const ANTARCTICA_ALPHA_THRESHOLD = 20;
  const antarcticaImg = await new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = "./assets/antarctica_contour.png";
  });

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.style.color = isError ? "#b91c1c" : "#6b7280";
  }

  async function loadSavedLayout() {
    try {
      const saved = await d3.json(`./data/space_of_concerns_layout_saved.json?ts=${Date.now()}`);
      if (!saved || !Array.isArray(saved.nodes)) return false;
      const byId = new Map(saved.nodes.map((d) => [d.id, d]));
      let nApplied = 0;
      nodes.forEach((n) => {
        const hit = byId.get(n.id);
        if (hit && Number.isFinite(hit.x) && Number.isFinite(hit.y)) {
          n.x = hit.x;
          n.y = hit.y;
          n.vx = 0;
          n.vy = 0;
          if (hit.pinned === true) {
            n.fx = hit.x;
            n.fy = hit.y;
          } else {
            n.fx = null;
            n.fy = null;
          }
          nApplied += 1;
        }
      });
      if (typeof saved.charge === "number") chargeInput.value = String(saved.charge);
      if (typeof saved.distance === "number") distanceInput.value = String(saved.distance);
      if (typeof saved.layout_mode === "string") layoutModeInput.value = saved.layout_mode;
      if (typeof saved.focus_node === "string" && sortedNodeIds.includes(saved.focus_node)) {
        focusNodeInput.value = saved.focus_node;
      }
      if (typeof saved.fit_to_antarctica === "boolean") {
        fitToAntarcticaInput.checked = saved.fit_to_antarctica;
      }
      if (typeof saved.edge_repel_strength === "number") {
        edgeRepelStrengthInput.value = String(saved.edge_repel_strength);
      }
      if (typeof saved.edge_repel_radius === "number") {
        edgeRepelRadiusInput.value = String(saved.edge_repel_radius);
      }
      if (typeof saved.edge_scale === "number") edgeScaleInput.value = String(saved.edge_scale);
      if (typeof saved.node_scale === "number") nodeScaleInput.value = String(saved.node_scale);
      if (typeof saved.uniform_node_size === "boolean") {
        uniformNodeSizeInput.checked = saved.uniform_node_size;
      }
      if (typeof saved.pin_dragged === "boolean") {
        pinDraggedInput.checked = saved.pin_dragged;
      }
      if (typeof saved.box_select === "boolean") {
        boxSelectInput.checked = saved.box_select;
      }
      if (Array.isArray(saved.selected_ids)) {
        selectedIds.clear();
        saved.selected_ids.forEach((id) => {
          if (sortedNodeIds.includes(id)) selectedIds.add(id);
        });
      }
      chargeBox.value = (+chargeInput.value).toFixed(0);
      distanceBox.value = (+distanceInput.value).toFixed(0);
      edgeScaleBox.value = (+edgeScaleInput.value).toFixed(2);
      nodeScaleBox.value = (+nodeScaleInput.value).toFixed(2);
      edgeRepelStrengthBox.value = (+edgeRepelStrengthInput.value).toFixed(3);
      edgeRepelRadiusBox.value = (+edgeRepelRadiusInput.value).toFixed(0);
      if (nApplied > 0) {
        setStatus(`Loaded saved layout (${nApplied} nodes).`);
        return true;
      }
      return false;
    } catch (err) {
      if (!String(err).includes("404")) {
        console.warn("Saved layout load failed:", err);
        setStatus("Saved layout could not be loaded.", true);
      }
      return false;
    }
  }

  function collectLayoutPayload() {
    const layoutNodes = nodes
      .filter((n) => Number.isFinite(n.x) && Number.isFinite(n.y))
      .map((n) => ({
        id: n.id,
        x: +(Number.isFinite(n.fx) ? n.fx : n.x).toFixed(3),
        y: +(Number.isFinite(n.fy) ? n.fy : n.y).toFixed(3),
        pinned: Number.isFinite(n.fx) && Number.isFinite(n.fy),
      }));
    return {
      saved_at: new Date().toISOString(),
      charge: +chargeInput.value,
      distance: +distanceInput.value,
      layout_mode: layoutModeInput.value,
      focus_node: focusNodeInput.value,
      fit_to_antarctica: fitToAntarcticaInput.checked,
      edge_repel_strength: +edgeRepelStrengthInput.value,
      edge_repel_radius: +edgeRepelRadiusInput.value,
      edge_scale: +edgeScaleInput.value,
      node_scale: +nodeScaleInput.value,
      uniform_node_size: uniformNodeSizeInput.checked,
      pin_dragged: pinDraggedInput.checked,
      box_select: boxSelectInput.checked,
      selected_ids: [...selectedIds],
      nodes: layoutNodes,
    };
  }

  const loadedSavedLayout = await loadSavedLayout();

  let centerX = 0;
  let centerY = 0;
  const themeCenters = new Map();
  const egoTargets = new Map();
  const antarcticaTargets = new Map();
  let antarcticaMaskPoints = [];
  let antarcticaMaskQuadtree = null;
  let antarcticaMaskImage = null;
  const mstShellTargets = new Map();

  const activeThemes = Object.keys(data.theme_colors).filter((theme) =>
    nodes.some((node) => node.theme === theme)
  );

  function updateThemeCenters() {
    if (activeThemes.length === 0) return;
    const width = panel.clientWidth || 1;
    const height = panel.clientHeight || 1;
    const radius = Math.max(90, Math.min(width, height) * 0.26);
    if (activeThemes.length === 1) {
      themeCenters.set(activeThemes[0], { x: centerX, y: centerY });
      return;
    }
    activeThemes.forEach((theme, idx) => {
      const angle = (2 * Math.PI * idx) / activeThemes.length - Math.PI / 2;
      themeCenters.set(theme, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      });
    });
  }

  function getThemeCenter(theme) {
    return themeCenters.get(theme) || { x: centerX, y: centerY };
  }

  function updateEgoTargets() {
    egoTargets.clear();
    const focusId = focusNodeInput.value;
    if (!focusId || !neighborMap.has(focusId)) return;

    const dist = new Map();
    const queue = [focusId];
    dist.set(focusId, 0);
    for (let i = 0; i < queue.length; i += 1) {
      const cur = queue[i];
      const curDist = dist.get(cur);
      for (const nxt of neighborMap.get(cur) || []) {
        if (!dist.has(nxt)) {
          dist.set(nxt, curDist + 1);
          queue.push(nxt);
        }
      }
    }

    const ring1 = [];
    const ring2 = [];
    const ring3 = [];
    nodes.forEach((n) => {
      if (n.id === focusId) return;
      const d = dist.has(n.id) ? dist.get(n.id) : Number.POSITIVE_INFINITY;
      if (d === 1) ring1.push(n);
      else if (d === 2) ring2.push(n);
      else ring3.push(n);
    });

    const nodeSort = (a, b) => {
      const themeCmp = String(a.theme).localeCompare(String(b.theme));
      return themeCmp !== 0 ? themeCmp : String(a.id).localeCompare(String(b.id));
    };
    ring1.sort(nodeSort);
    ring2.sort(nodeSort);
    ring3.sort(nodeSort);

    egoTargets.set(focusId, { x: centerX, y: centerY });
    const base = Math.max(70, Math.min(panel.clientWidth || 1, panel.clientHeight || 1) * 0.14);

    function placeRing(ringNodes, radius, phase) {
      if (ringNodes.length === 0) return;
      ringNodes.forEach((node, idx) => {
        const theta = phase + (2 * Math.PI * idx) / ringNodes.length;
        egoTargets.set(node.id, {
          x: centerX + radius * Math.cos(theta),
          y: centerY + radius * Math.sin(theta),
        });
      });
    }

    placeRing(ring1, base, -Math.PI / 2);
    placeRing(ring2, base * 1.95, -Math.PI / 3);
    placeRing(ring3, base * 2.85, -Math.PI / 4);
  }

  function getEgoTarget(id) {
    return egoTargets.get(id) || { x: centerX, y: centerY };
  }

  function getAntarcticaPlacement(width, height) {
    if (!antarcticaImg) return null;
    const imgW = antarcticaImg.width || 1;
    const imgH = antarcticaImg.height || 1;
    let drawW = width * ANTARCTICA_BG_SCALE;
    let drawH = drawW * (imgH / imgW);
    const maxH = height * 0.92;
    if (drawH > maxH) {
      drawH = maxH;
      drawW = drawH * (imgW / imgH);
    }
    const x = (width - drawW) * 0.5;
    const y = (height - drawH) * 0.5;
    return { x, y, w: drawW, h: drawH };
  }

  function sampleSpreadPoints(points, k) {
    if (points.length <= k) return points.slice();
    const selected = [];
    const minDist2 = new Float64Array(points.length);
    minDist2.fill(Number.POSITIVE_INFINITY);

    // Start from point closest to center.
    let bestIdx = 0;
    let bestD2 = Number.POSITIVE_INFINITY;
    for (let i = 0; i < points.length; i += 1) {
      const dx = points[i][0] - centerX;
      const dy = points[i][1] - centerY;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestIdx = i;
      }
    }

    for (let iter = 0; iter < k; iter += 1) {
      const p = points[bestIdx];
      selected.push(p);

      let nextIdx = 0;
      let nextScore = -1;
      for (let i = 0; i < points.length; i += 1) {
        const dx = points[i][0] - p[0];
        const dy = points[i][1] - p[1];
        const d2 = dx * dx + dy * dy;
        if (d2 < minDist2[i]) minDist2[i] = d2;
        if (minDist2[i] > nextScore) {
          nextScore = minDist2[i];
          nextIdx = i;
        }
      }
      bestIdx = nextIdx;
    }
    return selected;
  }

  function updateAntarcticaTargets() {
    antarcticaTargets.clear();
    antarcticaMaskPoints = [];
    antarcticaMaskQuadtree = null;
    antarcticaMaskImage = null;
    if (!antarcticaImg) return;

    const width = Math.max(1, Math.round(panel.clientWidth || 1));
    const height = Math.max(1, Math.round(panel.clientHeight || 1));
    const placement = getAntarcticaPlacement(width, height);
    if (!placement) return;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(antarcticaImg, placement.x, placement.y, placement.w, placement.h);

    const image = ctx.getImageData(0, 0, width, height);
    antarcticaMaskImage = image;

    const stride = Math.max(3, Math.round(Math.min(width, height) / 220));
    const points = [];
    for (let y = 0; y < height; y += stride) {
      for (let x = 0; x < width; x += stride) {
        const idx = (y * width + x) * 4 + 3;
        if (image.data[idx] > ANTARCTICA_ALPHA_THRESHOLD) {
          points.push([x, y]);
        }
      }
    }
    if (points.length === 0) return;

    antarcticaMaskPoints = points;
    antarcticaMaskQuadtree = d3.quadtree()
      .x((d) => d[0])
      .y((d) => d[1])
      .addAll(points);

    const sampled = sampleSpreadPoints(points, nodes.length);
    const nodeOrder = nodes
      .slice()
      .sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const targetOrder = sampled
      .slice()
      .sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]));

    for (let i = 0; i < nodeOrder.length; i += 1) {
      const t = targetOrder[i % targetOrder.length];
      antarcticaTargets.set(nodeOrder[i].id, { x: t[0], y: t[1] });
    }
  }

  function getAntarcticaTarget(id) {
    return antarcticaTargets.get(id) || { x: centerX, y: centerY };
  }

  function isInsideAntarcticaMask(x, y) {
    if (!antarcticaMaskImage) return true;
    const xi = Math.round(x);
    const yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= antarcticaMaskImage.width || yi >= antarcticaMaskImage.height) {
      return false;
    }
    const idx = (yi * antarcticaMaskImage.width + xi) * 4 + 3;
    return antarcticaMaskImage.data[idx] > ANTARCTICA_ALPHA_THRESHOLD;
  }

  function updateMstShellTargets() {
    mstShellTargets.clear();
    const root = focusNodeInput.value || sortedNodeIds[0];
    if (!root) return;

    const dist = new Map();
    const queue = [root];
    dist.set(root, 0);
    for (let i = 0; i < queue.length; i += 1) {
      const cur = queue[i];
      const curDist = dist.get(cur);
      for (const nxt of shellNeighborMap.get(cur) || []) {
        if (!dist.has(nxt)) {
          dist.set(nxt, curDist + 1);
          queue.push(nxt);
        }
      }
    }

    // Include any disconnected leftovers on an outer shell.
    const maxDepth = Math.max(0, ...dist.values());
    nodes.forEach((n) => {
      if (!dist.has(n.id)) dist.set(n.id, maxDepth + 1);
    });

    const byDepth = new Map();
    nodes.forEach((n) => {
      const d = dist.get(n.id);
      if (!byDepth.has(d)) byDepth.set(d, []);
      byDepth.get(d).push(n);
    });

    const shellStep = Math.max(56, Math.min(panel.clientWidth || 1, panel.clientHeight || 1) * 0.09);
    [...byDepth.keys()].sort((a, b) => a - b).forEach((depth) => {
      const shellNodes = byDepth.get(depth).sort((a, b) => String(a.id).localeCompare(String(b.id)));
      if (depth === 0) {
        mstShellTargets.set(root, { x: centerX, y: centerY });
        return;
      }
      const r = shellStep * depth;
      shellNodes.forEach((n, idx) => {
        const theta = -Math.PI / 2 + (2 * Math.PI * idx) / shellNodes.length;
        mstShellTargets.set(n.id, {
          x: centerX + r * Math.cos(theta),
          y: centerY + r * Math.sin(theta),
        });
      });
    });
  }

  function getMstShellTarget(id) {
    return mstShellTargets.get(id) || { x: centerX, y: centerY };
  }

  function applyCenterForces(reheat = true) {
    simulation.force("center", d3.forceCenter(centerX, centerY));
    if (reheat) simulation.alpha(0.25).restart();
  }

  function resize() {
    const width = panel.clientWidth;
    const height = panel.clientHeight;
    centerX = width / 2;
    centerY = height / 2;
    updateThemeCenters();
    updateEgoTargets();
    updateMstShellTargets();
    updateAntarcticaTargets();
    svg.attr("viewBox", [0, 0, width, height]);
    applyCenterForces(true);
    applyLayoutMode(false);
  }

  const linkWidthBase = d3.scaleLinear().domain(d3.extent(links, (d) => d.weight)).range([0.8, 4.2]);
  const nodeRadiusBase = d3.scaleLinear()
    .domain(d3.extent(nodes, (d) => d.weighted_degree))
    .range([4, 11]);
  const uniformRadiusBase = d3.mean(nodes, (d) => nodeRadiusBase(d.weighted_degree)) || 7;

  function getEdgeScale() {
    return +edgeScaleInput.value;
  }

  function getCharge() {
    return +chargeInput.value;
  }

  function getDistance() {
    return +distanceInput.value;
  }

  function getEdgeRepelStrengthBase() {
    return +edgeRepelStrengthInput.value;
  }

  function getEdgeRepelRadiusBase() {
    return +edgeRepelRadiusInput.value;
  }

  function getNodeScale() {
    return +nodeScaleInput.value;
  }

  function linkStrokeWidth(d) {
    return linkWidthBase(d.weight) * getEdgeScale();
  }

  function isPinned(nodeDatum) {
    return Number.isFinite(nodeDatum.fx) && Number.isFinite(nodeDatum.fy);
  }

  function setPinned(nodeDatum, pinned) {
    if (pinned) {
      nodeDatum.fx = Number.isFinite(nodeDatum.x) ? nodeDatum.x : 0;
      nodeDatum.fy = Number.isFinite(nodeDatum.y) ? nodeDatum.y : 0;
    } else {
      nodeDatum.fx = null;
      nodeDatum.fy = null;
    }
  }

  function nodeCircleRadius(d) {
    if (uniformNodeSizeInput.checked) {
      return uniformRadiusBase * getNodeScale();
    }
    return nodeRadiusBase(d.weighted_degree) * getNodeScale();
  }

  function clampToInputBounds(input, value) {
    const min = Number(input.min);
    const max = Number(input.max);
    let out = Number.isFinite(value) ? value : Number(input.value);
    if (Number.isFinite(min)) out = Math.max(min, out);
    if (Number.isFinite(max)) out = Math.min(max, out);
    return out;
  }

  function setEdgeScale(value) {
    const clamped = clampToInputBounds(edgeScaleInput, value);
    edgeScaleInput.value = clamped.toFixed(2);
    edgeScaleBox.value = clamped.toFixed(2);
  }

  function setCharge(value) {
    const clamped = clampToInputBounds(chargeInput, value);
    chargeInput.value = clamped.toFixed(0);
    chargeBox.value = clamped.toFixed(0);
  }

  function setDistance(value) {
    const clamped = clampToInputBounds(distanceInput, value);
    distanceInput.value = clamped.toFixed(0);
    distanceBox.value = clamped.toFixed(0);
  }

  function setNodeScale(value) {
    const clamped = clampToInputBounds(nodeScaleInput, value);
    nodeScaleInput.value = clamped.toFixed(2);
    nodeScaleBox.value = clamped.toFixed(2);
  }

  function setEdgeRepelStrength(value) {
    const clamped = clampToInputBounds(edgeRepelStrengthInput, value);
    edgeRepelStrengthInput.value = clamped.toFixed(3);
    edgeRepelStrengthBox.value = clamped.toFixed(3);
  }

  function setEdgeRepelRadius(value) {
    const clamped = clampToInputBounds(edgeRepelRadiusInput, value);
    edgeRepelRadiusInput.value = clamped.toFixed(0);
    edgeRepelRadiusBox.value = clamped.toFixed(0);
  }

  function shareEndpoint(l1, l2) {
    return (
      l1.source === l2.source ||
      l1.source === l2.target ||
      l1.target === l2.source ||
      l1.target === l2.target
    );
  }

  function createEdgeRepelForce(linkArray, strength = 0.08, radius = 190) {
    let localStrength = strength;
    let localRadius = radius;
    const eps = 1e-6;

    function force(alpha) {
      const r2 = localRadius * localRadius;
      const count = linkArray.length;
      for (let i = 0; i < count; i += 1) {
        const li = linkArray[i];
        const lix = (li.source.x + li.target.x) * 0.5;
        const liy = (li.source.y + li.target.y) * 0.5;
        for (let j = i + 1; j < count; j += 1) {
          const lj = linkArray[j];
          if (shareEndpoint(li, lj)) continue;

          const ljx = (lj.source.x + lj.target.x) * 0.5;
          const ljy = (lj.source.y + lj.target.y) * 0.5;
          let dx = ljx - lix;
          let dy = ljy - liy;
          let dist2 = dx * dx + dy * dy;
          if (dist2 >= r2) continue;
          if (dist2 < eps) {
            dx = (Math.random() - 0.5) * 1e-3;
            dy = (Math.random() - 0.5) * 1e-3;
            dist2 = dx * dx + dy * dy + eps;
          }

          const dist = Math.sqrt(dist2);
          const overlap = localRadius - dist;
          if (overlap <= 0) continue;

          const mag = Math.min(2.4, (overlap / localRadius) * localStrength * alpha);
          const nx = dx / dist;
          const ny = dy / dist;

          // Push both edges apart by nudging both endpoints of each link.
          li.source.vx -= nx * mag;
          li.source.vy -= ny * mag;
          li.target.vx -= nx * mag;
          li.target.vy -= ny * mag;
          lj.source.vx += nx * mag;
          lj.source.vy += ny * mag;
          lj.target.vx += nx * mag;
          lj.target.vy += ny * mag;
        }
      }
    }

    force.strength = (value) => {
      if (value === undefined) return localStrength;
      localStrength = +value;
      return force;
    };

    force.radius = (value) => {
      if (value === undefined) return localRadius;
      localRadius = +value;
      return force;
    };

    return force;
  }

  function createAntarcticaFitForce(strength = 0.12) {
    let localStrength = strength;
    function force(alpha) {
      if (!fitToAntarcticaInput.checked) return;
      for (const n of nodes) {
        const t = getAntarcticaTarget(n.id);
        n.vx += (t.x - n.x) * localStrength * alpha;
        n.vy += (t.y - n.y) * localStrength * alpha;
      }
    }
    force.strength = (value) => {
      if (value === undefined) return localStrength;
      localStrength = +value;
      return force;
    };
    return force;
  }

  function createAntarcticaContainForce(strength = 0.2) {
    let localStrength = strength;
    function force(alpha) {
      if (!fitToAntarcticaInput.checked || !antarcticaMaskQuadtree) return;
      for (const n of nodes) {
        if (isInsideAntarcticaMask(n.x, n.y)) continue;
        const nearest = antarcticaMaskQuadtree.find(n.x, n.y);
        if (!nearest) continue;
        n.vx += (nearest[0] - n.x) * localStrength * alpha;
        n.vy += (nearest[1] - n.y) * localStrength * alpha;
      }
    }
    force.strength = (value) => {
      if (value === undefined) return localStrength;
      localStrength = +value;
      return force;
    };
    return force;
  }

  const g = svg.append("g");
  const linkLayer = g.append("g").attr("stroke-linecap", "round");
  const nodeLayer = g.append("g");
  const labelLayer = g.append("g");
  const overlayLayer = svg.append("g");
  const selectionRect = overlayLayer
    .append("rect")
    .attr("class", "selection-rect")
    .style("display", "none");

  let boxSelecting = false;
  let selectStart = { x: 0, y: 0 };
  let dragGroup = [];
  let dragOffsets = new Map();

  const link = linkLayer
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", (d) => (d.kind === "mst" ? "#64748b" : "#94a3b8"))
    .attr("stroke-opacity", (d) => (d.kind === "mst" ? 0.45 : 0.32))
    .attr("stroke-width", (d) => linkStrokeWidth(d));

  const node = nodeLayer
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", (d) => nodeCircleRadius(d))
    .attr("fill", (d) => d.color)
    .attr("stroke", "#111827")
    .attr("stroke-width", 0.8)
    .call(
      d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded)
    );

  const labels = labelLayer
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("font-size", 10)
    .attr("fill", "#111827")
    .attr("dx", 8)
    .attr("dy", 3)
    .text((d) => d.id)
    .style("pointer-events", "none")
    .style("opacity", 0);

  function refreshNodeStyles() {
    node
      .attr("stroke", (d) => (selectedIds.has(d.id) ? "#0284c7" : "#111827"))
      .attr("stroke-dasharray", (d) => (isPinned(d) ? "2,1" : null))
      .attr("stroke-width", (d) => {
        if (selectedIds.has(d.id) && isPinned(d)) return 2.2;
        if (selectedIds.has(d.id)) return 1.8;
        if (isPinned(d)) return 1.2;
        return 0.8;
      });
  }

  node
    .on("click", (event, d) => {
      if (event.defaultPrevented) return;
      event.stopPropagation();
      if (boxSelectInput.checked) {
        if (!event.shiftKey) selectedIds.clear();
        if (selectedIds.has(d.id)) {
          if (event.shiftKey) selectedIds.delete(d.id);
        } else {
          selectedIds.add(d.id);
        }
        refreshNodeStyles();
        setStatus(`Selected: ${selectedIds.size}`);
        return;
      }
      setPinned(d, !isPinned(d));
      refreshNodeStyles();
      simulation.alpha(0.2).restart();
    })
    .on("mouseenter", (event, d) => {
      node.attr("opacity", 0.16);
      link.attr("opacity", 0.08);
      labels.style("opacity", (n) => (n.id === d.id || showLabels.checked ? 1 : 0));

      d3.select(event.currentTarget).attr("opacity", 1).attr("stroke-width", 2.3);
      link
        .filter((l) => l.source.id === d.id || l.target.id === d.id)
        .attr("opacity", 1)
        .attr("stroke", "#0f172a");
      node
        .filter((n) =>
          links.some(
            (l) =>
              (l.source.id === d.id && l.target.id === n.id) ||
              (l.target.id === d.id && l.source.id === n.id)
          )
        )
        .attr("opacity", 0.95);

      info.innerHTML = `
        <div><strong>${d.id}</strong></div>
        <div class="muted">${d.theme}</div>
        <div>Degree: ${d.degree}</div>
        <div>Weighted degree: ${d.weighted_degree.toFixed(3)}</div>
        <div>Pinned: ${isPinned(d) ? "yes" : "no"}</div>
        <div>Selected: ${selectedIds.has(d.id) ? "yes" : "no"}</div>
      `;
    })
    .on("mouseleave", () => {
      node.attr("opacity", 1);
      link.attr("opacity", 1).attr("stroke", (d) => (d.kind === "mst" ? "#64748b" : "#94a3b8"));
      labels.style("opacity", showLabels.checked ? 0.9 : 0);
      info.textContent = "Hover a node for details.";
      refreshNodeStyles();
    });

  function finalizeBoxSelection(event) {
    if (!boxSelecting) return;
    boxSelecting = false;
    selectionRect.style("display", "none");

    const [x, y] = d3.pointer(event, svg.node());
    const x0 = Math.min(selectStart.x, x);
    const y0 = Math.min(selectStart.y, y);
    const x1 = Math.max(selectStart.x, x);
    const y1 = Math.max(selectStart.y, y);

    const width = x1 - x0;
    const height = y1 - y0;
    if (width < 4 && height < 4) {
      if (!event.shiftKey) selectedIds.clear();
      refreshNodeStyles();
      return;
    }

    const hits = nodes.filter((n) => n.x >= x0 && n.x <= x1 && n.y >= y0 && n.y <= y1).map((n) => n.id);
    if (!event.shiftKey) selectedIds.clear();
    hits.forEach((id) => selectedIds.add(id));
    refreshNodeStyles();
    setStatus(`Selected: ${selectedIds.size}`);
  }

  svg
    .on("pointerdown", (event) => {
      if (!boxSelectInput.checked) return;
      if (event.target && String(event.target.tagName).toLowerCase() === "circle") return;
      const [x, y] = d3.pointer(event, svg.node());
      boxSelecting = true;
      selectStart = { x, y };
      selectionRect
        .style("display", null)
        .attr("x", x)
        .attr("y", y)
        .attr("width", 0)
        .attr("height", 0);
    })
    .on("pointermove", (event) => {
      if (!boxSelecting) return;
      const [x, y] = d3.pointer(event, svg.node());
      const x0 = Math.min(selectStart.x, x);
      const y0 = Math.min(selectStart.y, y);
      const x1 = Math.max(selectStart.x, x);
      const y1 = Math.max(selectStart.y, y);
      selectionRect
        .attr("x", x0)
        .attr("y", y0)
        .attr("width", x1 - x0)
        .attr("height", y1 - y0);
    })
    .on("pointerup", (event) => {
      finalizeBoxSelection(event);
    })
    .on("pointerleave", (event) => {
      finalizeBoxSelection(event);
    })
    .on("click", (event) => {
      if (!boxSelectInput.checked) return;
      if (event.target && String(event.target.tagName).toLowerCase() === "circle") return;
      if (!event.shiftKey) {
        selectedIds.clear();
        refreshNodeStyles();
      }
    });

  const linkForce = d3.forceLink(links)
    .id((d) => d.id)
    .distance(() => getDistance())
    .strength((d) => (d.kind === "mst" ? 0.9 : 0.45));
  const edgeRepelForce = createEdgeRepelForce(
    links,
    getEdgeRepelStrengthBase() || 0.09,
    getEdgeRepelRadiusBase() || 190
  );
  const antarcticaFitForce = createAntarcticaFitForce(0.12);
  const antarcticaContainForce = createAntarcticaContainForce(0.28);

  const simulation = d3.forceSimulation(nodes)
    .force(
      "link",
      linkForce
    )
    .force("charge", d3.forceManyBody().strength(getCharge()))
    .force("center", d3.forceCenter(0, 0))
    .force("bubble-x", d3.forceX(() => centerX).strength(0))
    .force("bubble-y", d3.forceY(() => centerY).strength(0))
    .force("ego-x", d3.forceX(() => centerX).strength(0))
    .force("ego-y", d3.forceY(() => centerY).strength(0))
    .force("mst-shell-x", d3.forceX(() => centerX).strength(0))
    .force("mst-shell-y", d3.forceY(() => centerY).strength(0))
    .force("edge-repel", edgeRepelForce)
    .force("antarctica-fit", antarcticaFitForce)
    .force("antarctica-contain", antarcticaContainForce)
    .force("collide", d3.forceCollide((d) => nodeCircleRadius(d) + 2))
    .on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

      labels.attr("x", (d) => d.x).attr("y", (d) => d.y);
    });

  function applyLayoutMode(reheat = true) {
    const mode = layoutModeInput.value;
    const baseRepelStrength = getEdgeRepelStrengthBase();
    const baseRepelRadius = getEdgeRepelRadiusBase();
    const fitScale = fitToAntarcticaInput.checked ? 1.0 : 0.0;
    if (mode === "bubble") {
      linkForce
        .distance(() => Math.max(10, getDistance() * 0.55))
        .strength((d) => (d.kind === "mst" ? 0.18 : 0.08));
      edgeRepelForce
        .strength(baseRepelStrength * 0.56)
        .radius(baseRepelRadius * 0.74);
      simulation.force("bubble-x", d3.forceX((d) => getThemeCenter(d.theme).x).strength(0.30));
      simulation.force("bubble-y", d3.forceY((d) => getThemeCenter(d.theme).y).strength(0.30));
      simulation.force("ego-x", d3.forceX(() => centerX).strength(0));
      simulation.force("ego-y", d3.forceY(() => centerY).strength(0));
      simulation.force("mst-shell-x", d3.forceX(() => centerX).strength(0));
      simulation.force("mst-shell-y", d3.forceY(() => centerY).strength(0));
      antarcticaFitForce.strength(0.09 * fitScale);
      antarcticaContainForce.strength(0.24 * fitScale);
      link
        .attr("stroke-opacity", (d) => (d.kind === "mst" ? 0.28 : 0.16))
        .attr("stroke", (d) => (d.kind === "mst" ? "#64748b" : "#94a3b8"));
    } else if (mode === "ego") {
      updateEgoTargets();
      linkForce
        .distance(() => Math.max(12, getDistance() * 0.62))
        .strength((d) => (d.kind === "mst" ? 0.35 : 0.16));
      edgeRepelForce
        .strength(baseRepelStrength * 0.78)
        .radius(baseRepelRadius * 0.87);
      simulation.force("bubble-x", d3.forceX(() => centerX).strength(0));
      simulation.force("bubble-y", d3.forceY(() => centerY).strength(0));
      simulation.force("ego-x", d3.forceX((d) => getEgoTarget(d.id).x).strength(0.58));
      simulation.force("ego-y", d3.forceY((d) => getEgoTarget(d.id).y).strength(0.58));
      simulation.force("mst-shell-x", d3.forceX(() => centerX).strength(0));
      simulation.force("mst-shell-y", d3.forceY(() => centerY).strength(0));
      antarcticaFitForce.strength(0.12 * fitScale);
      antarcticaContainForce.strength(0.26 * fitScale);
      link
        .attr("stroke-opacity", (d) => (d.kind === "mst" ? 0.34 : 0.2))
        .attr("stroke", (d) => (d.kind === "mst" ? "#475569" : "#94a3b8"));
    } else if (mode === "mst-shell") {
      updateMstShellTargets();
      linkForce
        .distance(() => Math.max(14, getDistance() * 0.66))
        .strength((d) => (d.kind === "mst" ? 0.42 : 0.14));
      edgeRepelForce
        .strength(baseRepelStrength * 0.72)
        .radius(baseRepelRadius * 0.92);
      simulation.force("bubble-x", d3.forceX(() => centerX).strength(0));
      simulation.force("bubble-y", d3.forceY(() => centerY).strength(0));
      simulation.force("ego-x", d3.forceX(() => centerX).strength(0));
      simulation.force("ego-y", d3.forceY(() => centerY).strength(0));
      simulation.force("mst-shell-x", d3.forceX((d) => getMstShellTarget(d.id).x).strength(0.72));
      simulation.force("mst-shell-y", d3.forceY((d) => getMstShellTarget(d.id).y).strength(0.72));
      antarcticaFitForce.strength(0.11 * fitScale);
      antarcticaContainForce.strength(0.28 * fitScale);
      link
        .attr("stroke-opacity", (d) => (d.kind === "mst" ? 0.38 : 0.18))
        .attr("stroke", (d) => (d.kind === "mst" ? "#334155" : "#94a3b8"));
    } else {
      linkForce
        .distance(() => getDistance())
        .strength((d) => (d.kind === "mst" ? 0.9 : 0.45));
      edgeRepelForce
        .strength(baseRepelStrength)
        .radius(baseRepelRadius);
      simulation.force("bubble-x", d3.forceX(() => centerX).strength(0));
      simulation.force("bubble-y", d3.forceY(() => centerY).strength(0));
      simulation.force("ego-x", d3.forceX(() => centerX).strength(0));
      simulation.force("ego-y", d3.forceY(() => centerY).strength(0));
      simulation.force("mst-shell-x", d3.forceX(() => centerX).strength(0));
      simulation.force("mst-shell-y", d3.forceY(() => centerY).strength(0));
      antarcticaFitForce.strength(0.10 * fitScale);
      antarcticaContainForce.strength(0.30 * fitScale);
      link
        .attr("stroke-opacity", (d) => (d.kind === "mst" ? 0.45 : 0.32))
        .attr("stroke", (d) => (d.kind === "mst" ? "#64748b" : "#94a3b8"));
    }
    if (reheat) simulation.alpha(0.28).restart();
  }

  function applyVisualScaleSettings(reheat = true) {
    edgeScaleBox.value = getEdgeScale().toFixed(2);
    nodeScaleBox.value = getNodeScale().toFixed(2);
    link.attr("stroke-width", (d) => linkStrokeWidth(d));
    node.attr("r", (d) => nodeCircleRadius(d));
    simulation.force("collide", d3.forceCollide((d) => nodeCircleRadius(d) + 2));
    if (reheat) simulation.alpha(0.25).restart();
  }

  showLabels.addEventListener("change", () => {
    labels.style("opacity", showLabels.checked ? 0.9 : 0);
  });

  chargeInput.addEventListener("input", () => {
    setCharge(+chargeInput.value);
    simulation.force("charge").strength(getCharge());
    simulation.alpha(0.35).restart();
  });

  chargeBox.addEventListener("input", () => {
    setCharge(+chargeBox.value);
    simulation.force("charge").strength(getCharge());
    simulation.alpha(0.35).restart();
  });

  distanceInput.addEventListener("input", () => {
    setDistance(+distanceInput.value);
    applyLayoutMode(true);
  });

  distanceBox.addEventListener("input", () => {
    setDistance(+distanceBox.value);
    applyLayoutMode(true);
  });

  layoutModeInput.addEventListener("change", () => {
    applyLayoutMode(true);
  });

  focusNodeInput.addEventListener("change", () => {
    if (layoutModeInput.value === "ego" || layoutModeInput.value === "mst-shell") {
      applyLayoutMode(true);
    }
  });

  edgeScaleInput.addEventListener("input", () => {
    setEdgeScale(+edgeScaleInput.value);
    applyVisualScaleSettings(true);
  });

  nodeScaleInput.addEventListener("input", () => {
    setNodeScale(+nodeScaleInput.value);
    applyVisualScaleSettings(true);
  });

  edgeScaleBox.addEventListener("input", () => {
    setEdgeScale(+edgeScaleBox.value);
    applyVisualScaleSettings(true);
  });

  edgeRepelStrengthInput.addEventListener("input", () => {
    setEdgeRepelStrength(+edgeRepelStrengthInput.value);
    applyLayoutMode(true);
  });

  edgeRepelStrengthBox.addEventListener("input", () => {
    setEdgeRepelStrength(+edgeRepelStrengthBox.value);
    applyLayoutMode(true);
  });

  edgeRepelRadiusInput.addEventListener("input", () => {
    setEdgeRepelRadius(+edgeRepelRadiusInput.value);
    applyLayoutMode(true);
  });

  edgeRepelRadiusBox.addEventListener("input", () => {
    setEdgeRepelRadius(+edgeRepelRadiusBox.value);
    applyLayoutMode(true);
  });

  nodeScaleBox.addEventListener("input", () => {
    setNodeScale(+nodeScaleBox.value);
    applyVisualScaleSettings(true);
  });

  uniformNodeSizeInput.addEventListener("change", () => {
    applyVisualScaleSettings(true);
  });

  fitToAntarcticaInput.addEventListener("change", () => {
    if (fitToAntarcticaInput.checked) {
      updateAntarcticaTargets();
    }
    applyLayoutMode(true);
  });

  boxSelectInput.addEventListener("change", () => {
    if (!boxSelectInput.checked) {
      boxSelecting = false;
      selectionRect.style("display", "none");
    }
  });

  clearSelectionButton.addEventListener("click", () => {
    selectedIds.clear();
    refreshNodeStyles();
    setStatus("Selection cleared.");
  });

  clearPinsButton.addEventListener("click", () => {
    nodes.forEach((n) => setPinned(n, false));
    refreshNodeStyles();
    simulation.alpha(0.5).restart();
    setStatus("All pins cleared.");
  });

  saveButton.addEventListener("click", async () => {
    saveButton.disabled = true;
    setStatus("Saving layout...");
    try {
      const payload = collectLayoutPayload();
      const response = await fetch("save-layout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      setStatus(`Layout saved (${payload.nodes.length} nodes).`);
    } catch (err) {
      console.error(err);
      setStatus(`Save failed: ${err}`, true);
    } finally {
      saveButton.disabled = false;
    }
  });

  resetButton.addEventListener("click", () => {
    simulation.alpha(0.8).restart();
  });

  function dragStarted(event) {
    if (!event.active) simulation.alphaTarget(0.25).restart();
    const subject = event.subject;
    const moveAsGroup = selectedIds.has(subject.id) && selectedIds.size > 1;
    dragGroup = moveAsGroup ? nodes.filter((n) => selectedIds.has(n.id)) : [subject];
    dragOffsets = new Map(
      dragGroup.map((n) => [
        n.id,
        {
          dx: (Number.isFinite(n.fx) ? n.fx : n.x) - subject.x,
          dy: (Number.isFinite(n.fy) ? n.fy : n.y) - subject.y,
        },
      ])
    );
    dragGroup.forEach((n) => {
      n.fx = Number.isFinite(n.fx) ? n.fx : n.x;
      n.fy = Number.isFinite(n.fy) ? n.fy : n.y;
    });
  }

  function dragged(event) {
    dragGroup.forEach((n) => {
      const off = dragOffsets.get(n.id) || { dx: 0, dy: 0 };
      n.fx = event.x + off.dx;
      n.fy = event.y + off.dy;
    });
  }

  function dragEnded(event) {
    if (!event.active) simulation.alphaTarget(0);
    const targets = dragGroup.length ? dragGroup : [event.subject];
    targets.forEach((n) => {
      if (pinDraggedInput.checked) setPinned(n, true);
      else setPinned(n, false);
    });
    dragGroup = [];
    dragOffsets = new Map();
    refreshNodeStyles();
  }

  setEdgeScale(+edgeScaleInput.value);
  setEdgeRepelStrength(+edgeRepelStrengthInput.value);
  setEdgeRepelRadius(+edgeRepelRadiusInput.value);
  setCharge(+chargeInput.value);
  setDistance(+distanceInput.value);
  setNodeScale(+nodeScaleInput.value);
  updateThemeCenters();
  updateEgoTargets();
  updateMstShellTargets();
  applyLayoutMode(false);
  applyVisualScaleSettings(false);
  refreshNodeStyles();
  resize();
  new ResizeObserver(resize).observe(panel);
  if (loadedSavedLayout) {
    simulation.alpha(0.12).restart();
  }
}

main().catch((err) => {
  console.error(err);
  const info = document.getElementById("info");
  info.textContent = `Failed to load graph JSON: ${err}`;
});
