(() => {
  const TACTIC_COLORS = {
    "benign": new THREE.Color(0x64748b),
    "scans": new THREE.Color(0x00f5ff),
    "webshell/reverse_shell": new THREE.Color(0xff00aa),
    "cracking": new THREE.Color(0xffb800),
    "privilege_escalation": new THREE.Color(0xff3b30),
    "service_stop/dnsteal": new THREE.Color(0xa855f7),
    "anomaly": new THREE.Color(0xf43f5e),
  };

  const TYPE_TO_SHAPE = { process: 0.0, file: 1.0, socket: 2.0 };
  const ICON_MAP = {
    process: "/images/process.png?v=custom",
    file: "/images/file.png?v=custom",
    socket: "/images/socket.png?v=custom",
  };

  const state = {
    nodes: new Map(),
    edges: new Map(),
    orderedNodes: [],
    subgraphNodeSet: new Set(),
    isolateSubgraph: false,
    showLogs: true,
    hoveredId: null,
  };

  const container = document.getElementById("canvas-container");
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(54, window.innerWidth / window.innerHeight, 1, 4000);
  camera.position.set(0, 0, 390);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const graphGroup = new THREE.Group();
  scene.add(graphGroup);

  const nodeVertexShader = `
    attribute vec3 aColor;
    attribute float aAlpha;
    attribute float aSize;
    attribute float aShape;
    attribute float aMalicious;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vShape;
    varying float vMalicious;
    uniform float uTime;

    void main() {
      vColor = aColor;
      vAlpha = aAlpha;
      vShape = aShape;
      vMalicious = aMalicious;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      float pulse = aMalicious > 0.5 ? (1.0 + 0.16 * sin(uTime * 5.0)) : 1.0;
      gl_PointSize = aSize * pulse * (320.0 / max(45.0, -mvPosition.z));
      gl_Position = projectionMatrix * mvPosition;
    }
  `;

  const nodeFragmentShader = `
    varying vec3 vColor;
    varying float vAlpha;
    varying float vShape;
    varying float vMalicious;

    void main() {
      vec2 uv = gl_PointCoord * 2.0 - 1.0;
      float dist = 0.0;
      if (vShape < 0.5) {
        dist = length(uv);
      } else if (vShape < 1.5) {
        dist = max(abs(uv.x), abs(uv.y)) * 1.12;
      } else {
        dist = (abs(uv.x) + abs(uv.y)) * 0.95;
      }
      if (dist > 1.0) discard;

      float core = smoothstep(1.0, 0.58, dist);
      vec3 finalColor = vMalicious > 0.5 ? mix(vColor, vec3(1.0), smoothstep(0.52, 0.0, dist) * 0.42) : vColor;
      float alpha = vAlpha * (vMalicious > 0.5 ? mix(0.45, 1.0, core) : core);
      gl_FragColor = vec4(finalColor, alpha);
    }
  `;

  const nodeUniforms = { uTime: { value: 0.0 } };
  const nodeGeometry = new THREE.BufferGeometry();
  const nodeMaterial = new THREE.ShaderMaterial({
    uniforms: nodeUniforms,
    vertexShader: nodeVertexShader,
    fragmentShader: nodeFragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const nodePoints = new THREE.Points(nodeGeometry, nodeMaterial);
  nodePoints.frustumCulled = false;
  graphGroup.add(nodePoints);

  const edgeGeometry = new THREE.BufferGeometry();
  const edgeMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.62,
    blending: THREE.AdditiveBlending,
  });
  const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
  edgeLines.frustumCulled = false;
  graphGroup.add(edgeLines);

  const MAX_PARTICLES = 600;
  const particlePositions = new Float32Array(MAX_PARTICLES * 3);
  const particleColors = new Float32Array(MAX_PARTICLES * 3);
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
  particleGeometry.setAttribute("color", new THREE.BufferAttribute(particleColors, 3));
  const particleMaterial = new THREE.PointsMaterial({
    size: 6.5,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const attackParticles = new THREE.Points(particleGeometry, particleMaterial);
  attackParticles.frustumCulled = false;
  graphGroup.add(attackParticles);

  let activeAttackEdges = [];

  function seededOffset(idStr) {
    let h1 = 2166136261, h2 = 16777619, h3 = 374761393;
    for (let i = 0; i < idStr.length; i++) {
      const c = idStr.charCodeAt(i);
      h1 = Math.imul(h1 ^ c, 16777619);
      h2 = Math.imul(h2 ^ (c + i), 2246822519);
      h3 = Math.imul(h3 ^ (c * 7), 3266489917);
    }
    const theta = ((Math.abs(h1) % 3600) / 3600) * Math.PI * 2;
    const phi = Math.acos(2.0 * ((Math.abs(h2) % 1000) / 1000) - 1.0);
    const r = 20 + (Math.abs(h3) % 28);
    return new THREE.Vector3(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi) * 0.75,
      r * Math.sin(phi) * Math.sin(theta)
    );
  }

  function resolveActiveColor(attackClass, isActive) {
    if (!isActive) {
      return TACTIC_COLORS.benign;
    }
    if (attackClass && attackClass !== "benign" && TACTIC_COLORS[attackClass]) {
      return TACTIC_COLORS[attackClass];
    }
    return TACTIC_COLORS.anomaly;
  }

  function spawnPositionForNewNode(n, anchorMap) {
    const off = seededOffset(n.id);
    const anchor = anchorMap ? anchorMap.get(n.id) : null;
    if (anchor) {
      return anchor.clone().add(off.multiplyScalar(0.75));
    }
    const typeBiasY = n.type === "socket" ? 18 : (n.type === "file" ? -18 : 0);
    return new THREE.Vector3(off.x * 1.15, off.y + typeBiasY, off.z * 1.15);
  }

  function upsertNode(n, anchorMap) {
    const prev = state.nodes.get(n.id);
    if (prev) {
      prev.type = n.type;
      prev.label = n.label;
      prev.anomaly_score = n.anomaly_score;
      prev.recon_score = n.recon_score;
      prev.p_value = n.p_value;
      prev.attack_class = n.attack_class;
      prev.is_malicious = n.is_malicious;
    } else {
      state.nodes.set(n.id, {
        ...n,
        pos: spawnPositionForNewNode(n, anchorMap),
        vel: new THREE.Vector3(0, 0, 0),
        spawn: 0.08,
      });
    }
  }

  function edgeKey(e) {
    return e.id || `${e.source}->${e.type}->${e.target}`;
  }

  function rebuildTopologyAndVisuals() {
    const visibleNodes = [];
    for (const n of state.nodes.values()) {
      if (state.isolateSubgraph && state.subgraphNodeSet.size > 0 && !state.subgraphNodeSet.has(n.id)) {
        continue;
      }
      visibleNodes.push(n);
    }
    state.orderedNodes = visibleNodes;

    const count = visibleNodes.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const alphas = new Float32Array(count);
    const sizes = new Float32Array(count);
    const shapes = new Float32Array(count);
    const malFlags = new Float32Array(count);

    const visibleMap = new Map();
    for (let i = 0; i < count; i++) {
      const n = visibleNodes[i];
      visibleMap.set(n.id, n);
      positions[i * 3] = n.pos.x;
      positions[i * 3 + 1] = n.pos.y;
      positions[i * 3 + 2] = n.pos.z;

      const isMal = Boolean(n.is_malicious);
      const col = resolveActiveColor(n.attack_class, isMal);
      colors[i * 3] = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;

      const baseAlpha = isMal ? 0.98 : 0.22;
      const baseSize = isMal ? 22.0 : 11.0;
      alphas[i] = baseAlpha * n.spawn;
      sizes[i] = baseSize * (0.45 + 0.55 * n.spawn);
      shapes[i] = TYPE_TO_SHAPE[n.type] ?? 0.0;
      malFlags[i] = isMal ? 1.0 : 0.0;
    }

    nodeGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    nodeGeometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
    nodeGeometry.setAttribute("aAlpha", new THREE.BufferAttribute(alphas, 1));
    nodeGeometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    nodeGeometry.setAttribute("aShape", new THREE.BufferAttribute(shapes, 1));
    nodeGeometry.setAttribute("aMalicious", new THREE.BufferAttribute(malFlags, 1));

    const linePos = [];
    const lineCol = [];
    activeAttackEdges = [];

    for (const e of state.edges.values()) {
      const s = visibleMap.get(e.source);
      const t = visibleMap.get(e.target);
      if (!s || !t) continue;
      linePos.push(s.pos.x, s.pos.y, s.pos.z, t.pos.x, t.pos.y, t.pos.z);

      if (e.is_attack_path) {
        const c = resolveActiveColor(e.attack_class, true);
        lineCol.push(c.r, c.g, c.b, c.r, c.g, c.b);
        activeAttackEdges.push({ sNode: s, tNode: t, color: c });
      } else {
        lineCol.push(0.20, 0.25, 0.34, 0.20, 0.25, 0.34);
      }
    }

    edgeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePos, 3));
    edgeGeometry.setAttribute("color", new THREE.Float32BufferAttribute(lineCol, 3));

    document.getElementById("stat-counts").textContent = `${state.nodes.size} / ${state.edges.size}`;
  }

  function stepContinuousPhysics() {
    const arr = state.orderedNodes;
    const n = arr.length;
    if (n === 0) return;

    for (let i = 0; i < n; i++) {
      const u = arr[i];
      if (u.spawn < 1.0) {
        u.spawn = Math.min(1.0, u.spawn + 0.035);
      }
      const targetY = u.type === "socket" ? 20.0 : (u.type === "file" ? -20.0 : 0.0);
      u.vel.x += -u.pos.x * 0.0028;
      u.vel.y += (targetY - u.pos.y) * 0.0032;
      u.vel.z += -u.pos.z * 0.0028;
    }

    const stride = n > 280 ? Math.ceil(n / 180) : 1;
    for (let i = 0; i < n; i++) {
      const u = arr[i];
      for (let j = i + 1; j < n; j += stride) {
        const v = arr[j];
        const dx = u.pos.x - v.pos.x;
        const dy = u.pos.y - v.pos.y;
        const dz = u.pos.z - v.pos.z;
        const dSq = dx * dx + dy * dy + dz * dz;
        if (dSq > 0.01 && dSq < 2100.0) {
          const d = Math.sqrt(dSq);
          const rep = ((45.8 - d) / d) * 0.028 * stride;
          u.vel.x += dx * rep;
          u.vel.y += dy * rep;
          u.vel.z += dz * rep;
          v.vel.x -= dx * rep;
          v.vel.y -= dy * rep;
          v.vel.z -= dz * rep;
        }
      }
    }

    for (const e of state.edges.values()) {
      const s = state.nodes.get(e.source);
      const t = state.nodes.get(e.target);
      if (!s || !t) continue;
      const dx = t.pos.x - s.pos.x;
      const dy = t.pos.y - s.pos.y;
      const dz = t.pos.z - s.pos.z;
      const dist = Math.max(1.0, Math.sqrt(dx * dx + dy * dy + dz * dz));
      const spring = ((dist - 56.0) / dist) * 0.016;
      s.vel.x += dx * spring;
      s.vel.y += dy * spring;
      s.vel.z += dz * spring;
      t.vel.x -= dx * spring;
      t.vel.y -= dy * spring;
      t.vel.z -= dz * spring;
    }

    let cx = 0.0, cy = 0.0, cz = 0.0;
    for (let i = 0; i < n; i++) {
      const u = arr[i];
      u.vel.multiplyScalar(0.82);
      const speed = u.vel.length();
      if (speed > 3.2) {
        u.vel.multiplyScalar(3.2 / speed);
      }
      u.pos.add(u.vel);
      cx += u.pos.x;
      cy += u.pos.y;
      cz += u.pos.z;
    }

    cx /= n;
    cy /= n;
    cz /= n;
    for (let i = 0; i < n; i++) {
      arr[i].pos.x -= cx * 0.08;
      arr[i].pos.y -= cy * 0.08;
      arr[i].pos.z -= cz * 0.08;
    }

    const posAttr = nodeGeometry.getAttribute("position");
    const alphaAttr = nodeGeometry.getAttribute("aAlpha");
    const sizeAttr = nodeGeometry.getAttribute("aSize");
    if (posAttr && posAttr.count === n) {
      for (let i = 0; i < n; i++) {
        const u = arr[i];
        posAttr.array[i * 3] = u.pos.x;
        posAttr.array[i * 3 + 1] = u.pos.y;
        posAttr.array[i * 3 + 2] = u.pos.z;
        const isMal = Boolean(u.is_malicious);
        alphaAttr.array[i] = (isMal ? 0.98 : 0.22) * u.spawn;
        sizeAttr.array[i] = (isMal ? 22.0 : 11.0) * (0.45 + 0.55 * u.spawn);
      }
      posAttr.needsUpdate = true;
      alphaAttr.needsUpdate = true;
      sizeAttr.needsUpdate = true;
    }

    const edgePosAttr = edgeGeometry.getAttribute("position");
    if (edgePosAttr) {
      let ptr = 0;
      const visibleSet = new Set(arr.map((x) => x.id));
      for (const e of state.edges.values()) {
        if (!visibleSet.has(e.source) || !visibleSet.has(e.target)) continue;
        const s = state.nodes.get(e.source);
        const t = state.nodes.get(e.target);
        if (!s || !t || ptr + 5 >= edgePosAttr.array.length) continue;
        edgePosAttr.array[ptr++] = s.pos.x;
        edgePosAttr.array[ptr++] = s.pos.y;
        edgePosAttr.array[ptr++] = s.pos.z;
        edgePosAttr.array[ptr++] = t.pos.x;
        edgePosAttr.array[ptr++] = t.pos.y;
        edgePosAttr.array[ptr++] = t.pos.z;
      }
      edgePosAttr.needsUpdate = true;
    }
  }

  function applyDelta(delta) {
    if (!delta) return;
    (delta.removed_node_ids || []).forEach((id) => state.nodes.delete(id));
    (delta.removed_edge_ids || []).forEach((id) => state.edges.delete(id));

    const anchorMap = new Map();
    for (const e of (delta.added_edges || [])) {
      const existingSrc = state.nodes.get(e.source);
      const existingDst = state.nodes.get(e.target);
      if (existingSrc && !state.nodes.has(e.target)) {
        anchorMap.set(e.target, existingSrc.pos);
      } else if (existingDst && !state.nodes.has(e.source)) {
        anchorMap.set(e.source, existingDst.pos);
      }
    }

    (delta.added_nodes || []).forEach((n) => upsertNode(n, anchorMap));
    (delta.updated_nodes || []).forEach((n) => upsertNode(n, null));
    (delta.added_edges || []).forEach((e) => state.edges.set(edgeKey(e), e));
    (delta.updated_edges || []).forEach((e) => state.edges.set(edgeKey(e), e));

    rebuildTopologyAndVisuals();
    refreshHoveredHudIfOpen();
  }

  function updateTelemetryUI(tel) {
    if (!tel) return;
    if (tel.latency_ms !== undefined) {
      document.getElementById("stat-latency").textContent = `${tel.latency_ms} ms`;
    }
    if (tel.estimated_fpr !== undefined) {
      document.getElementById("stat-fpr").textContent = `${(tel.estimated_fpr * 100).toFixed(2)}%`;
    }
    if (tel.threshold !== undefined) {
      document.getElementById("threshold-val").textContent = Number(tel.threshold).toFixed(4);
    }
    if (tel.alpha !== undefined) {
      document.getElementById("alpha-val").textContent = Number(tel.alpha).toFixed(3);
    }
  }

  function pushAlertCard(alertObj, ts) {
    if (!alertObj || !alertObj.has_attack) return;
    state.subgraphNodeSet = new Set(alertObj.subgraph_node_ids || []);
    const feed = document.getElementById("alert-timeline");
    const empty = feed.querySelector(".empty-msg");
    if (empty) empty.remove();

    const card = document.createElement("div");
    card.className = "alert-card";
    const dtStr = new Date((ts || Date.now() / 1000) * 1000).toLocaleTimeString();
    card.innerHTML = `
      <strong>[${dtStr}] ${alertObj.dominant_attack_class}</strong>
      <div>2-Hop Subgraph: ${(alertObj.subgraph_node_ids || []).length} nodes (${alertObj.seed_count} seeds)</div>
    `;
    feed.prepend(card);
    while (feed.children.length > 18) {
      feed.removeChild(feed.lastChild);
    }
  }

  const logStreamEl = document.getElementById("live-log-stream");
  const MAX_VISIBLE_LOGS = 15;
  const LOG_TTL_MS = 9000;

  function appendLiveLogItem(item) {
    if (!state.showLogs || !logStreamEl) return;
    const row = document.createElement("div");
    row.className = "log-entry";

    const hostSpan = document.createElement("span");
    hostSpan.className = "log-host";
    hostSpan.textContent = `[${item.host || "vm"}]`;

    const textNode = document.createTextNode(String(item.line || ""));
    row.appendChild(hostSpan);
    row.appendChild(textNode);
    logStreamEl.appendChild(row);

    while (logStreamEl.children.length > MAX_VISIBLE_LOGS) {
      logStreamEl.removeChild(logStreamEl.firstChild);
    }

    setTimeout(() => {
      row.classList.add("expiring");
      setTimeout(() => {
        if (row.parentNode === logStreamEl) {
          logStreamEl.removeChild(row);
        }
      }, 780);
    }, LOG_TTL_MS);
  }

  const socket = io();
  socket.on("graph_init", (payload) => {
    state.nodes.clear();
    state.edges.clear();
    (payload.nodes || []).forEach((n) => upsertNode(n, null));
    (payload.edges || []).forEach((e) => state.edges.set(edgeKey(e), e));
    updateTelemetryUI(payload.telemetry);
    if (payload.attack_alert && payload.attack_alert.has_attack) {
      pushAlertCard(payload.attack_alert, payload.timestamp);
    }
    rebuildTopologyAndVisuals();
  });

  socket.on("graph_delta", (payload) => {
    updateTelemetryUI(payload.telemetry);
    applyDelta(payload.delta);
  });

  socket.on("attack_alert", (payload) => {
    pushAlertCard(payload.alert, payload.timestamp);
    if (state.isolateSubgraph) rebuildTopologyAndVisuals();
  });

  socket.on("audit_log_batch", (payload) => {
    if (!state.showLogs || !payload || !Array.isArray(payload.logs)) return;
    const slice = payload.logs.slice(-10);
    slice.forEach((item, idx) => {
      if (slice.length > 3) {
        setTimeout(() => appendLiveLogItem(item), idx * 55);
      } else {
        appendLiveLogItem(item);
      }
    });
  });

  socket.on("threshold_updated", (res) => {
    document.getElementById("stat-reinf-ms").textContent = `${res.latency_ms} ms`;
    updateTelemetryUI(res);
    if (res.attack_alert) {
      state.subgraphNodeSet = new Set(res.attack_alert.subgraph_node_ids || []);
    }
    applyDelta(res.graph_delta);
  });

  const alphaSlider = document.getElementById("alpha-slider");
  alphaSlider.addEventListener("input", (ev) => {
    const alpha = parseFloat(ev.target.value);
    document.getElementById("alpha-val").textContent = alpha.toFixed(3);
    socket.emit("update_threshold", { alpha });
  });

  const btnLogs = document.getElementById("btn-toggle-logs");
  btnLogs.addEventListener("click", () => {
    state.showLogs = !state.showLogs;
    btnLogs.classList.toggle("active", state.showLogs);
    btnLogs.textContent = `Live Audit Log: ${state.showLogs ? "ON" : "OFF"}`;
    logStreamEl.classList.toggle("hidden", !state.showLogs);
    if (!state.showLogs) {
      logStreamEl.innerHTML = "";
    }
  });

  const btnSubgraph = document.getElementById("btn-subgraph-only");
  btnSubgraph.addEventListener("click", () => {
    state.isolateSubgraph = !state.isolateSubgraph;
    btnSubgraph.classList.toggle("active", state.isolateSubgraph);
    btnSubgraph.textContent = `2-Hop Attack Chain Only: ${state.isolateSubgraph ? "ON" : "OFF"}`;
    rebuildTopologyAndVisuals();
  });

  document.getElementById("btn-sim-attack").addEventListener("click", () => {
    fetch("/api/simulate/live_attack", { method: "POST" });
  });

  let isDragging = false;
  let prevMouse = { x: 0, y: 0 };
  let lastPointer = { x: -999, y: -999 };
  const hud = document.getElementById("inspector-hud");
  const hudIcon = document.getElementById("hud-single-icon");
  const projVec = new THREE.Vector3();
  const viewVec = new THREE.Vector3();

  function populateHud(n, clientX, clientY) {
    hud.classList.remove("hidden");
    const targetSrc = ICON_MAP[n.type] || ICON_MAP.process;
    if (hudIcon.getAttribute("src") !== targetSrc) {
      hudIcon.setAttribute("src", targetSrc);
    }
    document.getElementById("hud-type-badge").textContent = String(n.type || "process").toUpperCase();
    document.getElementById("hud-node-id").textContent = n.id;
    document.getElementById("hud-label").textContent = n.label;
    document.getElementById("hud-score").textContent = Number(n.anomaly_score).toFixed(4);
    document.getElementById("hud-pval").textContent = n.p_value !== undefined ? Number(n.p_value).toFixed(4) : "-";
    document.getElementById("hud-class").textContent = n.attack_class;

    if (clientX !== undefined && clientY !== undefined) {
      const hudW = 364;
      const hudH = 190;
      const left = Math.min(window.innerWidth - hudW - 18, Math.max(18, clientX + 18));
      const top = Math.min(window.innerHeight - hudH - 18, Math.max(76, clientY - 24));
      hud.style.left = `${left}px`;
      hud.style.top = `${top}px`;
    }
  }

  function refreshHoveredHudIfOpen() {
    if (!state.hoveredId) return;
    const n = state.nodes.get(state.hoveredId);
    if (n) {
      populateHud(n);
    } else {
      state.hoveredId = null;
      hud.classList.add("hidden");
    }
  }

  function pickHoveredNodeScreenSpace(clientX, clientY) {
    const arr = state.orderedNodes;
    if (arr.length === 0) return null;

    graphGroup.updateMatrixWorld();
    camera.updateMatrixWorld();

    const halfW = window.innerWidth * 0.5;
    const halfH = window.innerHeight * 0.5;
    let bestNode = null;
    let bestDistSq = Infinity;

    for (let i = 0; i < arr.length; i++) {
      const n = arr[i];
      viewVec.copy(n.pos).applyMatrix4(graphGroup.matrixWorld);
      projVec.copy(viewVec).project(camera);
      if (projVec.z < -1.0 || projVec.z > 1.0) continue;

      const sx = (projVec.x + 1.0) * halfW;
      const sy = (-projVec.y + 1.0) * halfH;

      const camDist = Math.max(45.0, -viewVec.copy(n.pos).applyMatrix4(graphGroup.matrixWorld).applyMatrix4(camera.matrixWorldInverse).z);
      const basePt = (n.is_malicious ? 22.0 : 11.0) * (320.0 / camDist);
      const hitRadius = Math.max(13.0, basePt * 0.65);

      const dx = sx - clientX;
      const dy = sy - clientY;
      const dSq = dx * dx + dy * dy;
      if (dSq <= hitRadius * hitRadius && dSq < bestDistSq) {
        bestDistSq = dSq;
        bestNode = n;
      }
    }
    return bestNode;
  }

  window.addEventListener("mousedown", (e) => {
    if (e.target === renderer.domElement) {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    }
  });
  window.addEventListener("mouseup", () => { isDragging = false; });
  window.addEventListener("wheel", (e) => {
    camera.position.z = Math.min(1100, Math.max(120, camera.position.z + e.deltaY * 0.35));
  });

  window.addEventListener("mousemove", (e) => {
    lastPointer = { x: e.clientX, y: e.clientY };
    if (isDragging) {
      graphGroup.rotation.y += (e.clientX - prevMouse.x) * 0.005;
      graphGroup.rotation.x += (e.clientY - prevMouse.y) * 0.005;
      prevMouse = { x: e.clientX, y: e.clientY };
      return;
    }

    const hit = pickHoveredNodeScreenSpace(e.clientX, e.clientY);
    if (hit) {
      state.hoveredId = hit.id;
      populateHud(hit, e.clientX, e.clientY);
    } else {
      state.hoveredId = null;
      hud.classList.add("hidden");
    }
  });

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  let frames = 0;
  let lastFpsCheck = performance.now();

  function animate(nowMs) {
    requestAnimationFrame(animate);
    const t = nowMs * 0.001;
    nodeUniforms.uTime.value = t;
    if (!isDragging) {
      graphGroup.rotation.y += 0.0009;
    }

    stepContinuousPhysics();

    const activeCount = Math.min(MAX_PARTICLES, activeAttackEdges.length * 4);
    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (i < activeCount) {
        const edgeObj = activeAttackEdges[i % activeAttackEdges.length];
        const phase = ((i / 4) * 0.25 + t * 0.72) % 1.0;
        const sPos = edgeObj.sNode.pos;
        const tPos = edgeObj.tNode.pos;
        particlePositions[i * 3] = THREE.MathUtils.lerp(sPos.x, tPos.x, phase);
        particlePositions[i * 3 + 1] = THREE.MathUtils.lerp(sPos.y, tPos.y, phase);
        particlePositions[i * 3 + 2] = THREE.MathUtils.lerp(sPos.z, tPos.z, phase);
        particleColors[i * 3] = edgeObj.color.r;
        particleColors[i * 3 + 1] = edgeObj.color.g;
        particleColors[i * 3 + 2] = edgeObj.color.b;
      } else {
        particlePositions[i * 3 + 1] = -99999.0;
      }
    }
    particleGeometry.attributes.position.needsUpdate = true;
    particleGeometry.attributes.color.needsUpdate = true;

    renderer.render(scene, camera);

    frames++;
    if (nowMs - lastFpsCheck >= 1000) {
      document.getElementById("stat-fps").textContent = String(frames);
      frames = 0;
      lastFpsCheck = nowMs;
    }
  }

  requestAnimationFrame(animate);
})();
