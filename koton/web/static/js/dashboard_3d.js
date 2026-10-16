// Window 3: 3D Force-Directed Causal Graph SOC Dashboard Logic
let Graph = null;
let autoRotate = false;

document.addEventListener("DOMContentLoaded", () => {
  const elem = document.getElementById("3d-graph");

  // Initialize 3D Force-Directed Graph with stabilized physics
  Graph = ForceGraph3D()(elem)
    .backgroundColor("#05080f")
    .nodeId("id")
    .d3AlphaDecay(0.04)         // Fast physics convergence
    .d3VelocityDecay(0.4)       // Dampen node velocity to prevent oscillation
    .warmupTicks(30)            // Pre-calculate positions on initial load
    .cooldownTicks(80)          // Settle and freeze simulation after layout
    .nodeLabel((n) => `[${n.type.toUpperCase()}] ${n.name}\nAnomaly Score: ${n.score}`)
    .nodeThreeObject((node) => {
      const isAnomaly = node.is_anomaly;
      let geometry;
      let color;
      let emissive;

      if (isAnomaly) {
        color = 0xef4444; // Bright Glowing Red
        emissive = 0xaa0000;
      } else {
        if (node.type === "process") {
          color = 0x00e5ff;      // Cyan Spheres 🔵
          emissive = 0x004455;
        } else if (node.type === "file") {
          color = 0x10b981;      // Green Cubes 🟢
          emissive = 0x004422;
        } else if (node.type === "socket") {
          color = 0xf59e0b;      // Orange Octahedrons 🟠
          emissive = 0x442200;
        } else {
          color = 0x94a3b8;
          emissive = 0x111111;
        }
      }

      // 3D Geometry per Entity Type
      const size = isAnomaly ? 6.5 : 4.2;
      if (node.type === "file") {
        geometry = new THREE.BoxGeometry(size * 1.4, size * 1.4, size * 1.4);
      } else if (node.type === "socket") {
        geometry = new THREE.OctahedronGeometry(size * 1.35);
      } else {
        geometry = new THREE.SphereGeometry(size, 20, 20);
      }

      const material = new THREE.MeshStandardMaterial({
        color: color,
        emissive: emissive,
        roughness: 0.3,
        metalness: 0.2,
        transparent: true,
        opacity: isAnomaly ? 0.98 : 0.88,
      });

      const mesh = new THREE.Mesh(geometry, material);

      // Glowing Pulse Halo for Malicious Nodes
      if (isAnomaly) {
        const glowGeo = new THREE.SphereGeometry(size * 1.6, 14, 14);
        const glowMat = new THREE.MeshBasicMaterial({
          color: 0xff1111,
          wireframe: true,
          transparent: true,
          opacity: 0.55,
        });
        mesh.add(new THREE.Mesh(glowGeo, glowMat));
      }

      return mesh;
    })
    // Links styling & animated causal attention particle flow
    .linkWidth((l) => (l.is_kill_chain ? 3.0 : 0.8))
    .linkColor((l) => (l.is_kill_chain ? "#ef4444" : "rgba(30, 58, 95, 0.45)"))
    .linkDirectionalParticles((l) => (l.is_kill_chain ? 6 : 0))
    .linkDirectionalParticleWidth((l) => (l.is_kill_chain ? 4.0 : 0))
    .linkDirectionalParticleSpeed((l) => (l.is_kill_chain ? 0.024 : 0))
    .linkDirectionalParticleColor((l) => "#ff2222")
    .linkLabel((l) => `${l.relation} ${l.is_kill_chain ? '[ATTACK KILL-CHAIN]' : ''}`)
    .onNodeClick((node) => {
      showNodeInspector(node);
      // Smoothly zoom in on the clicked node
      const distance = 45;
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
      Graph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        1500
      );
    });

  // Lighting & Camera
  Graph.cameraPosition({ x: 0, y: 0, z: 240 });

  // Connect Server-Sent Events (SSE)
  connectSSE();

  // Control Event Listeners
  setupControls();
});

function connectSSE() {
  const evtSource = new EventSource("/api/stream");

  evtSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.error) return;
      updateDashboard(data);
    } catch (err) {
      console.error("[!] SSE parse error:", err);
    }
  };

  evtSource.onerror = () => {
    console.warn("[!] SSE connection dropped. Reconnecting...");
  };
}

function updateDashboard(data) {
  // Update Top Stats
  const stats = data.stats || {};
  const nodeCountElem = document.getElementById("nodeCount");
  const anomalyCountElem = document.getElementById("anomalyCount");
  if (nodeCountElem) nodeCountElem.innerText = stats.total_nodes || (data.nodes ? data.nodes.length : 0);
  if (anomalyCountElem) anomalyCountElem.innerText = stats.anomaly_count || 0;

  const systemPulse = document.getElementById("systemPulse");
  const systemStatusText = document.getElementById("systemStatusText");
  const threatBadge = document.getElementById("threatBadge");

  if (systemPulse && systemStatusText && threatBadge) {
    if (stats.anomaly_count > 0) {
      systemPulse.className = "pulse-badge danger";
      systemStatusText.innerText = "THREAT DETECTED (KILL CHAIN ACTIVE)";
      systemStatusText.style.color = "#f87171";
      threatBadge.className = "metric-badge threat-active";
    } else {
      systemPulse.className = "pulse-badge";
      systemStatusText.innerText = "NORMAL / MONITORING";
      systemStatusText.style.color = "#e2e8f0";
      threatBadge.className = "metric-badge";
    }
  }

  // Update Incident Feed Drawer
  updateAlertDrawer(data.alerts || []);

  // Update Graph Data smoothly with in-place diffing (no shaking)
  if (data.nodes && data.links) {
    syncGraphData(data.nodes, data.links);
  }
}

/**
 * Synchronizes new graph state into the existing 3d-force-graph simulation.
 * Preserves existing node coordinates (x, y, z) and velocities to completely eliminate shaking.
 */
function syncGraphData(newNodes, newLinks) {
  const currentData = Graph.graphData();
  const currentNodes = currentData.nodes || [];
  const currentLinks = currentData.links || [];

  // Map existing nodes by ID
  const existingNodeMap = new Map();
  currentNodes.forEach((n) => existingNodeMap.set(n.id, n));

  const newNodeIdSet = new Set(newNodes.map((n) => n.id));
  let nodesChanged = false;

  // 1. Filter out removed nodes
  const updatedNodes = [];
  for (const curr of currentNodes) {
    if (newNodeIdSet.has(curr.id)) {
      updatedNodes.push(curr);
    } else {
      nodesChanged = true;
    }
  }

  // 2. Update existing nodes in-place or append new nodes
  let hasVisualUpdates = false;
  for (const newNode of newNodes) {
    if (existingNodeMap.has(newNode.id)) {
      const existing = existingNodeMap.get(newNode.id);
      // Check if anomaly status or score changed
      if (
        existing.is_anomaly !== newNode.is_anomaly ||
        existing.score !== newNode.score
      ) {
        hasVisualUpdates = true;
      }
      // Update properties in-place (PRESERVES x, y, z, vx, vy, vz!)
      existing.name = newNode.name;
      existing.score = newNode.score;
      existing.is_anomaly = newNode.is_anomaly;
      existing.details = newNode.details;
    } else {
      // Genuinely new node
      nodesChanged = true;
      // Position new node with slight jitter near center
      newNode.x = (Math.random() - 0.5) * 40;
      newNode.y = (Math.random() - 0.5) * 40;
      newNode.z = (Math.random() - 0.5) * 40;
      updatedNodes.push(newNode);
      existingNodeMap.set(newNode.id, newNode);
    }
  }

  // 3. Compare and update links
  const getLinkId = (l) => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return `${s}->${t}:${l.relation || ""}`;
  };

  const existingLinkMap = new Map();
  currentLinks.forEach((l) => existingLinkMap.set(getLinkId(l), l));

  let linksChanged = false;
  if (currentLinks.length !== newLinks.length) {
    linksChanged = true;
  }

  const updatedLinks = [];
  for (const newLink of newLinks) {
    const lid = getLinkId(newLink);
    if (existingLinkMap.has(lid)) {
      const existing = existingLinkMap.get(lid);
      if (existing.is_kill_chain !== newLink.is_kill_chain) {
        hasVisualUpdates = true;
      }
      existing.is_kill_chain = newLink.is_kill_chain;
      existing.attention_weight = newLink.attention_weight;
      existing.relation = newLink.relation;
      updatedLinks.push(existing);
    } else {
      linksChanged = true;
      updatedLinks.push(newLink);
    }
  }

  // Initial load or topological change
  if (currentNodes.length === 0 || nodesChanged || linksChanged) {
    Graph.graphData({
      nodes: updatedNodes,
      links: updatedLinks,
    });
    // For subsequent node additions, gently reheat without jarring movement
    if (currentNodes.length > 0) {
      Graph.d3AlphaTarget(0.06);
      setTimeout(() => Graph.d3AlphaTarget(0), 400);
    }
  } else if (hasVisualUpdates) {
    // Only anomaly badges or scores changed: refresh Three.js visual objects without moving nodes!
    Graph.refresh();
  }
}

function updateAlertDrawer(alerts) {
  const listElem = document.getElementById("alertList");
  const counterElem = document.getElementById("alertCounter");

  if (!listElem || !counterElem) return;

  counterElem.innerText = `${alerts.length} Alerts`;

  if (alerts.length === 0) {
    listElem.innerHTML = `
      <div style="color:#64748b; font-size:0.8rem; text-align:center; padding:30px 10px;">
        Monitoring Linux audit stream...<br>No anomalous kill chains detected.
      </div>
    `;
    return;
  }

  listElem.innerHTML = "";
  alerts.forEach((alert) => {
    const card = document.createElement("div");
    card.className = `alert-card ${alert.severity === "CRITICAL" ? "critical" : ""}`;
    card.style.cursor = "pointer";
    card.innerHTML = `
      <div class="alert-title">
        <span>${escapeHtml(alert.title)}</span>
        <span class="tag-technique">${escapeHtml(alert.technique)}</span>
      </div>
      <div class="alert-meta">
        <div><strong>Entity:</strong> ${escapeHtml(alert.entity)}</div>
        <div><strong>Anomaly Score:</strong> <span style="color:#f87171; font-weight:bold;">${alert.score}</span> &bull; ${alert.timestamp}</div>
        ${alert.cmd ? `<div style="color:#cbd5e1; margin-top:2px;"><code>${escapeHtml(alert.cmd)}</code></div>` : ""}
      </div>
    `;
    // Click on alert zooms to node in 3D graph
    card.addEventListener("click", () => {
      const graphData = Graph.graphData();
      const targetNode = graphData.nodes.find(n => n.id === alert.entity);
      if (targetNode) {
        showNodeInspector(targetNode);
        const distance = 40;
        const distRatio = 1 + distance / Math.hypot(targetNode.x, targetNode.y, targetNode.z);
        Graph.cameraPosition(
          { x: targetNode.x * distRatio, y: targetNode.y * distRatio, z: targetNode.z * distRatio },
          targetNode,
          1200
        );
      }
    });
    listElem.appendChild(card);
  });
}

function showNodeInspector(node) {
  const drawer = document.getElementById("nodeInspector");
  const title = document.getElementById("inspTitle");
  const body = document.getElementById("inspBody");

  if (!drawer || !title || !body) return;

  title.innerText = `[${node.type.toUpperCase()}] ${node.name}`;
  title.style.color = node.is_anomaly ? "#f87171" : "#00e5ff";

  let d = node.details || {};
  let rows = [
    `<strong>Entity ID:</strong> ${escapeHtml(node.id)}`,
    `<strong>Entity Type:</strong> ${node.type.toUpperCase()}`,
    `<strong>GNN Anomaly Score:</strong> <span style="color:${node.is_anomaly ? '#ef4444' : '#10b981'}; font-weight:bold; font-size:0.95rem;">${node.score}</span>`,
    `<strong>Verdict:</strong> ${node.is_anomaly ? '<span style=\"color:#ef4444; font-weight:bold;\">MALICIOUS / APT THREAT</span>' : '<span style=\"color:#10b981;\">BENIGN</span>'}`
  ];

  if (node.type === "process") {
    rows.push(`<strong>PID:</strong> ${d.pid || 'N/A'}`);
    rows.push(`<strong>UID:</strong> ${d.uid !== undefined ? d.uid : 1000} ${d.uid === 0 ? '(ROOT)' : ''}`);
    rows.push(`<strong>Binary Path:</strong> ${escapeHtml(d.path || 'N/A')}`);
    rows.push(`<strong>Command:</strong> <code>${escapeHtml(d.cmdline || 'N/A')}</code>`);
    if (node.is_anomaly) {
      const tech = "sh" in node.name ? "T1059.004 (Unix Shell)" : ("curl" in node.name ? "T1105 (Ingress Tool Transfer)" : "T1068 (Privilege Escalation)");
      rows.push(`<strong>MITRE ATT&CK:</strong> <span class=\"tag-technique\">${tech}</span>`);
    }
  } else if (node.type === "file") {
    rows.push(`<strong>Filepath:</strong> ${escapeHtml(d.path || 'N/A')}`);
    rows.push(`<strong>Extension:</strong> ${escapeHtml(d.ext || 'N/A')}`);
    if (node.is_anomaly) {
      const tech = d.path && d.path.includes("passwd") ? "T1005 (Data from Local System)" : "T1105 (Ingress Tool Transfer / Dropped Payload)";
      rows.push(`<strong>MITRE ATT&CK:</strong> <span class=\"tag-technique\">${tech}</span>`);
    }
  } else if (node.type === "socket") {
    rows.push(`<strong>Remote IP:</strong> ${escapeHtml(d.ip || 'N/A')}`);
    rows.push(`<strong>Remote Port:</strong> ${d.port || 80}`);
    rows.push(`<strong>Protocol:</strong> ${escapeHtml(d.protocol || 'TCP')}`);
    if (node.is_anomaly) {
      rows.push(`<strong>MITRE ATT&CK:</strong> <span class=\"tag-technique\">T1071.001 (Application Layer Protocol: C2)</span>`);
    }
  }

  body.innerHTML = rows.join("<br>");
  drawer.style.display = "block";
}

function setupControls() {
  const closeBtn = document.getElementById("btnCloseInspector");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      document.getElementById("nodeInspector").style.display = "none";
    });
  }

  // Threshold Slider
  const slider = document.getElementById("thresholdSlider");
  const thVal = document.getElementById("thresholdVal");
  if (slider && thVal) {
    slider.addEventListener("input", async (e) => {
      const val = parseFloat(e.target.value);
      thVal.innerText = val.toFixed(2);
      try {
        await fetch("/api/config/threshold", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold: val })
        });
      } catch (err) {}
    });
  }

  // Reset Graph Button
  const resetBtn = document.getElementById("btnResetGraph");
  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/graph/reset", { method: "POST" });
      } catch (err) {}
    });
  }

  // Auto-Rotate Button
  const rotateBtn = document.getElementById("btnToggleRotate");
  if (rotateBtn) {
    rotateBtn.addEventListener("click", () => {
      autoRotate = !autoRotate;
      rotateBtn.innerText = autoRotate ? "Stop Rotate" : "Toggle Auto-Rotate";
      if (autoRotate) {
        animateRotate();
      }
    });
  }
}

function animateRotate() {
  if (!autoRotate) return;
  const camera = Graph.camera();
  const angle = 0.003;
  const x = camera.position.x;
  const z = camera.position.z;
  camera.position.x = x * Math.cos(angle) + z * Math.sin(angle);
  camera.position.z = z * Math.cos(angle) - x * Math.sin(angle);
  camera.lookAt(Graph.scene().position);
  requestAnimationFrame(animateRotate);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
