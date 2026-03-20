/**
 * Hyperhedron Zoo — Three.js Poincaré ball visualizer.
 *
 * Geodesic faces
 * --------------
 * In the Poincaré ball model, hyperbolic geodesic planes appear as Euclidean
 * spherical caps (spheres orthogonal to the unit sphere).  To render them
 * correctly we tessellate each face as a flat polygon in the **Klein model**
 * (where geodesic planes are flat Euclidean polygons), then map every point
 * through Klein → Poincaré:
 *
 *   Klein → Poincaré:  P = K / (1 + √(1−|K|²))
 *   Poincaré → Klein:  K = 2P / (1 + |P|²)
 *
 * This naturally produces the correct inward-bowing spherical-cap faces.
 * Edges are treated the same way: interpolated in Klein, giving circular arcs.
 *
 * Möbius involution (hyperbolic translation)
 * ------------------------------------------
 *   φ_a(x) = [ (1−|a|²)x − (1+|x|²−2⟨a,x⟩)a ] / (1−2⟨a,x⟩+|a|²|x|²)
 *
 * Maps a→0.  Shift+drag and double-click apply this to all vertex positions.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Tessellation quality ──────────────────────────────────────────────────
const FACE_STEPS = 8;   // subdivisions per fan-triangle → 8² = 64 subtriangles
const EDGE_STEPS = 32;  // segments per edge arc

// ── Face palette ──────────────────────────────────────────────────────────
const PALETTE = (() => {
  const c = [];
  for (let i = 0; i < 26; i++) c.push(new THREE.Color().setHSL(i / 26, 0.72, 0.52));
  return c;
})();

// ── Three.js globals ──────────────────────────────────────────────────────
let renderer, scene, camera, controls;
let polyGroup  = null;
let faceMeshes = [];
let edgeLine   = null;
let showFaces  = true;
let showEdges  = true;

// ── Hyperbolic state ──────────────────────────────────────────────────────
let origVertices = [];   // original Poincaré coords from JSON
let curVertices  = [];   // current (after Möbius transforms)
let currentData  = null;

// ── Raycaster ─────────────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
const mouse2d   = new THREE.Vector2();

// ── Shift-drag ────────────────────────────────────────────────────────────
let shiftDrag = false;
let shiftPrev = { x: 0, y: 0 };
const HYP_DRAG_SPEED = 0.004;

// ═════════════════════════════════════════════════════════════════════════
// Coordinate transforms
// ═════════════════════════════════════════════════════════════════════════

function poincareToKlein([x, y, z]) {
  const s = 2.0 / (1.0 + x*x + y*y + z*z);
  return [x*s, y*s, z*s];
}

function kleinToPoincare([x, y, z]) {
  const d = 1.0 + Math.sqrt(Math.max(0.0, 1.0 - (x*x + y*y + z*z)));
  return [x/d, y/d, z/d];
}

// ═════════════════════════════════════════════════════════════════════════
// Klein-space tessellation
// ═════════════════════════════════════════════════════════════════════════

/**
 * Subdivide triangle (a, b, c) [Klein coords] into N² small triangles,
 * map each vertex to Poincaré ball, return flat Float32 position array.
 */
function tessellateKleinTriangle(a, b, c, N) {
  const out = [];
  function poinc(u, v) {
    const w = 1 - u - v;
    return kleinToPoincare([a[0]*u+b[0]*v+c[0]*w,
                             a[1]*u+b[1]*v+c[1]*w,
                             a[2]*u+b[2]*v+c[2]*w]);
  }
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N - i; j++) {
      // Upward triangle
      out.push(...poinc( i/N,     j/N    ),
               ...poinc((i+1)/N,  j/N    ),
               ...poinc( i/N,    (j+1)/N ));
      // Downward triangle (only where it fits)
      if (j < N - i - 1) {
        out.push(...poinc((i+1)/N,  j/N    ),
                 ...poinc((i+1)/N, (j+1)/N),
                 ...poinc( i/N,    (j+1)/N));
      }
    }
  }
  return out;
}

// ═════════════════════════════════════════════════════════════════════════
// Scene setup
// ═════════════════════════════════════════════════════════════════════════

function init() {
  const canvas = document.getElementById('canvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0f0f1a);

  scene  = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, 1, 0.01, 20);
  camera.position.set(0, 0, 2.8);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.9);
  d1.position.set(2, 3, 4); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x8899ff, 0.4);
  d2.position.set(-3, -1, -2); scene.add(d2);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance   = 1.1;
  controls.maxDistance   = 8.0;

  // Poincaré ball boundary sphere
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 48, 48),
    new THREE.MeshBasicMaterial({ color: 0x334488, transparent: true,
                                   opacity: 0.06, side: THREE.BackSide })
  ));
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 18, 18),
    new THREE.MeshBasicMaterial({ color: 0x445599, wireframe: true,
                                   transparent: true, opacity: 0.12 })
  ));

  onResize();
  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('dblclick',   onDoubleClick);
  renderer.domElement.addEventListener('mousedown',  onMouseDown);
  renderer.domElement.addEventListener('mousemove',  onMouseMove);
  renderer.domElement.addEventListener('mouseup',    onMouseUp);

  document.getElementById('btn-reset').addEventListener('click', resetOrigin);
  document.getElementById('btn-faces').addEventListener('click', toggleFaces);
  document.getElementById('btn-edges').addEventListener('click', toggleEdges);

  animate();
  window._selectPolyhedron = selectPolyhedron;
}

function onResize() {
  const wrap = document.getElementById('canvas-wrap');
  renderer.setSize(wrap.clientWidth, wrap.clientHeight, false);
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ═════════════════════════════════════════════════════════════════════════
// Select & render
// ═════════════════════════════════════════════════════════════════════════

async function selectPolyhedron(name, itemEl) {
  document.querySelectorAll('.poly-item').forEach(el => el.classList.remove('active'));
  if (itemEl) itemEl.classList.add('active');

  showOverlay('loading', `Computing ${name}…`);
  document.getElementById('toolbar-title').textContent = name;
  document.getElementById('btn-reset').disabled = true;

  try {
    const data = await fetchJSON(`/api/polyhedra/${name}`);
    currentData  = data;
    origVertices = data.vertices.map(v => [...v]);
    curVertices  = data.vertices.map(v => [...v]);
    rebuildScene();
    document.getElementById('toolbar-info').innerHTML =
      `<span class="badge">${data.num_faces} faces</span>
       <span class="badge">${data.num_vertices} vertices</span>
       <span class="badge">${data.num_edges} edges</span>`;
    updateSidebarDot(name, true);
    document.getElementById('btn-reset').disabled = false;
    hideOverlay();
  } catch (e) {
    showOverlay('error', null, String(e));
  }
}

function updateSidebarDot(name, computed) {
  const item = document.querySelector(`.poly-item[data-name="${name}"]`);
  if (item) item.querySelector('.poly-dot').className =
    `poly-dot ${computed ? 'computed' : 'pending'}`;
}

// ═════════════════════════════════════════════════════════════════════════
// Geometry: Klein tessellation → curved Poincaré faces + arc edges
// ═════════════════════════════════════════════════════════════════════════

/** Rebuild the Three.js group from current curVertices. */
function rebuildScene() {
  if (polyGroup) scene.remove(polyGroup);
  polyGroup  = new THREE.Group();
  faceMeshes = [];

  currentData.faces.forEach((vorder, fi) => {
    const mesh = buildFaceMesh(fi, vorder);
    faceMeshes.push(mesh);
    polyGroup.add(mesh);
  });

  edgeLine = buildEdgeLines(currentData.edges);
  polyGroup.add(edgeLine);

  scene.add(polyGroup);
  applyVisibility();
}

function buildFaceMesh(fi, vorder) {
  // Convert Poincaré vertex positions to Klein
  const kv = vorder.map(vi => poincareToKlein(curVertices[vi]));

  // Fan tessellation in Klein space, mapped to Poincaré
  const allPos = [];
  for (let t = 0; t < vorder.length - 2; t++) {
    allPos.push(...tessellateKleinTriangle(kv[0], kv[t+1], kv[t+2], FACE_STEPS));
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(allPos), 3));
  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
    color:      PALETTE[fi % PALETTE.length],
    transparent: true,
    opacity:     0.55,
    side:        THREE.DoubleSide,
    shininess:   70,
    specular:    new THREE.Color(0.25, 0.25, 0.25),
    depthWrite:  false,
  }));
  mesh.userData = { fi, vorder };
  return mesh;
}

function buildEdgeLines(edges) {
  // Interpolate each edge in Klein space → circular arc in Poincaré
  const pos = [];
  for (const [vi, vj] of edges) {
    const ka = poincareToKlein(curVertices[vi]);
    const kb = poincareToKlein(curVertices[vj]);
    let prev = kleinToPoincare(ka);
    for (let s = 1; s <= EDGE_STEPS; s++) {
      const t  = s / EDGE_STEPS;
      const kk = [ka[0]*(1-t)+kb[0]*t, ka[1]*(1-t)+kb[1]*t, ka[2]*(1-t)+kb[2]*t];
      const curr = kleinToPoincare(kk);
      pos.push(...prev, ...curr);
      prev = curr;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  return new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 1.5 }));
}

// ═════════════════════════════════════════════════════════════════════════
// Möbius involution  φ_a(x) = [(1−|a|²)x − (1+|x|²−2⟨a,x⟩)a] / denom
// ═════════════════════════════════════════════════════════════════════════

function mobiusTransform(a) {
  const [ax, ay, az] = a;
  const aSq = ax*ax + ay*ay + az*az;
  curVertices = curVertices.map(([x, y, z]) => {
    const xSq   = x*x + y*y + z*z;
    const xDotA = x*ax + y*ay + z*az;
    const denom  = 1 - 2*xDotA + aSq*xSq;
    const f1 = (1 - aSq) / denom;
    const f2 = (1 + xSq - 2*xDotA) / denom;
    return [f1*x - f2*ax, f1*y - f2*ay, f1*z - f2*az];
  });
  rebuildScene();
}

function translateHyp(dx, dy, dz, t) {
  const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (len < 1e-10) return;
  const r = Math.tanh(t / 2);
  mobiusTransform([r*dx/len, r*dy/len, r*dz/len]);
}

function resetOrigin() {
  if (!currentData) return;
  curVertices = origVertices.map(v => [...v]);
  rebuildScene();
}

// ═════════════════════════════════════════════════════════════════════════
// Mouse events
// ═════════════════════════════════════════════════════════════════════════

function onMouseDown(e) {
  if (e.shiftKey) {
    shiftDrag = true;
    shiftPrev = { x: e.clientX, y: e.clientY };
    controls.enabled = false;
  }
}

function onMouseMove(e) {
  if (!shiftDrag || !currentData) return;
  const dx = e.clientX - shiftPrev.x;
  const dy = e.clientY - shiftPrev.y;
  shiftPrev = { x: e.clientX, y: e.clientY };

  const fwd   = new THREE.Vector3(); camera.getWorldDirection(fwd);
  const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
  const up    = new THREE.Vector3().crossVectors(right, fwd).normalize();
  const dir   = right.clone()
    .multiplyScalar(dx * HYP_DRAG_SPEED)
    .addScaledVector(up, -dy * HYP_DRAG_SPEED);
  const dist  = dir.length();
  if (dist > 1e-8) translateHyp(dir.x, dir.y, dir.z, dist);
}

function onMouseUp() {
  if (shiftDrag) { shiftDrag = false; controls.enabled = true; }
}

function onDoubleClick(e) {
  if (!currentData) return;
  const rect = renderer.domElement.getBoundingClientRect();
  mouse2d.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse2d.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse2d, camera);
  const hits = raycaster.intersectObjects(faceMeshes, false);
  if (!hits.length) return;
  const pt = hits[0].point;
  if (pt.lengthSq() >= 1.0) return;
  mobiusTransform([pt.x, pt.y, pt.z]);
}

// ═════════════════════════════════════════════════════════════════════════
// Visibility
// ═════════════════════════════════════════════════════════════════════════

function applyVisibility() {
  faceMeshes.forEach(m => { m.visible = showFaces; });
  if (edgeLine) edgeLine.visible = showEdges;
}

function toggleFaces() {
  showFaces = !showFaces;
  document.getElementById('btn-faces').style.opacity = showFaces ? 1 : 0.45;
  applyVisibility();
}

function toggleEdges() {
  showEdges = !showEdges;
  document.getElementById('btn-edges').style.opacity = showEdges ? 1 : 0.45;
  applyVisibility();
}

// ═════════════════════════════════════════════════════════════════════════
// Overlay / utility
// ═════════════════════════════════════════════════════════════════════════

function showOverlay(kind, msg, err) {
  document.getElementById('spinner').style.display = kind === 'loading' ? 'block' : 'none';
  document.getElementById('overlay-msg').textContent   = msg || '';
  document.getElementById('overlay-error').textContent = err || '';
  document.getElementById('overlay').classList.remove('hidden');
}

function hideOverlay() {
  document.getElementById('overlay').classList.add('hidden');
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) { const b = await r.text(); throw new Error(`HTTP ${r.status}: ${b}`); }
  return r.json();
}

// ── Start ─────────────────────────────────────────────────────────────────
init();
