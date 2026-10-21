// Fig 1: a bench scale weighing the model in each format against the Mac's 128 GB.
// The stack on the platform is 45 plates, one per decoder layer; its height is to scale with the model's size.
import * as k from "../vendor/anatomy/kit/iso-kit.mjs";

export const FORMATS = [
  { id: "bf16", name: "BF16 original", gb: 598, note: "321B weights at 2 bytes each" },
  { id: "fp8", name: "FP8 download", gb: 305, note: "the checkpoint Z.ai ships" },
  { id: "q4", name: "4-bit, all 288 experts", gb: 168, note: "what a plain quantizer gives you" },
  { id: "prune", name: "153 experts, 4-bit", gb: 92, note: "after pruning, before bit allocation" },
  { id: "ours", name: "GLM 5.3 Flash-sauhard", gb: 71.6, note: "153 experts, 3.48 bits average, LSQ rounding" },
];
const MAC = 128;
const CAP = 84;
const DIAL_MAX = 160;
const STOP = 172; // stop pin, in GB on the dial
const PER_GB = 0.29; // world units of stack height per GB
const LAYERS = 45;

const WIDTH = 600;
const HEIGHT = 470;
const BASE = { x: -136, y: -74, w: 272, d: 148, r: 8 };
const BASE_H = 12;
const FOOT_H = 5;
const PLAT = { x: -116, y: -54, w: 122, d: 108, r: 5 };
const PLAT_Z = 6;
const PLAT_H = 6;
const PLAT_TOP = PLAT_Z + PLAT_H;
const STACK = { x: -100, y: -40, w: 90, d: 80, r: 3 };
const TOWER = { x: 36, y: -34, w: 68, d: 68, r: 6 };
const TOWER_H = 58;
const DIAL = [70, 0];
const DIAL_R = 28;
const DIAL_Z = TOWER_H;
const DIAL_H = 9;
const DIAL_TOP = DIAL_Z + DIAL_H;
const FROM = (-150 * Math.PI) / 180;
const SPAN = (300 * Math.PI) / 180;
const KNOB = [112, 50];
const KNOB_R = 7;
const SINK = 0.025; // platform travel per GB, capped at the stop

const P = k.fitProjection(
  [...k.boxCorners(BASE, -BASE_H - FOOT_H, 0), [STACK.x, STACK.y, PLAT_TOP + FORMATS[0].gb * PER_GB], [STACK.x + STACK.w, STACK.y + STACK.d, PLAT_TOP + FORMATS[0].gb * PER_GB], [DIAL[0] + DIAL_R, DIAL[1] + DIAL_R, DIAL_TOP]],
  WIDTH,
  HEIGHT,
  { pad: 26, azimuth: 38 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const angleOf = (gb) => FROM + (Math.min(gb, STOP) / DIAL_MAX) * SPAN;

function base() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(BASE, 22))) {
    out.push(S(k.cylinder(x, y, 7, -BASE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -BASE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid", inner: L(k.sideSeam(BASE, -5, P, 8), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(BASE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(BASE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // ruler strip along the front edge, one tick per 4 units
  const strip = { x: -112, y: BASE.y + BASE.d - 15, w: 150, d: 7, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const t = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 4, 5, strip.y + 0.8, 0.6, [2, 3.6], P);
  out.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  out.push(D([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));
  // label plate on the right of the front edge
  const label = { x: 50, y: BASE.y + BASE.d - 20, w: 34, d: 12, r: 1.4 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.6), 0.8, P, 4), { tone: "faint" }));
  for (const [i, len] of [[0, 22], [1, 16], [2, 19]]) {
    out.push(L(k.lineOnTop([label.x + 5, label.y + 3.4 + i * 2.6], [label.x + 5 + len, label.y + 3.4 + i * 2.6], 0.8, P), { tone: "lo", free: true }));
  }
  out.push(D([at([label.x + 2.2, label.y + 2.2, 0.8]), at([label.x + label.w - 2.2, label.y + label.d - 2.2, 0.8])], { size: 0.4 }));
  return out.join("");
}

function linkage() {
  // a lever bar from under the platform into the tower, in a guide block
  const out = [];
  out.push(S(k.slabOf({ x: -6, y: -4, w: 44, d: 8, r: 1.5 }, 0, 3, P, 3, 0.4), { tone: "lo" }));
  out.push(S(k.slabOf({ x: 14, y: -3, w: 24, d: 6, r: 1.2 }, 3, 3, P, 3, 0.4), { tone: "mid" }));
  out.push(S(k.slabOf({ x: 20, y: -7, w: 8, d: 14, r: 1.5 }, 0, 8, P, 3, 0.5), { tone: "mid" }));
  out.push(D([at([24, -4.5, 8]), at([24, 4.5, 8])], { size: 0.45 }));
  return out.join("");
}

function posts() {
  const out = [];
  for (const [x, y] of byDepth(k.corners(PLAT, 12))) {
    out.push(S(k.cylinder(x, y, 5.5, 0, 1.4, P, 24), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 3.4, 1.4, PLAT_Z - 1.4, P, 20), { tone: "mid", crease: "none" }));
    out.push(L(k.sideArc(x, y, 3.4, 3.2, P, 12), { tone: "faint" }));
  }
  return out.join("");
}

function platform() {
  const out = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(PLAT, 2), PLAT_Z, 0.1, P)}" style="filter:blur(3px)"/>`];
  out.push(S(k.slabOf(PLAT, PLAT_Z, PLAT_H, P, 6, 1), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLAT, 4), PLAT_TOP, P, 6), { tone: "faint" }));
  out.push(L(k.planOutline(k.insetPlan(STACK, -3), PLAT_TOP, P, 4), { tone: "lo" }));
  const screws = k.corners(PLAT, 7);
  out.push(D(screws.map(([x, y]) => at([x, y, PLAT_TOP])), { size: 0.45 }));
  return out.join("");
}

// the stack is rebuilt live; this is the same function, shipped to the page
function stackSvg(height, P) {
  const t = height / LAYERS;
  const parts = [];
  for (let i = 0; i < LAYERS; i++) {
    const z = PLAT_TOP + i * t;
    const lit = i === LAYERS - 1;
    parts.push(solidSvg(slabOf(STACK, z, Math.max(t, 0.05), P, 3, t > 1.2 ? 0.4 : 0), { tone: "hi", crease: t > 1 ? "lo" : "none", lit }));
  }
  return parts.join("");
}

function tower() {
  const out = [`<path class="iso-halo" d="${k.haloOf(TOWER, 0, TOWER_H, P)}"/>`];
  out.push(S(k.slabOf(TOWER, 0, TOWER_H, P, 6, 1.2), { tone: "mid", inner: L(k.sideSeam(TOWER, 12, P, 6) + k.sideSeam(TOWER, TOWER_H - 8, P, 6), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(TOWER, 3), TOWER_H, P, 6), { tone: "faint" }));
  out.push(D(k.corners(TOWER, 5).map(([x, y]) => at([x, y, TOWER_H])), { size: 0.45 }));
  return out.join("");
}

function vents() {
  // vents as thin closed slots on the +y face, drawn as rounded rectangles in that plane
  const out = [];
  const y0 = TOWER.y + TOWER.d;
  for (let i = 0; i < 5; i++) {
    const z = 18 + i * 5;
    const a = [TOWER.x + 16, y0, z], b = [TOWER.x + 52, y0, z], c = [TOWER.x + 52, y0, z + 2], d = [TOWER.x + 16, y0, z + 2];
    const pts = [a, b, c, d].map(at);
    out.push(`<path class="iso-line" data-tone="lo" d="M${pts.map((p) => p.map((v) => v.toFixed(1)).join(" ")).join("L")}Z"/>`);
  }
  return out.join("");
}

function dial() {
  const out = [];
  const [x, y] = DIAL;
  out.push(S(k.cylinder(x, y, DIAL_R + 3, DIAL_Z, 2, P, 64), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z + 2, DIAL_H - 2, P, 64, 0.8), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 3.2, DIAL_TOP, 90, P, 0.3), { tone: "lo" }));
  out.push(L(k.sideArc(x, y, DIAL_R, DIAL_TOP - 3.2, P, 40), { tone: "faint" }));
  out.push(L(k.ring(x, y, DIAL_R - 2.4, DIAL_TOP, P, 64), { tone: "lo" }));
  const ticks = k.radialTicks(x, y, DIAL_R - 3, 32, 4, DIAL_TOP, [1.4, 3.2], P, FROM, SPAN); // every 5 GB, major every 20
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  // the Mac's 128 GB and the server's 84 GB cap, as long marks
  const mark = (gb, len) => {
    const a = angleOf(gb), r0 = DIAL_R - 3 - len, r1 = DIAL_R - 2.4;
    return k.lineOnTop([x + Math.sin(a) * r0, y - Math.cos(a) * r0], [x + Math.sin(a) * r1, y - Math.cos(a) * r1], DIAL_TOP, P);
  };
  out.push(`<path class="iso-line" data-tone="hi" d="${mark(MAC, 5.4)}" style="stroke-width:1.2"/>`);
  out.push(`<path class="iso-line" data-tone="red" d="${mark(CAP, 5.4)}" style="stroke-width:1.2"/>`);
  // stop pin past the end of the scale
  const sa = angleOf(STOP + 2.5);
  const pin = [x + Math.sin(sa) * (DIAL_R - 6), y - Math.cos(sa) * (DIAL_R - 6)];
  out.push(S(k.cylinder(pin[0], pin[1], 1.1, DIAL_TOP, 2.2, P, 12), { tone: "mid" }));
  out.push(k.faceTextSvg(k.topMatrix([x, y + 11], DIAL_TOP, P, "x"), "GB", { size: 3, tone: "lo", anchor: "middle" }));
  out.push(L(k.ring(x, y, 1.8, DIAL_TOP, P, 16), { tone: "mid" }));
  return out.join("");
}

const needleAt = (angle) => {
  const [x, y] = DIAL;
  return k.lineOnTop([x - Math.sin(angle) * 4, y + Math.cos(angle) * 4], [x + Math.sin(angle) * (DIAL_R - 4.5), y - Math.cos(angle) * (DIAL_R - 4.5)], DIAL_TOP + 0.2, P);
};

function knobBase() {
  const out = [];
  const [x, y] = KNOB;
  out.push(S(k.cylinder(x, y, KNOB_R + 4, 0, 1.2, P, 40), { tone: "lo" }));
  // five detents, one per format, on the collar
  const pts = FORMATS.map((_, i) => { const a = -1.1 + i * 0.55; return at([x + Math.sin(a) * (KNOB_R + 2.4), y - Math.cos(a) * (KNOB_R + 2.4), 1.2]); });
  out.push(D(pts, { size: 0.55, tone: "mid" }));
  out.push(S(k.cylinder(x, y, KNOB_R, 1.2, 5, P, 36), { tone: "mid" }));
  out.push(L(k.knurl(x, y, KNOB_R, 1.2, 6.2, 40, P), { tone: "lo" }));
  return out.join("");
}
const knobPointer = (i) => {
  const [x, y] = KNOB;
  const a = -1.1 + i * 0.55;
  return k.lineOnTop([x, y], [x + Math.sin(a) * (KNOB_R - 1.2), y - Math.cos(a) * (KNOB_R - 1.2)], 6.2, P);
};

const start = FORMATS.length - 1;
const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label: "A bench scale. On its platform stands a stack of 45 plates, one per decoder layer of GLM-5.3-Flash, as tall as the model is large in the chosen format. The dial on the tower reads the load in gigabytes, with a white mark at the Mac's 128 GB and a red mark at the server's 84 GB cap; anything above 172 GB pins the needle against its stop.",
  body: [
    base(),
    linkage(),
    posts(),
    `<g data-part="platform">${platform()}<g data-part="stack">${stackSvg(FORMATS[start].gb * PER_GB, P)}</g></g>`,
    tower(),
    vents(),
    dial(),
    `<path data-part="needle" class="iso-line" data-tone="lit" data-free="start" d="${needleAt(angleOf(FORMATS[start].gb))}" style="stroke-width:1.2"/>`,
    knobBase(),
    `<path data-part="pointer" class="iso-line" data-tone="hi" data-free="end" d="${knobPointer(start)}" style="stroke-width:1"/>`,
    `<path data-part="hub" class="iso-line" data-tone="mid" d="${k.ring(KNOB[0], KNOB[1], 1.4, 6.2, P, 12)}"/>`,
  ].join(""),
});
// the earlier stackSvg call ran with the kit's names; give Node the same globals
function solidSvg(p, s) { return k.solidSvg(p, s); }
function slabOf(...a) { return k.slabOf(...a); }

export const html = k.plateHtml({
  fig: "Fig 1",
  title: "Weighing the model",
  hint: "Turn the knob, or press 1 to 5",
  readout: readoutOf(start),
  keys: [
    { mark: "raised", label: "45 plates, one per layer, to scale with size" },
    { mark: "lit", label: "Needle: gigabytes on the platform" },
    { mark: "red", label: "Server cap, 84 GB" },
  ],
  caption: "The Mac has 128 GB, and the model has to share it with everything else. The full model weighs 598 GB in BF16 and 305 GB as shipped. Pruning experts and spending bits where they count brings GLM 5.3 Flash-sauhard to 71.6 GB.",
  body: `<div class="iso-stage" id="scale-stage" tabindex="0" role="slider" aria-label="Model format: arrow keys or 1 to 5" aria-valuemin="1" aria-valuemax="5" aria-valuenow="${start + 1}">${svg}</div>
<div class="fmt-row" role="group" aria-label="Model format">${FORMATS.map((f, i) => `<button type="button" id="fmt-${f.id}" class="fmt" data-i="${i}" aria-pressed="${i === start}"><span>${f.name}</span><b>${f.gb} GB</b></button>`).join("")}</div>`,
});

function readoutOf(i) {
  const f = FORMATS[i];
  if (f.gb > MAC) return `${f.name} · ${f.gb} GB · ${(f.gb / MAC).toFixed(1)}× the Mac`;
  if (f.gb > CAP) return `${f.name} · ${f.gb} GB · fits, but over the 84 GB cap`;
  return `${f.name} · ${f.gb} GB · ${(MAC - f.gb).toFixed(0)} GB left on the Mac`;
}

export const live = `
{
const root = document.getElementById("fig-scale");
const P = ${JSON.stringify(P)};
const FORMATS = ${JSON.stringify(FORMATS)};
const MAC = ${MAC}, CAP = ${CAP}, DIAL_MAX = ${DIAL_MAX}, STOP = ${STOP}, PER_GB = ${PER_GB}, LAYERS = ${LAYERS}, SINK = ${SINK};
const STACK = ${JSON.stringify(STACK)}, PLAT_TOP = ${PLAT_TOP}, DIAL = ${JSON.stringify(DIAL)}, DIAL_R = ${DIAL_R}, DIAL_TOP = ${DIAL_TOP}, FROM = ${FROM}, SPAN = ${SPAN}, KNOB = ${JSON.stringify(KNOB)}, KNOB_R = ${KNOB_R};
const stage = root.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const plat = svg.querySelector('[data-part="platform"]');
const stack = svg.querySelector('[data-part="stack"]');
const needle = svg.querySelector('[data-part="needle"]');
const pointer = svg.querySelector('[data-part="pointer"]');
const readout = root.querySelector("[data-readout]");
const buttons = [...root.querySelectorAll(".fmt")];
const still = matchMedia("(prefers-reduced-motion: reduce)");
const angleOf = (gb) => FROM + (Math.min(gb, STOP) / DIAL_MAX) * SPAN;
const stackSvg = ${stackSvg.toString()};
const needleAt = ${needleAt.toString().replace(/k\./g, "")};
const knobPointer = ${knobPointer.toString().replace(/k\./g, "")};
const readoutOf = ${readoutOf.toString()};
let index = ${start}, load = FORMATS[index].gb, needleA = angleOf(load), needleV = 0, shown = -1;
let frame = 0, last = 0, clock = 0, touring = !still.matches, visible = false, idle = 0;
function pick(i, byUser) {
  index = i;
  buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  stage.setAttribute("aria-valuenow", String(i + 1));
  pointer.setAttribute("d", knobPointer(i));
  readout.textContent = readoutOf(i);
  stage.setAttribute("aria-valuetext", readoutOf(i));
  if (byUser) { touring = false; clearTimeout(idle); idle = setTimeout(() => { touring = !still.matches; clock = 0; run(); }, 6000); }
  run();
}
function draw() {
  const h = load * PER_GB;
  if (Math.abs(h - shown) > 0.02) { stack.innerHTML = stackSvg(h, P); shown = h; }
  plat.setAttribute("transform", translateAlong(P, [0, 0, -Math.min(load, STOP) * SINK]));
  needle.setAttribute("d", needleAt(needleA));
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (touring) { clock += dt; const next = Math.floor(clock / 2.8) % FORMATS.length; if (next !== index) pick(next, false); }
  const target = FORMATS[index].gb;
  load = still.matches ? target : load + (target - load) * (1 - Math.exp(-dt / 0.35));
  if (Math.abs(target - load) < 0.05) load = target;
  // the needle rides a damped spring toward the load and stops hard on the pin
  const want = angleOf(load);
  for (let s = 0; s < 4; s++) { needleV += ((want - needleA) * 160 - needleV * 15) * dt / 4; needleA += needleV * dt / 4; }
  const stopA = angleOf(STOP);
  if (needleA > stopA) { needleA = stopA; needleV = -needleV * 0.25; }
  if (still.matches) { needleA = want; needleV = 0; }
  draw();
  const settled = !touring && load === target && Math.abs(needleV) < 1e-4 && Math.abs(want - needleA) < 1e-4;
  frame = visible && !settled ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
buttons.forEach((b, i) => b.addEventListener("click", () => pick(i, true)));
stage.addEventListener("keydown", (e) => {
  let to = null;
  if (e.key >= "1" && e.key <= "5") to = Number(e.key) - 1;
  if (e.key === "ArrowRight" || e.key === "ArrowUp") to = Math.min(FORMATS.length - 1, index + 1);
  if (e.key === "ArrowLeft" || e.key === "ArrowDown") to = Math.max(0, index - 1);
  if (to === null) return;
  e.preventDefault(); pick(to, true);
});
stage.addEventListener("click", (e) => { const box = svg.getBoundingClientRect(); if ((e.clientX - box.left) / box.width > 0.78 && (e.clientY - box.top) / box.height > 0.6) pick((index + 1) % FORMATS.length, true); });
pick(index, false);
draw();
}
`;
