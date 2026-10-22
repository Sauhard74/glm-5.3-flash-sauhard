// Records build/out-video.html frame by frame on its virtual clock, then encodes an MP4 with ffmpeg.
//   CHROME_PATH=... node site/build/record.mjs <frames-dir> <out.mp4> [--seconds 59] [--from 0]
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [framesDir, outMp4] = process.argv.slice(2);
const arg = (name, d) => { const i = process.argv.indexOf(name); return i > 0 ? Number(process.argv[i + 1]) : d; };
const SECONDS = arg("--seconds", 59), FROM = arg("--from", 0), FPS = 30;
if (!framesDir || !outMp4 || !process.env.CHROME_PATH) { console.error("usage: CHROME_PATH=... node record.mjs <frames-dir> <out.mp4>"); process.exit(1); }
mkdirSync(framesDir, { recursive: true });

const profile = mkdtempSync(join(tmpdir(), "rec-"));
const chrome = spawn(process.env.CHROME_PATH, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run",
  "--hide-scrollbars", "--allow-file-access-from-files", "--force-color-profile=srgb", "about:blank"]);
const port = await new Promise((done) => chrome.stderr.on("data", (d) => { const m = /127\.0\.0\.1:(\d+)/.exec(d.toString()); if (m) done(m[1]); }));
const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id) { pending.get(d.id)?.(d); pending.delete(d.id); }
  else if (d.method === "Runtime.exceptionThrown") console.error("page:", d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text);
  else if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") console.error("page:", d.params.args.map((a) => a.value ?? a.description).join(" "));
};
await new Promise((r) => (ws.onopen = r));
const send = (method, params = {}) => new Promise((r) => { const c = ++id; pending.set(c, r); ws.send(JSON.stringify({ id: c, method, params })); });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Runtime.enable"); await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: new URL("./out-video.html", import.meta.url).href });
for (let i = 0; i < 100 && !(await evaluate("!!window.__ready")); i++) await new Promise((r) => setTimeout(r, 100));
await evaluate("window.__ready");
await new Promise((r) => setTimeout(r, 800));
await evaluate("__settle()");

const total = Math.round(SECONDS * FPS), start = Math.round(FROM * FPS);
const t0 = Date.now();
for (let f = 0; f < total; f++) {
  await evaluate(`__advance(${1000 / FPS})`);
  if (f < start) continue;
  await evaluate("__settle()");
  const shot = await send("Page.captureScreenshot", { format: "jpeg", quality: 92 });
  writeFileSync(join(framesDir, `${String(f).padStart(5, "0")}.jpg`), Buffer.from(shot.result.data, "base64"));
  if (f % 150 === 0) console.log(`frame ${f}/${total} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
chrome.kill("SIGKILL"); rmSync(profile, { recursive: true, force: true });

const ff = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(FPS), "-start_number", String(start), "-i", join(framesDir, "%05d.jpg"),
  "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", outMp4], { stdio: "inherit" });
console.log(ff.status === 0 ? `wrote ${outMp4}` : "ffmpeg failed");
process.exit(ff.status ?? 1);
