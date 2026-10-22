// Writes build/out-og.html, a 1200x630 social preview card; screenshot it to public/og.png (see README).
import { writeFileSync } from "node:fs";
import { ISO_CSS } from "../vendor/anatomy/kit/iso-kit.mjs";
import { kitScript } from "../vendor/anatomy/scripts/inline-kit.mjs";
import * as scale from "./fig-scale.mjs";

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>${ISO_CSS}
*{box-sizing:border-box} body{margin:0;background:#0b0b0b}
.card{width:1200px;height:630px;display:grid;grid-template-columns:600px 600px;background:radial-gradient(120% 120% at 0% 0%,#151b2e 0%,#0b0b0b 55%);color:#ececec;font-family:"Familjen Grotesk",sans-serif;overflow:hidden;position:relative}
.text{padding:64px 0 56px 64px;display:flex;flex-direction:column;gap:18px}
.eyebrow{font:500 15px "JetBrains Mono",monospace;letter-spacing:.14em;text-transform:uppercase;color:#7b9dff}
h1{margin:0;font-size:68px;line-height:.98;letter-spacing:-.04em;font-weight:700}
p{margin:0;font-size:25px;line-height:1.35;color:#b4b4b8;max-width:500px}
.stats{display:flex;gap:30px;margin-top:auto}
.stats b{display:block;font-size:40px;font-weight:600;letter-spacing:-.03em}
.stats span{font:14px "JetBrains Mono",monospace;color:#7a7a80}
.fig{position:relative;padding:20px 30px 0 0}
.fig .iso-plate-corners,.fig .fmt-row,.fig figcaption{display:none}
.fig .iso-plate{background:transparent;box-shadow:none;border:0}
.url{position:absolute;right:40px;bottom:30px;font:15px "JetBrains Mono",monospace;color:#6e6e74}
</style></head><body>
<div class="card" id="og">
  <div class="text">
    <span class="eyebrow">MacBook Pro M5 Max · 128 GB</span>
    <h1>GLM 5.3<br>Flash-sauhard</h1>
    <p>A 321B model rebuilt to run offline in 71.6 GB, with room left to work.</p>
    <div class="stats">
      <div><b>71.6 GB</b><span>resident</span></div>
      <div><b>92.4%</b><span>matches original</span></div>
      <div><b>30</b><span>tok/s</span></div>
    </div>
  </div>
  <div class="fig iso" data-theme="dark" id="fig-scale">${scale.html}</div>
  <span class="url">glm-flash-sauhard.vercel.app</span>
</div>
<script type="module">${kitScript().replace(/<\/(script)/gi, "<\\/$1")}
matchMedia = () => ({ matches: true, addEventListener() {} });
${scale.live}</script>
</body></html>`;
writeFileSync(new URL("./out-og.html", import.meta.url), html);
console.log("wrote out-og.html");
