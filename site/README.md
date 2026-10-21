# GLM 5.3 Flash-sauhard site

The one-page write-up of this project, live at https://glm-flash-sauhard.vercel.app.

- `build/` makes the page. `assemble.mjs` writes `public/index.html`; `fig-scale.mjs`, `fig-case.mjs` and
  `fig-chest.mjs` are the three interactive figures, drawn with the Anatomy isometric kit.
- `build/experts.json` holds per-layer routing data for Fig 2, exported from calibration by `export_experts.py`.
- `vendor/anatomy/` is the part of [Anatomy](https://github.com/wheresryan22/anatomy) the build uses (MIT, commit in `COMMIT`).

```sh
node site/build/assemble.mjs                 # rebuild public/index.html
node site/build/test-page.mjs fig-case       # one figure alone, for checking (writes build/out-*.html)
cd site/public && vercel --prod              # deploy (project is linked in public/.vercel, not committed)
```
