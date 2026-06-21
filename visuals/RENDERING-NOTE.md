# Rendering note — why a standalone mermaid SVG previews as whitespace, and the fix

**Symptom.** Open a rendered `visuals/*.svg` in anything that is *not* a browser (an
IDE image preview, Windows Photos, Slack/email thumbnail, a PDF/Office embed) and you
see **blank / whitespace** (or colored boxes with no text), even though the same
diagram renders fine inline on GitHub.

**Cause (measured, not guessed).** Two independent things:

1. **`htmlLabels: true` (mermaid's default).** Every label is emitted as an SVG
   `<foreignObject>` containing HTML (`<div>`/`<span>`). `<foreignObject>` is only
   rendered by a full **browser** engine. A non-browser SVG renderer drops it — so the
   text vanishes. Evidence: the default-rendered deck SVGs contain **`<text>` = 0** —
   *all* label text is in `<foreignObject>`.
2. **`width="100%"` with no intrinsic `height`.** The root `<svg>` carries a `viewBox`
   but `width="100%"`; some viewers then compute a 0/percentage height and collapse the
   image. (E.g. `06-context-mmu.svg`'s root has `width="100%"`, no height.)

GitHub's *inline* ```` ```mermaid ```` blocks are unaffected — GitHub renders them in a
browser, where `<foreignObject>` works.

**The fix (used for every `recall-NN-*.svg` / `.png` in this dir):**

```bash
# from this dir; PUPPETEER chromium cache must be present
export PUPPETEER_CACHE_DIR="$HOME/.cache/puppeteer"
echo '{"flowchart":{"htmlLabels":false},"securityLevel":"loose"}' > cfg.json

# PNG — chromium RASTERIZES the foreignObject HTML, so text is baked into pixels.
# This is the bullet-proof "never whitespace anywhere" artifact.
node_modules/.bin/mmdc -p .puppeteer.json -i X.mmd -o X.png -b white -s 2

# SVG — htmlLabels:false makes labels real <text>/<tspan> (render in any SVG viewer);
# then pin explicit width/height from the viewBox so the image always has intrinsic size.
node_modules/.bin/mmdc -p .puppeteer.json -c cfg.json -i X.mmd -o X.svg -b white
#   post-process: replace  width="100%"  with  width="<vbW>" height="<vbH>"
```

After the fix the `recall-*` SVGs carry real text (`<tspan>`/`<text>` 10–38 per
diagram) and explicit pixel dimensions. (Residue: short *edge* labels can still emit a
`<foreignObject>` even with `htmlLabels:false`; the PNG has them unconditionally, so
the PNG is the canonical "always-visible" render and the SVG is the scalable one.)

**Bonus finding — render is the only real syntax witness.** Rendering surfaced a parse
error a structural linter and two adversarial review passes all missed: a node id
named **`call`** (`recall-05`) — `call` is a reserved Mermaid keyword (`click call
callback()`), so the diagram failed to parse (and would have failed GitHub's inline
render too). Fixed by renaming the node. Lesson: **always render** the `.mmd` before
trusting it; lint + review do not run the Mermaid parser.

> The pre-existing master-deck SVGs (`00`–`25`) were rendered with the *default*
> (`htmlLabels:true`) settings, so they exhibit the whitespace symptom in non-browser
> viewers. Re-rendering them with the recipe above (or just using their `.png`) fixes
> it; left as-is here to avoid clobbering a peer's artifacts mid-session.
