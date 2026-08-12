# Token-savings hero module

**For:** teams paying for long-running, tool-using agent sessions.

**Problem:** an avoidable model round trip repeats resident-prefix work, adds
latency, and enlarges every later context.

**Today:** agents commonly ask the model to rediscover known work while cache,
routing, compaction, and control remain separate concerns.

**Better because:** fak puts one performance checkpoint at the boundary: reuse
stable prefixes, serve repeats locally, route each call, shed stale turns, skip
kernel-known work, and reuse live KV when fak owns inference. The same checkpoint
keeps the existing security value story intact.

This is a standalone 20-second module. The renderer emits its own MP4/GIF/poster
and then splices it **before** the existing `fak-hero-values.mp4`; the existing
video source and assets remain unchanged. The homepage uses the resulting
`fak-homepage-hero.*` assets, so token savings leads and the established
pre-execution policy story follows.

Render and verify:

```powershell
$ffmpeg = "C:\path\to\ffmpeg.exe"
go run ./tools/videogen -trailer -config tools/videogen/projects/token-savings/trailer.json -verify
go run ./tools/videogen -trailer -config tools/videogen/projects/token-savings/trailer.json -all -ffmpeg $ffmpeg
```

Before publishing, inspect the committed `review-contact-sheet.png` at 680 px and
360 px and confirm `readability-audit.json`. The card renderer uses measured glyph
bounds, shrinks long labels to a 24 px inset, and centers both axes. Also inspect
the first frame, each scene midpoint, the module seam at 20 seconds, and the final
CTA. The current checked pass transcribed all six savings cards and found no text
outside its box.
