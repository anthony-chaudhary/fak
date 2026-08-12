# Hero values video

**For:** engineering teams running tool-using agents.

**Problem:** repeated setup, indiscriminate model calls, and ever-growing
conversation history make agents slower and more expensive, while unconstrained
tool calls widen risk.

**Today:** those concerns are commonly handled by separate wrappers, caches,
routers, summarizers, and policy layers.

**Better because:** fak puts one reusable checkpoint before every tool call, so
performance work compounds and a default-deny capability floor shares the same
seam.

The old 58.9-second cut failed its actual job: full diagrams became unreadable
thumbnails, terminal cards competed with each other, and the viewer had to
study several product values before seeing an action. The replacement is a
20-second silent trailer with one conversion story only:

**dangerous tool call → fak checks before execution → nothing ran → try
`fak guard -- claude`.**

`trailer.json` is the editable source. The generic trailer renderer supplies
large-type scene composition and produces the MP4, GIF, poster, phone-scale
contact sheet, and machine-readable readability audit. The checked adversarial
review after the second render could transcribe every key line at 360 px and
returned PASS for the four-beat story; the first render returned FAIL because
the command and supporting text were too small, which drove the larger CTA and
copy reduction now in the manifest.

The depicted tool call and amount are synthetic and intentionally contain no
private infrastructure, customer data, credentials, or benchmark claims. They
show the public pre-execution policy seam, not a customer event.

Render and verify:

```powershell
$ffmpeg = "C:\path\to\ffmpeg.exe"
go run ./tools/videogen -trailer -config tools/videogen/projects/hero-values/trailer.json -verify
go run ./tools/videogen -trailer -config tools/videogen/projects/hero-values/trailer.json -all -ffmpeg $ffmpeg
```

Before publishing, inspect `out/review-contact-sheet.png` at 680 px and 360 px,
confirm `out/readability-audit.json`, then have a fresh reviewer transcribe each
beat and the CTA without reading this README.
