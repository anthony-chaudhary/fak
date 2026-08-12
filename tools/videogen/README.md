# fak video generator

`videogen` is fak's public, deterministic terminal-video pipeline. It was
adapted from the renderer developed in the sibling tensor-build workspace at
`work@95bbf653` and is kept here because a successful production tool must not
die in a private scratch checkout.

The root facade is standard-library-only. Rasterization lives in a discovered
nested Go module so fak's zero-dependency root module stays unchanged. The
pipeline turns a checked-in terminal transcript, timing stream, and chapter
cards into:

- visual-first scenes composed from the repo's existing produced PNGs;
- an animated GIF;
- a chaptered H.264 MP4;
- a poster PNG, frame playlist, and machine-checkable timeline; and
- verification that pacing, chapter count, return codes, frame continuity,
  and encoded duration satisfy the project's declared contract.

## Start a project

```sh
go run ./tools/videogen -new my-explainer
# edit tools/videogen/projects/my-explainer/render.json and captures
go run ./tools/videogen -config tools/videogen/projects/my-explainer/render.json -verify
go run ./tools/videogen -config tools/videogen/projects/my-explainer/render.json -all
```

`-all` requires `ffmpeg`. When it is not on `PATH`, pass its executable
explicitly without changing machine-wide setup:

```powershell
go run ./tools/videogen -config tools/videogen/projects/my-explainer/render.json `
  -all -ffmpeg C:\path\to\ffmpeg.exe
```

Record a real command stream rather than hand-writing evidence:

```sh
command 2>&1 | go run ./tools/videogen \
  -record-typescript capture.typescript -record-timing capture.timing
```

## Public-safety boundary

Only checked-in, synthetic or already-public command output belongs in this
repository. Do not capture private hostnames, credentials, internal paths,
customer data, private benchmark endpoints, or provider tokens. Redact before
commit and inspect both `capture.typescript` and generated timeline metadata;
video pixels are not a safe redaction layer. Hardware-dependent captures must
run on a sanctioned node, then return only scrubbed public artifacts.

## Design and verification

The renderer is intentionally content-agnostic. `render.json` owns the story,
visual sequence, chapter cards, pacing thresholds, and output paths. Prefer an
existing produced visual over a prose card whenever the idea already has a
diagram, chart, product capture, or proof artifact. Add it as a config-relative
image scene:

```json
{"chapter":"PAY THE PREFIX ONCE", "image":"../../../../visuals/65-pay-the-prefix-once.png", "imageSecs":5.2}
```

The renderer letterboxes without cropping and uses the same scene and dwell in
the GIF and MP4. This makes the repeatable process **produce visual once → cite
it from one or more video manifests → regenerate and verify**, rather than
redrawing the same concept as video-only text. PNG is the deterministic render
input; retain the SVG source beside it when one exists. The terminal engine owns
ANSI parsing, deterministic bitmap rendering, frame pacing, progress bars,
MP4 chapters, and effect read-back. Tests cover the root facade and nested
renderer separately:

```sh
go test ./tools/videogen
go test ./tools/videogen/terminal/...
```

The second command is invoked from the nested module by CI's dependency
quarantine and may also be run directly with `go -C tools/videogen/terminal test ./...`.
