# Microcontext end-to-end marketing video

Issue: [#8782](https://github.com/anthony-chaudhary/fak/issues/8782)

This project is the capture-ready master for the requested story:

1. a 12-second best-hits montage;
2. install and no-key preflight;
3. bounded microcontext selfcheck;
4. exact Qwen3.8-27B result with its real operating envelope;
5. a deliberately unforgeable MacBook/fak-native capture slot;
6. the real existing fak interface; and
7. an end-to-end value recap and CTA.

`claims.json` is the source boundary. Every number and status shown on screen
maps to a committed artifact. The preproduction render is useful now, but it is
not the final requested master because the Mac-native capture is still missing.

## Verify and render

```powershell
go run ./tools/videogen -config tools/videogen/projects/microcontext-e2e/render.json -verify
go run ./tools/videogen -config tools/videogen/projects/microcontext-e2e/render.json -all `
  -ffmpeg "C:\Program Files\AMD\AI_Bundle\Amuse\ffmpeg.exe" `
  -timeline tools/videogen/projects/microcontext-e2e/out/timeline.json `
  -frames tools/videogen/projects/microcontext-e2e/out/frames
```

Before publication, follow `CAPTURE-MACBOOK.md`, replace the placeholder chapter,
change the output names to remove `preproduction`, update `claims.json`, rerender,
read chapters and duration back with ffprobe, and visually review the poster plus
one frame from every chapter.

## Twenty-second best-hits cutdown

`cutdown.json` renders the social opener independently from the master. It keeps
the exact-target operating envelope on screen and sends viewers to the full
install/run/inspect walkthrough.

```powershell
go run ./tools/videogen -trailer -config tools/videogen/projects/microcontext-e2e/cutdown.json -verify
go run ./tools/videogen -trailer -config tools/videogen/projects/microcontext-e2e/cutdown.json -all -ffmpeg $ffmpeg
```

The checked outputs are `visuals/fak-microcontext-best-hits.{mp4,gif}`, its
poster, and the machine-readable readability audit.
