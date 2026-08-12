# Token-savings homepage hero

**For:** teams paying for long-running, tool-using agent sessions.

**Problem:** avoidable model round trips repeat prefix work, add latency, and
make every later context larger.

**Today:** that waste is invisible and abstract; a stack of text cards does not
make the compounding cost understandable.

**Better because:** this 20-second visual module turns model work into moving
particles. The first scene contrasts repeated work with a compact reused set;
the second makes fak a hub whose six saving mechanisms share one checkpoint;
the third visibly contracts the stream from avoided turn to smaller context to
longer session; the final scene shows performance and control crossing the same
boundary.

This is the homepage hero. It renders natively at **1920x1080 and 60 fps**. The
older pre-execution-policy video remains intact at `visuals/fak-hero-values.mp4`
and is no longer spliced into the homepage asset.

## Deterministic visual audit

`-verify` now does more than count words. It:

- rejects token visuals below 1920x1080 or 60 fps;
- reserves a 156 px horizontal safe area at 1080p;
- measures rendered glyph widths and fits every centered headline inside it;
- renders three representative samples per scene through the real renderer;
- validates duration, pacing, CTA hold, and text density;
- writes `readability-audit.json` with the checked resolution, safe margin, and
  sample count.

The committed `review-contact-sheet.png` is a 1920x1080 four-frame proof sheet. The audit also records the furthest measured text edge (`maxTextRightPx`); at 1080p it must remain left of the 1764 px safe-area boundary.
It should still receive a human/agent visual read, but clipping and resolution
no longer rely on that reviewer noticing a defect.

```powershell
$ffmpeg = "C:\path\to\ffmpeg.exe"
go run ./tools/videogen -trailer -config tools/videogen/projects/token-savings/trailer.json -verify
go run ./tools/videogen -trailer -config tools/videogen/projects/token-savings/trailer.json -all -ffmpeg $ffmpeg
```
