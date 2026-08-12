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

This cut reuses the checked-in agent-kernel diagrams for the prefix, context MMU, policy path, and independent-proof witness, with the terminal capture reserved for
the live behavioral spine. Those image scenes are declared in `render.json`,
so replacing or reordering a visual is a manifest edit, not one-off video work.

The checked-in transcript is synthetic and intentionally contains no private
infrastructure, customer data, credentials, or unverified benchmark numbers.
It is a narrative capture, not a claim that each illustrated command is a
literal current CLI surface. The final offline self-check summarizes fak's
canonical public 60-second proof.

Render and verify:

```powershell
$ffmpeg = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
go run ./tools/videogen -config tools/videogen/projects/hero-values/render.json -all -ffmpeg $ffmpeg
```

Outputs are reproducible under `out/`; commit the MP4, poster, GIF, and timeline
only after visual inspection and duration verification.
