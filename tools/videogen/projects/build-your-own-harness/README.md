# Build your own harness video

**For:** developers who want their own agent product rather than another fixed chat shell.

**Problem:** a harness looks like a pile of UI, provider, tool, memory, and policy choices, so the first useful end-to-end run gets buried under framework work.

**Today:** people either fork a large harness or keep assembling wrappers whose ownership boundaries are unclear.

**Better because:** `fak harness init` generates the replaceable shell while keeping the product core yours, then the same kernel seam connects models, tools, memory, policy, and UI.

**Witness:** this 20-second silent trailer shows the whole value path: choose the parts, generate one typed blueprint, run the real prompt-to-tool spine, and copy the starting command. The moving packets deliberately improve on the older isolated “ball” visual: motion now carries meaning along explicit connections, assembles a blueprint in sequence, and ends in a witnessed self-check.

`trailer.json` is the editable source. The existing first-class Go trailer renderer produces the MP4, GIF, poster, phone-scale contact sheet, and machine-readable readability audit.

Render and verify:

```powershell
$ffmpeg = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
go run ./tools/videogen -trailer -config tools/videogen/projects/build-your-own-harness/trailer.json -verify
go run ./tools/videogen -trailer -config tools/videogen/projects/build-your-own-harness/trailer.json -all -ffmpeg $ffmpeg
```

Before publishing, inspect `out/review-contact-sheet.png` at 680 px and 360 px and confirm every beat and the CTA can be transcribed without this README.
