# MacBook capture contract — issue #8782

This slot is the only acceptable proof for the requested **Qwen/Qwen3.8-27B on a
MacBook through fak's native engine** scene. Until it exists, `render.json`
shows an eight-second `CAPTURE REQUIRED` card and the output filename includes
`preproduction`.

## Required receipt

Capture one continuous terminal take showing:

1. `sw_vers` and `system_profiler SPHardwareDataType` with user/serial identifiers cropped.
2. `fak version` and the exact Git revision used to build it.
3. The exact model id, upstream revision, local file SHA-256, quantization, and
   context size.
4. The launch command selecting fak's native Metal engine explicitly. An
   `auto`, fallback, Ollama, LM Studio, llama.cpp, or remote OpenAI-compatible
   endpoint does not satisfy this scene.
5. A successful prompt and machine-readable receipt naming the engine, model,
   hardware backend, prompt tokens, generated tokens, load time, first-token
   time, decode rate, peak RSS, and exit status.
6. `fak model qwen38-ladder verify
   docs/_witnesses/issue-8623-qwen38-27b/evidence-complete.json` to connect the
   live capture to the already committed exact-target campaign without claiming
   that campaign ran on the Mac.

The exact launch command must come from the native Qwen3.8/Metal implementation
at the captured commit. Do not substitute a plausible command in advance: CLI
shape is part of the witness. Record at 1728x1117 or larger, 60 fps, with a
large monospace font and no notifications, keys, hostnames, private paths, model
cache tokens, or device serial numbers visible.

## Acceptance

- A second reviewer can read engine, model, backend, and result at 720p.
- The receipt and screen recording agree byte-for-byte on model revision and
  artifact hash.
- The engine field says fak-native/Metal; it does not merely say `metal` while a
  different runtime executes the graph.
- Public-safe scrub passes before the clip replaces the placeholder.
- The claim manifest changes `mac-native.status` from `missing` to `witnessed`
  and names the committed receipt and clip.
