---
title: "Production brief: fak microcontext, end to end"
description: "Production brief and script notes for the fak microcontext end-to-end video."
---

# Production brief: fak microcontext, end to end

The master leads with outcomes, then earns them. The first twelve seconds cut
between managed context, shared prefix reuse, and the exact Qwen3.8 result. The
middle uses large, copyable terminal commands to move from installation through
preflight and microcontext orchestration. The final third shows the real fak
operator interface and reconnects every result to a receipt.

## Voice-over spine

> Ten thousand logical contexts. One bounded fabric. And an exact Qwen3.8
> target that answered sooner when thinking was disabled for a task that asked
> for only the answer. Here is the whole path—not a benchmark collage. Install
> one Go binary. Prove its checkpoint without a key or GPU. Run the bounded
> microcontext fabric. Then inspect the exact model campaign and its envelope.
> The MacBook chapter has a harder rule: it appears only when the receipt says
> the graph ran inside fak on Apple Metal. No silent runtime swap. From the
> interface, you can see what ran, what reused context, and what the kernel can
> actually prove. One base. Bounded physical slots. Many logical contexts. Make
> fak the checkpoint for longer-running agents.

## Editorial rules

- Keep the montage under 13 seconds and show the numeric result immediately.
- Put the full historical operating envelope on screen whenever the 3,378→376 ms p95 time-to-first-correct-answer result appears: Qwen/Qwen3.8-27B via vLLM on NVIDIA A100-SXM4-40GB, one prompt, three trials per arm, historical no-thinking campaign. This is not a general model-speed claim or evidence of fak-native/Metal execution.
- Label the 10,000-logical-context / 64-worker selfcheck as deterministic orchestration evidence, not real-model throughput.
- Label the scripted install terminal `RECREATED WALKTHROUGH`.
- The committed claim manifest still marks the native Mac capture `missing`. Keep the `CAPTURE REQUIRED` chapter and preproduction labeling until the receipt and clip satisfy `CAPTURE-MACBOOK.md`; do not publish the preproduction master as proof of the Mac run.
- Preserve chapter metadata so evaluators can jump directly to every claim.
- Music may add momentum but cannot obscure spoken caveats or terminal audio.

The render source, claims, and Mac capture contract live in
`tools/videogen/projects/microcontext-e2e/`.
