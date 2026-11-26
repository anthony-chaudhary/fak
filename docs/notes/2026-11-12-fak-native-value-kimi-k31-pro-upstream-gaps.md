# Why fak-native: what every upstream engine is missing for Kimi-K3.1-Pro (2026-11-12)

**Use:** this note is the worked EXAMPLE of the fak-native value argument — a frontier model lands, day-0 support targets only the newest GPU arch, and every structural gap that remains is a layer fak owns end-to-end (kernels → cache/MMU → scheduler → spec-decode, one binary). Companion sizing note: [`2026-11-12-kimi-k31-pro-nvfp4-h200-sizing.md`](2026-11-12-kimi-k31-pro-nvfp4-h200-sizing.md). Ticket set: filed 2026-11-12 (see "Ticket set" below).

## Verdict

1. **No shipping engine can serve K3.1-Pro on Hopper today.** The per-component SM90 kernels exist (FlashKDA "SM90 and above"; FlashMLA's pinned Hopper FP8 sparse-DSA kernel; Marlin W4A16 for FP4 MoE) — but **no engine wires the combination**: K3.1-Pro's packed 3-state cache (MLA latent + indexer keys + KDA SSM) has zero Hopper wiring anywhere, no engine ships a Hopper preset, and no K3.1 Hopper run is published. Sources: [model card](https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4) (Blackwell only), [recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro.md) (hardware keys gb300/b300/vr200/mi355x/ironwood/810e/m890 — no Hopper), six-lane research in the sizing note.
2. **Expert-granular CPU offload does not exist in any shipping engine** — vLLM has RFCs only ([#33869](https://github.com/vllm-project/vllm/issues/33869), [#63666](https://github.com/vllm-project/vllm/issues/63666)); ktransformers and llama.cpp lack KDA/DSA/NVFP4 support entirely. This is a memory-scheduling problem — exactly fak's cache/MMU layer.
3. **Spec-decode fast paths are arch-locked upstream**: `TOKENSPEED_MLA` and CuTe fused verify/commit are SM10x-only; FlashInfer fused speculative KDA is SM100 ([#54255](https://github.com/vllm-project/vllm/pull/54255)); MTP is impossible (`num_nextn_predict_layers: 0`). The DSpark *draft itself is arch-portable* (AMD serves it on MI355X; V4-DSpark validated on 8×H200) — the portable piece is orphaned because the fused fast path is arch-locked.

## The matrix — what each engine is missing (2026-11-12)

| Capability | vLLM | SGLang | ktransformers / llama.cpp | fak-native path |
|---|---|---|---|---|
| K3.1-Pro on Hopper at all | Recipe Blackwell-only; no Hopper key; no published run | K3 measured 16.8 tok/s on 16×H200, but K3.1 H200 row = "placeholder, not verified" | No `kimi_k31` arch support | Port the proven ingredients + wire the packed cache (tickets filed) |
| Expert-granular CPU offload | RFC-only (#33869, #63666) | none shipping for this arch | static split only; CPU computes misses; no KDA/DSA/NVFP4 | engine-level hot-expert cache in the cache/MMU layer (ESTIMATE ~40–85 tok/s decode on 4×H200 vs ~4 tok/s naive streaming) |
| Spec-decode for the KDA+DSA hybrid | fused verify/commit SM10x-only; MTP dead (0 nextn layers) | EAGLE3 plumbing unverified on SM90 | none | draft is arch-portable → generic verify/commit path fak owns |
| KV dtype policy for this family | K3 H200 bf16-only ([#51313](https://github.com/vllm-project/vllm/issues/51313)) while NVFP4-KV latent is Blackwell-only | FP8 KV = 2× tokens, works on H200 | n/a | one KV-dtype policy layer owned with the cache, per-arch |
| Multi-node Hopper preset | exists for K3 only (2-node TP16, IB pins); none for K3.1 | none for K3.1 | n/a | own the TP16 / TP8×PP2 profile |
| Witnessed receipts for any of it | blog numbers on the newest arch only | community discussion threads | none | `fak serve-receipt` (#14362): deterministic memory-math witness per config |

## The three structural gaps (the "engines are really missing things" evidence)

1. **The last mile is unwired everywhere.** Day-0 lands on the newest arch (Blackwell CuTe/trtllm-gen, #66030/#66037/#66532); the "new model × previous-gen arch" matrix cell is the one nobody fills. Six-lane verdict: ingredients individually SHIPPED on SM90, combination unwitnessed.
2. **Offload is a research RFC everywhere.** Hot-expert caching (Fiddler/MoE-Infinity-class) has measured anchors (17% cache → 46% hit; 10% → >70%) but no shipping integration for linear-attention + sparse-attention + FP4 MoE.
3. **Spec-decode fast paths are arch-locked.** The draft model is portable; the fused verify/commit is not; nobody ships the generic path with tuned-enough blocks to net a gain on Hopper.

## Why ownership (not kernel count) is the lever

- **The single-stream step is overhead-bound, not GEMM-bound.** HBM floor ≈ 0.8 ms/token at TP16 vs ~59.5 ms measured (K3) → ~98% of the step is all-to-all/launch/attention overhead. GB300's 375.9 tok/s/user is an *integration* property (NVLink-fused TP8 + DCP8 + DSpark), not a kernel property. Owning scheduler + comm + kernels in one binary is what lets fak attack the 98%.
- **Upstream fills an N×M×K matrix with separate teams** (N models × M archs × K features). fak owns the whole stack, so the missing cell is *one integration test away* instead of one team's roadmap away.
- **Offload belongs in the cache/MMU.** Engines bolt expert offload on as RFCs; fak's memory layer is the natural home — hot-set placement, PCIe budgeting, and KV/SSM state management are the same subsystem.
- **Witnessed by default.** `fak serve-receipt` (#14362) + claim gates make every config's math deterministic and every perf claim witnessed — upstream ships blog numbers for the newest arch and leaves the rest unwitnessed.

## Honest limits

- **fak has none of this yet** — this note is the requirement set, not a claim. Adjacent seams already in tree: GLM5Next KDA kernels (#10784–#10796, closed), gfx1151 Wave32 KDA recurrent (#12074, closed), expert-spill sizing (#14252/#14258), tiered placement (#13990), FlashMLA SM100 sparse port (#13993, open). Growth-mindset: not built yet; the ticket set below is the next checkable step.
- All performance numbers above are ESTIMATE until witnessed; sourced numbers keep their URLs in the sizing note's evidence index.
- **Operator decision (2026-11-12): K2-Thinking and older models are dropped as serving targets** — K3.1-Pro is the target; predecessor K3 appears only as kernel/shape precedent.

## Ticket set (filed 2026-11-12)

Fak-native set (this note + the sizing note are the evidence base):

- **#14368** — spine: end-to-end witnessed smoke on supported hardware (8×B300, then 16×H200)
- **#14371** — FlashKDA CUTLASS prefill + FLA Triton decode bulk port (SM90/SM100)
- **#14370** — FlashMLA Hopper (SM90) FP8 sparse-DSA decode + indexer bulk port
- **#14369** — DSpark speculative decoding (arch-portable draft, generic verify path)
- **#14372** — hot-expert CPU-RAM cache for discrete-GPU MoE (hit-rate witness; K3.1-Pro 680 GB expert payload)
- **#14373** — 2-node TP16 / TP8×PP2 serving profile (extends the #10946 collectives)
- **#14374** — witnessed bench campaign (spec-decode crossover + offload envelope; Promotion requires #14368/#14371/#14370)

Model-support leaves already filed the same day (parallel session): #14363 (forward dispatch, L8), #14364 (NVFP4 routed-expert MoE), #14365 (MXFP8 attention/shared experts), #14366 (MoonViT-V2), #14367 (1M-ctx KDA/DSA envelope witness) — over the landed L1–L6 GGUF admission stack (`internal/ggufload/kimik31*.go`, `internal/model/kimik31_*.go`). Receipts schema: #14362.

## Evidence index

- Reuse the sizing note's evidence index (all URLs): [`2026-11-12-kimi-k31-pro-nvfp4-h200-sizing.md`](2026-11-12-kimi-k31-pro-nvfp4-h200-sizing.md)
- Model card (Blackwell-only): https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4
- Recipe hardware keys (no Hopper): https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro.md
- Expert-offload RFCs: https://github.com/vllm-project/vllm/issues/33869 · https://github.com/vllm-project/vllm/issues/63666
- DSpark framework (arch-portable draft): https://arxiv.org/abs/2607.05147 · draft: https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview
- K3 on 16×H200 measured: https://huggingface.co/moonshotai/Kimi-K3/discussions/136
- Serving-receipt formalization: https://github.com/anthony-chaudhary/fak/issues/14362
