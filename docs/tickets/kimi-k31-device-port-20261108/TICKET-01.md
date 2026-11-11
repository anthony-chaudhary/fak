# Kimi K3.1 device port: staged plan for H200 qualification

Ref HITL request `kimi-k31-h200-qual` (stage=planned); bring-up parent ticket
`1611-kimi-k31`. This is a plan ticket: it stages the subsystem port; each stage
below becomes its own leaf when executed. No code lands with this file.

Process cause: none

## Pinned references

- Upstream reference implementation: sgl-project/sglang main tip
  `a7d9f81ee24c5cceeceeb4ccb9072f68d56cb10d` (`python/sglang/srt/models/kimi_k3.py`
  + `kimi_linear.py`, fetched 2026-11-08), Apache-2.0. The introduction commit is
  "[Feat] model: Add basic Kimi-K3.1 support (NV, AMD) (#48415)" =
  `61a3c7f2a45f6cfd5fdcec90fd5019d2334c2f0e`; re-verify pins at claim-window time
  and record exact revisions in receipts.
- Study notes: the 2026-11-08 SGLang deep-study note (kernel inventory rows
  1-18, borrow candidates sglang-kimi-k31-1..9, ranked port plan) and the
  2026-11-07 bring-up study note (loader seams L0-L11, memory-fit math, qual
  arms; public companion: docs/notes/KIMI-K31-PORT-STUDY-2026-11-06.md). The
  two study notes live in the private companion repo; this ticket carries the
  load-bearing pins forward so execution does not depend on them. Borrow
  verdicts are ADAPT/DIRECT-PORT, structure+oracle only.
- Already landed in this repo (the oracles the composite calls per layer):
  `internal/ggufload/kimik31*.go` (arch admission + tensor census + tokenizer pin),
  `internal/model/kimik31_kda.go` (KDA oracle), `internal/model/kimik31_dsa.go` +
  `kimik31_dsa_forward.go` (Gated-DSA oracle), `internal/model/kimik31_trio.go`
  (paged top-k contract), `internal/kimik3page` (64B-aligned paged layout),
  plus the reusable device seams: `internal/compute/cuda/qwen35_gdn.cuh` (GDN
  decode/prefill register path), `internal/compute/dsa.go` +
  `internal/compute/bitonic_topk.go` + `internal/model/paged_glmdsa.go`
  (GLM-5.2 DSA seam), DeepSeek MLA cache insert, and the MoE
  device spine (`moe_device_gateup.go`, `v41_expert_parallel_device.go`,
  slotstream expert paging, `glm5next_moe_router.go`).

Model geometry (config.json @ pin): 93 layers, hidden 7168, vocab 163840,
max_position 1048576; 69 KDA layers + 24 full-attention layers at one-based
indices 4,8,...,92,93 (the final layer breaks the period-4 pattern, so a bare
`%4` predicate is wrong); `first_k_dense_replace: 1`; `attn_res_block_size: 12`;
KDA head_dim 128 x 96 heads, short conv kernel 4, full-rank gate; absorbed-form MLA
(q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128,
`mla_use_nope: true`, `mla_use_output_gate: true`); DSA indexer 32 index heads x
128 dim, token-level top-2048 (`index_topk_freq: 1`, `index_skip_topk_offset: 1`);
latent MoE 896 routed experts, top-16, latent 3584, expert FFN 3072, SiTU
activation (beta 4.0, linear_beta 25.0), 2 shared experts, sigmoid grouped
router with `e_score_correction_bias` + renormalize.

## Process contract (fak.port-process.v1, from the request's qualification_plan)

Every stage holds these numbers; a stage that cannot hold them is not done.

- Precision: Q4_K/Q6_K stored AND loaded (packing_stage=load from BF16 upstream
  safetensors); F16 activations; F32 accumulators; `forbid_precision_widening`.
  Capability rejection must be visible, never a silent F32/Q8 substitution:
  `fak serve --model kimi-k31 --quant unsupported-quant --dry-run` requires a
  visible rejection. Precision witness:
  `fak serve --model kimi-k31 --profile on --dry-run` (selected kernels,
  stored/loaded types, fallback count, staged conversion bytes).
- Host decode budget (steady decode, eager execution mode): max 64 kernel
  launches/token, max 4 synchronizations/token, max 2 scratch allocations/token.
- Layout: row-major F16 activations; KDA state [layers, heads, head_dim,
  head_dim] pool; DSA indexer pages; quant block bytes and tile strides from the
  GLM5Next KDA/DSA/MoE reference layouts. Steady-state repack is FORBIDDEN:
  zero repack bytes/token (`max_repack_bytes_per_token: 0`), witnessed by
  warmed-decode profile of `fak serve --model kimi-k31 --profile on`.
- Workload for all measurements: batch 1, prompt 4096, decode 256, warmup 64,
  ABBA timing protocol, timing profiling off, diagnostic profiling on.
- Serving opt-in during bring-up: `fak serve --model kimi-k31` (default-on once
  qualified).

Parity witness (canonical, every stage):

```text
go -C C:/work/fak test ./internal/model/... -count=1 -run 'TestKimiK31.*Parity'
```

Fixed tensors, atol=0.001 rtol=0.001; the landed L5/L6 host oracles are the
parity reference, so the composite must keep each layer's oracle path callable
per layer. Route regression:
`go -C C:/work/fak test ./internal/model/... ./internal/ggufload/... -count=1`.

## Stages (dependency order)

### Stage 1 — Composite forward (L8)

- Scope: wire `model.ClassifyForwardPath` to a native `ForwardKimiK31` composite
  that runs embed -> 93 x [norm -> attn(KDA|MLA+DSA) -> residual(s) -> norm ->
  MoE] -> final norm -> lm_head under one `layer_types` dispatch, calling the
  landed KDA and Gated-DSA host oracles per layer; replace the interim
  boundary-only dispatch (the L1 admission writes K3.1-specific `kda`/`dsa`
  layer-kind tokens that degrade to unknown while the forward is unbuilt) with
  the native `ForwardKimiK31` path in the same change. Whole-forward greedy-id
  parity on the tiny-model fixture before any perf claim.
- Files touched: `internal/model/arch_support.go`, new
  `internal/model/kimik31_forward.go` (+ tests), calling
  `internal/model/kimik31_kda.go` / `kimik31_dsa_forward.go` as oracles;
  embedding/lm_head reuse existing seams.
- Parity witness: canonical command above; adds whole-forward
  `TestKimiK31CompositeParity` (greedy ids vs oracle-composed reference).
- Precision contract: global contract applies; linear paths route through the
  existing Q4_K/Q6_K quant kernels (q4k_matmul / iq* matvec), no widening.
- h2h claim-window condition (rtp-test-h200): NOT arm-ready — host-only stage;
  the candidate arm needs the device path of stages 5-6. Baseline (SGLang) arm
  readiness is independent of this stage. No claim window may be acquired on
  the strength of this stage.

### Stage 2 — SiTU activation

- Scope: DIRECT-PORT the SiTU-GLU elementwise op (SituAndMul, beta 4.0,
  linear_beta/clamp 25.0) from SGLang `layers/activation.py` +
  `kernels/ops/activation/kimi_k3_situ_and_mul.py` @ `a7d9f81ee2` (Apache-2.0,
  borrow sglang-kimi-k31-1): host form first, then the CUDA elementwise kernel.
- Files touched: `internal/model` (activation dispatch into the MoE expert
  path), `internal/compute/cuda` (new small elementwise .cuh + Go seam), tests.
- Parity witness: canonical command; adds `TestKimiK31SituAndMulParity` against
  the pinned SGLang formula on fixed tensors.
- Precision contract: F16 in/out, F32 intermediate inside the clamp; no
  widening of stored weights; covered by the precision witness command.
- h2h claim-window condition (rtp-test-h200): NOT arm-ready — activates the MoE
  expert path only after stage 6 geometry; no claim window.

### Stage 3 — Cross-layer attention residual (attn-res, block size 12)

- Scope: DIRECT-PORT the cross-layer attention residual: every 12th layer is a
  block-write layer carrying its attention output; subsequent layers add the
  carried residual (SGLang `kimi_k3.py:2832-3061` region + llama.cpp
  cross-layer residuals; borrow sglang-kimi-k31-2). Host accumulator inside the
  composite; the qwen35 "attn_residual" per-layer add is NOT this mechanism.
- Files touched: `internal/model` (residual carry in `kimik31_forward.go`),
  tests pinning the write/add cadence over 93 layers.
- Parity witness: canonical command; adds
  `TestKimiK31AttnResidualParity` (block-write cadence + carried values vs
  oracle dumps).
- Precision contract: F32 accumulator for the carried residual stream, F16
  storage of the written block value; forbid_precision_widening applies.
- h2h claim-window condition (rtp-test-h200): NOT arm-ready — correctness
  prerequisite only; no claim window.

### Stage 4 — MLA output gate

- Scope: DIRECT-PORT the `mla_use_output_gate` sigmoid gate in TP-local space
  before o_proj on all 24 full-attention layers (SGLang
  `kimi_k3.py:2422-2470` @ `a7d9f81ee2`; borrow sglang-kimi-k31-3).
- Files touched: `internal/model` (MLA forward path feeding
  `kimik31_dsa_forward.go`), tests.
- Parity witness: canonical command; adds
  `TestKimiK31MlaOutputGateParity` (gated vs ungated divergence pinned).
- Precision contract: F16 gate input/output, F32 sigmoid accumulation; no
  widening.
- h2h claim-window condition (rtp-test-h200): NOT arm-ready — host-oracle
  slice; the device attend still comes from stage 5; no claim window.

### Stage 5 — KDA/DSA CUDA kernels

- Scope: the decode-path device kernels.
  (a) KDA fused decode: adapt the `qwen35_gdn.cuh` register path (head_dim 128,
  conv 4) into the K3.1 form — FLA-style L2-norm placement on q/k, full-rank `g`
  gate, per-value-channel decay, fused gated-RMSNorm(sigmoid) readout, per-segment
  transposed fp32 conv + recurrence in ONE kernel (borrow sglang-kimi-k31-5,
  SGLang `kda_fused_decode.cuh` @ `a7d9f81ee2`), with the Q4_K/Q6_K
  expert-free linear path through the existing quant kernels.
  (b) DSA indexer: token-level scoring + deterministic top-2048 with
  `skip_topk_offset: 1` and per-cell index-key cache paged in
  `internal/kimik3page` (borrow sglang-kimi-k31-6, SGLang
  `srt/layers/attention/dsa/` @ `a7d9f81ee2`); the device topk MUST hold the
  same deterministic tie rule as `SelectKimiK31DSATopK` (host oracle) or the
  argmax-exact witness diverges.
  (c) Sparse MLA attend: absorbed-form, no-RoPE dot products, sigmoid output
  gate, sparse softmax-V over ONLY the selected cells via the
  DSASparseBackend seam (borrow sglang-kimi-k31-7).
- Files touched: `internal/compute/cuda` (new KDA fused-decode .cuh as the
  qwen35_gdn successor), `internal/compute/dsa.go`, `internal/compute/bitonic_topk.go`,
  `internal/model/paged_glmdsa.go`, `internal/kimik3page` (IK-cache pages),
  `internal/model` (device dispatch behind the composite).
- Parity witness: canonical command plus
  `go test ./internal/compute/... -run 'GDN|DSA' -count=1`; device-vs-host
  oracle parity on fixed fixtures, selection bit-identity included.
- Precision contract: F16 activations, F32 accumulators and fp32 conv/recurrence
  state; KDA recurrent state stays fp32 in the paged pool; Q4_K/Q6_K weights,
  forbid_precision_widening; launch counts for this stage feed the 64-launches /
  4-syncs / 2-scratch budget measured at bring-up.
- h2h claim-window condition (rtp-test-h200): NOT arm-ready until stage 6 —
  the candidate arm serves end-to-end only when the MoE geometry lands; no
  claim window yet.

### Stage 6 — 896-expert MoE geometry

- Scope: latent-MoE front merge (fp32 MoEGate + down-proj 7168->3584 + shared
  gate_up merged front weight) and routed latent geometry: 896 experts x
  gate_up 3584->2x3072 (w1/w3 non-interleaved) + down 3072->3584, top-16
  sigmoid grouped router with `e_score_correction_bias` + renormalize,
  routed_scaling 1.0, latent norm, up-proj 3584->7168, 2 shared experts
  (borrow sglang-kimi-k31-4, SGLang `kimi_k3.py:476-700` @ `a7d9f81ee2`);
  slotstream expert paging sized for the 896-expert slot geometry; routed
  expert GEMM on the existing Q4_K/Q6_K quant kernels.
- Files touched: `internal/model/moe.go` (latent front/latent norm),
  `internal/compute` (`moe_device_gateup.go`, `v41_expert_parallel_device.go`,
  slotstream plan, `glm5next_moe_router.go` bias/grouped params), tests.
- Parity witness: canonical command plus
  `go test ./internal/model/... -run TestMoE -count=1`; closed oracle over
  fixed router-score fixtures incl. 896-expert boundary cases.
- Precision contract: router scores in F32 (noaux_tc), expert GEMM Q4_K/Q6_K
  weights with F16 activations and F32 accumulators, forbid_precision_widening;
  steady decode must show zero repack bytes/token.
- h2h claim-window condition (rtp-test-h200): CANDIDATE ARM READY CONDITION —
  stages 1-6 parity green AND the process-contract budget witnessed on the node
  (`fak serve --model kimi-k31 --profile on`: launches <= 64/token, syncs <=
  4/token, scratch allocs <= 2/token, repack bytes/token == 0; fuse to the cap
  first if the naive count exceeds it — the fusion pass is part of this stage's
  exit, not a later optimization). Only then acquire the exclusive-node claim
  window and run the ABBA pair, one arm per invocation:

```text
# baseline arm (SGLang reference)
fak bench h2h run --arm sglang=http://127.0.0.1:8002/v1 --model k31 --host rtp-test-h200 --cold 512,2048,8192 --reps 3 --decode 256 --turns 6 --system 4096 --turn-tokens 256 --answer 16 --speculative 0 --gpu-busy off --ledger /data/bench/h2h-k31-sglang.jsonl --json
# candidate arm (fak port)
fak bench h2h run --arm fak=http://127.0.0.1:8003/v1 --model k31 --host rtp-test-h200 --cold 512,2048,8192 --reps 3 --decode 256 --turns 6 --system 4096 --turn-tokens 256 --answer 16 --speculative 0 --gpu-busy off --ledger /data/bench/h2h-k31-fak.jsonl --json
```

  Metrics: decode ms/token and decode tok/s (ABBA, profile-off). Acceptance:
  0-error h2h on both arms + measured decode tok/s delta folded into
  RESULTS.md. If the baseline arm types-F on this node (no documented single-H200
  SGLang shape), the ratio bar falls back to the R0 reference arm per the
  bring-up study; record the typed result in the receipt.

## Out of scope (INSPIRE-ONLY / follow-ons)

- Multi-GPU serving machinery (MegaMoE, breakable graphs, prefill CP, DFlash;
  SGLang perf PRs #48584/#48710/#48711/#48740/#48721): only if a multi-GPU arm
  is ever requested.
- Vision tower (MoonViT3d): text stack only, matching the request.
- Weights acquisition: HF license is `other` — pin the exact license text and
  the converter revision (llama.cpp `conversion/kimi_k3.py` is the concordant
  one the landed oracles cite) before any node copy; a separate gated step.
- Post-qualification leaves: KDA chunked prefill on device, IK-cache paging at
  long context, prefill parity + measured prefill tok/s.

## Done condition (this plan ticket)

- This file lands on public trunk scrub-clean (leak audit exit 0) with DCO
  sign-off; it names every stage's scope, files, parity witness, precision
  contract, and claim-window condition.
- Execution starts at stage 1 as its own leaf worktree under this plan; the
  request advances to executing stage 1 once this lands.
