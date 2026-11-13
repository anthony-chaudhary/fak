# Spine map — the six top K3.1 items (2026-11-12)

Canonical state lives in the GitHub issues; this doc is the spine decomposition that
turns each top item into **minimal end-to-end spine → follow-on leaves → witness**,
grounded in the in-tree seams as of trunk `b556cd799`. Parent serving spine: #14368.

| # | Item | Spine (this map) | Hardware-gated follow-on |
|---|---|---|---|
| #14371 | FlashKDA CUTLASS prefill + FLA Triton decode | host chunk-reformulation prefill, tolerance-equivalent vs KDA oracle + sotamatrix row | CUTLASS `.cu` port + H200/B300 tok/s |
| #14370 | FlashMLA SM90 FP8 sparse-DSA decode | filed breakdown (vendor + build admission + parity harness) | sm_90 TU build + H200 decode rung |
| #14369 | DSpark spec-decode | `dspark` selector + rank-512 Markov geometry + confidence-scheduled block length (Qwen3.8 target) | K3.1 wiring (needs #14363 forward), H200 crossover |
| #14372 | hot-expert CPU-RAM cache | advisor serve hookup + h=0.5/0.75/0.85 budget-fraction sweep + Qwen3.8 shape row | K3.1-Pro 680 GB profile + 4xH200 run |
| #14373 | 2-node TP16 / TP8xPP2 H200 profile | hwspec profile rows + profile math witness | `--nnodes/--node-rank` flags + real 2-node boot |
| #14374 | witnessed bench campaign | campaign scaffold + arm-matrix plan folding `fak.serving-sweep.v1` receipts | witnessed arms after #14368/#14371/#14370 |

## #14371 — FlashKDA (spine: chunk-reformulation prefill, host)

- **Attach**: `internal/model/kimik31_kda.go` — `ForwardKimiK31KDAPrefillSeq` (:356) is the
  bit-exact oracle (prefill==steps, `TestKimiK31KDAPrefillSeqEqualsSteps` :237; independent
  float64 transcription in `kimik31_kda_naive_test.go`). State/params reuse:
  `GLM5NextKDALayerState`, `KimiK31KDAParams` (gate lower bound -5.0, use_full_rank_gate).
- **Spine**: `ForwardKimiK31KDAChunkedPrefill` — the chunk-level reformulation of the
  delta-rule scan (within-chunk cumulative decay + inter-chunk carry), host f32, equivalent
  to the token scan on synthetic K3.1 shapes (96 heads x 128 dim, conv kernel 4). Witness
  precision: chunk=1 must equal the scan exactly (same op order); larger chunks are a
  reordered arithmetic so the witness is a tight relative tolerance vs the scan AND vs the
  independent float64 transcription, with the bound stated in the test. This is the
  algorithm the CUTLASS kernel computes; the kernel port inherits the parity test.
- **Prior art (sotamatrix)**: add the FlashKDA row (MoonshotAI/FlashKDA CUTLASS chunked
  prefill, SM90+; fla-org/flash-linear-attention#852 Triton decode + auto-dispatch) with
  route=borrow + oracle=the in-tree KDA oracle. `fak sota` check runs before any kernel.
- **Leaves**: CUTLASS `.cu` port beside `cuda_qwen35_gdn.go`/`cuda_flashmla_sparse_prefill.cu`
  arch-gated via `cuda_arch_coverage.go` SM90 entry; FLA Triton decode fallback seam;
  GLM5Next session dispatch behind a `VulkanGLMKDAStepper`-style interface
  (`glm5next_forward.go:597`); H200/B300 prefill tok/s (fleet nodes).
- **Witness**: chunked==scan equivalence on K3.1 shapes (chunk=1 exact; larger chunks
  tight-tolerance, CPU, in-session); device parity + tok/s delta on H200/B300
  (hardware-gated, fleet).

## #14370 — FlashMLA SM90 FP8 sparse-DSA decode (spine: filed breakdown)

- **Attach**: SM100 pattern to clone — `cuda_flashmla_sparse_decode.cu` (sm_100a/103a TU,
  vendor-pinned `third_party/flashmla/`), build contract `internal/computebuild/cuda_flashmla_prefill.go`,
  flags `internal/devcmd/build_cuda.go:79-81`; DSA oracles `internal/model/kimik31_dsa_forward.go`
  (`ForwardKimiK31DSAStep` :60, top-k 2048 / 32x128 indexer geometry pinned in
  `internal/ggufload/kimik31.go:32,88`); fail-closed fence `internal/model/kvlayout.go:233-306`
  (`selectSparseIndexScorer` requires SM>=100).
- **Spine** (filed; needs the pinned upstream tree + nvcc, not available on this host):
  vendor ba89a34's SM90 sparse decode kernel dir (absent from `third_party/flashmla/csrc/`,
  which vendors only `sm100/decode/sparse/head64`) + `DECODE-PINNED-SOURCE.json` entry +
  build tag `fak_cuda_flashmla_decode_sm90` + `--arch sm_90` admission + the FIRST
  device-vs-K3.1-DSA-oracle parity test harness (the SM100 qualification is pinned to the
  FlashMLA physical fixture; no oracle parity test exists for any arch).
- **Leaves**: indexer (lightning/top-k-2048 CUDA; fak's only indexer ports are the SGLang
  V4.1 block indexer `block_sparse_indexer.go` and host `SelectKimiK31DSATopK`); paged-KV
  record bridge (oracle flat cell arrays -> 528B E4M3+E8M0 device record); runtime dispatch
  from serving decode (today qualification-only); H200 decode rung (fleet).
- **Witness**: sparse-DSA decode equivalence vs the dense-MLA/K3.1 DSA oracle on synthetic
  shapes (top-k 2048), then end-to-end in the #14368 spine smoke (hardware-gated).

## #14369 — DSpark spec-decode (spine: method + scheduling on the generic engines)

- **Attach**: closed selector set `{mtp, ngram, dflash2}` at `cmd/fak/serve.go:469/494`
  (`validateServeSpeculativeSelector`); DFlash-2 drafter `internal/model/dflash2_reference.go`
  (5-tap readback, `len(taps) > 5` refused :109, `rank <= 256` bound); block driver
  `internal/agent/inkernel_dflash2.go` (`runDFlash2Request` :489, fixed `min(8, remaining)`);
  confidence-scheduling precedent `internal/model/selfspecgov.go` `DraftDepth` :89;
  acceptance law `internal/model/speculative.go` (#13664 monitors).
- **Spine**: `dspark` selector member + DSpark geometry (rank-512 Markov-head correction
  past the 256 bound) + confidence-scheduled per-request block length replacing the fixed
  budget, wired through an `EnableDSpark` sibling of `EnableDFlash2`
  (`internal/gateway/gateway_planners.go:128`). Target: Qwen3.8 first (model-agnostic;
  K3.1 has no forward path yet — `ClassifyForwardPath` has no `ForwardKimiK31` case,
  `internal/model/arch_support.go:130`).
- **Leaves**: K3.1 DSpark wiring (gated on #14363 forward + tap readback in the K3.1
  forward); fused verify path (upstream is SM10x-only; SM90 uses the generic verify path);
  H200 crossover measurement (owned by #14374).
- **Witness**: driver test (block draft + Markov correction + scheduled length) on the
  Qwen3.8 target; acceptance length + tok/s at c=1/8/32/64 vs no-spec baseline is #14374's
  owned witness (hardware-gated).

## #14372 — hot-expert CPU-RAM cache (spine: hookup + h-sweep, Qwen3.8 first)

- **Attach**: learner->advisor->residency loop — `internal/model/expert_hot_set.go`
  (`ExpertHotSetLearner`), `expert_hot_set_advisor.go`, `internal/ctxmmu/expert_residency_advisor.go:62`
  (`NewExpertResidencyPolicyWithAdvisor`); policy #14251 `internal/modelloadplan/expert_cache_lrfu.go`
  (policy-only); ring stats `internal/model/expert_ring_*.go`; telemetry
  `internal/model/expert_cache_telemetry.go:22` + `pkg/moecache`; bench hosts
  `internal/model/expert_cache_bench.go` (drives the SHIPPED ring) and `cmd/hotsetcheck`.
- **Spine**: (a) the documented advisor serve hookup (advisor.go header names it), (b) the
  h=0.5/0.75/0.85 budget-fraction sweep added to `expert_cache_bench.go` with hit-rate
  receipts, (c) the Qwen3.8-class shape row (expert count, per-expert bytes) so the sweep
  is model-agnostic, (d) live ring counters folded into `Telemetry()` per turn.
- **Leaves**: popularity-profiled placement (heat-weighted within the ring; today
  `expert_spill_placement.go` grades per layer, not per-expert heat); `ExpertCacheLRFU`
  adapter into the residency loop; K3.1-Pro 680 GB profile; 4xH200 run (fleet).
- **Witness**: hit-rate + decode tok/s at h=0.5/0.75/0.85 on the Qwen3.8-class bench vs
  the ESTIMATE ceiling (42/84/140 tok/s); K3.1-Pro profile is the follow-on rung.

## #14373 — 2-node TP16 / TP8xPP2 H200 profile (spine: profile rows + math witness)

- **Attach**: `pkg/hwspecs/hwspecs.go` (h200 row exists; `fak.hwspecs.v1`); launcher shape
  `cmd/fak/cluster.go` (`coordinator|worker`); collectives `internal/model/dist_collective.go`
  (TCP star, cross-node capable, host f32; per-rank ForwardTP "co-arrives with MLA-aware TP"
  :46) + device NCCL PG `internal/compute/cuda_collective_pg.go` (same-box only); warmup
  witness `cmd/fak/serve_fit_confirm.go` (one-line weights/KV report) + `RunWarmup`
  (`internal/gateway/readiness_warmup.go:385`); admission gate
  `internal/bench/placementreservation.go` (12 witnessed NCCL ranks — TP16's 16 clear it).
- **Spine**: hwspec profile rows for the two 2-node shapes (TP16: ~101-110 GB weights/GPU;
  TP8xPP2: 96.79 GiB weights + 27.15 GiB KV = 1.66M tokens, 256k ctx, Marlin expert-dim
  padding avoided) with the sizing math as the deterministic witness; IB requirement stated
  in the profile.
- **Leaves**: `--nnodes/--node-rank/--master-addr` on the cluster launcher; cross-node
  device-NCCL PG extension (or DistComm-routed device tensors); MLA-aware TP;
  2-node boot witness on fleet H200 (warmup line + one e2e request).
- **Witness**: profile rows + math (in-session, deterministic); 2-node boot on fleet H200
  (hardware-gated).

## #14374 — witnessed bench campaign (spine: scaffold + arm-matrix plan)

- **Attach**: `fak webbench` serving-sweep (`internal/webbench/serving_sweep.go`,
  receipt `fak.serving-sweep.v1`); multi-arm pattern `fak bench h2h --arm`
  (`bench_h2h_bringup.go`); spec counters `internal/enginestep/enginestep.go:109-116`
  (`spec_accepted_per_round`, `spec_position_*`); claim gate
  `internal/webbench/serving_sweep_claim.go:54`.
- **Spine**: campaign scaffold `docs/benchmarks/k31-campaign/` + the arm-matrix plan
  (arms: no-spec / num_spec=4 / num_spec=5 x c=1/8/32/64; offload arm h=0.5/0.75/0.85
  after #14372) that folds per-arm `fak.serving-sweep.v1` artifacts into one campaign
  index; additive spec-metric join (acceptance length into sweep points).
- **Leaves**: offload receipt fields (schema home for hot-expert hit-rate; additive);
  witnessed arms on 16xH200 + 8xB300 (Promotion-requires #14368/#14371/#14370); anchors
  to beat/explain: K3 16.8 tok/s (SGLang 16xH200), GB300 375.9 tok/s/user, ~98% overhead
  share roofline.
- **Witness**: receipts under `docs/benchmarks/k31-campaign/` per the bench-workstream
  doctrine (separate from fixes); every number net-true end-to-end.

## Relations (typed, per the issue contract)

- `Start blocked by`: none — every spine has an executable disjoint slice.
- `Coordinates with`: #14368 (parent spine) for all six; #13993/#14000/#14266 (SM100
  siblings) for #14370; #14251/#9137/#9142/#13128/#3886 for #14372; #10946 for #14373;
  #12190/#12727/#13664 for #14369; #14362 for #14374.
- `Promotion requires`: #14374 closure/claims gated on #14368/#14371/#14370 (per its body).

## Execution log

- 2026-11-12: seam maps captured (six explore passes, trunk `b556cd799`); spine map
  written; in-session spines dispatched: #14371 host chunked prefill + sotamatrix row,
  #14374 campaign scaffold + arm plan, #14373 hwspec profile rows, then #14372 hookup +
  h-sweep, then #14369 dspark method. #14370 spine filed (needs pinned upstream tree +
  nvcc; not available on this host).
