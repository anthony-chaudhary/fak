# Concept study — colibri (consumer-hardware GLM-5.2 MoE SSD-streaming inference engine) → witnessed borrows for fak

- **Date:** 2026-07-11
- **Source (pinned):** [`JustVugg/colibri`](https://github.com/JustVugg/colibri) @ `1bdaeee82ed143c6b7480186e5b9a4614909aa55` (`1bdaeee`) — **Apache-2.0** (fak is Apache-2.0 → every borrow is **`inspire`**, clean-room, source-cited; **no bytes vendored**).
- **What it is:** a from-scratch, single-C-file (`c/glm.c`, ~2,400 lines) inference engine that runs the **744B GLM-5.2 MoE** (and OLMoE) on **consumer hardware** by keeping the ~17B dense part resident at int4 (~9.9 GB) and **streaming the 21,504 routed experts from SSD** with a per-layer LRU cache + pinned hot-store + OS page cache as a free L2. Faithful GLM-5.2: MLA q/kv-LoRA + compressed KV (576 floats/token, 57×), DeepSeek-V3 sigmoid router, DSA lightning-indexer sparse attention, **native MTP speculative decode**.
- **Epic:** **#4352** `epic(colibri-study)`. Label `colibri-inspired`.
- **Method (deep, fanned-out, witnessed):** 12 parallel subsystem readers over the pinned clone (MTP/spec-decode · expert residency · quant kernels · MLA/KV/DSA · conversion pipeline · resource planning · disk I/O · CUDA tier · tokenizer/formats · OpenAI server+web · eval/oracle · worldview) + a **completeness critic** (opened the unread `c/olmoe.c` differential-twin engine and the `run_text`/`moe()` regions no reader mapped) → **202 candidates**, each **witnessed on-axis** against fak (`fak_feature_query`/`fak index` + raw grep + reading the fak seam), classified at the property grain: **104 PARTIAL · 28 DIVERGENT · 70 PRESENT**. Filed 12 leaves + 2 enriches; recorded the rest.

## The decisive tension — same problem, opposite trust model

colibri optimizes for **one user who owns the whole box**: it grows a binary KV file and commits by rewriting the trailing record-count last (O(1) crash-safe truncate-rollback, `c/glm.c:1888`), infers the quant tier from raw byte-count with no on-disk discriminant (`c/glm.c:696`), silently falls a failed GPU tensor back to CPU (`c/glm.c:482`), and fits the whole runtime in one auditable file. fak optimizes for an **audited multi-tenant fleet**: append-only tamper-evident hash-chain stores (no Truncate API — `internal/journal/journal.go:140`), explicit format discriminants (`internal/l3kv/store.go:38`), refuse-not-diverge on a failed tensor, files capped ~1500 lines for per-unit steerability. Those are the **28 DIVERGENT** axes — earned "we chose differently, and the reason still holds." Underneath the trust model, colibri's **physics-bound single-node MoE streaming** is directly borrowable — it lives the "answer correctly on a machine cheaper than an H100 fan" constraint fak's Mac/CPU-offload tracks (#2722/#3809/#974) also target, with honest, community-measured, caveat-inline numbers (the same distrust-of-self-report thesis fak's benchmark-honesty gates enforce).

## Candidate table — FILED borrows (all `inspire`)

| Borrow | Source `@1bdaeee` | Axis | fak seam | Witness | Filed |
|---|---|---|---|---|---|
| MTP/draft-head quant-precision floor by measured **acceptance** (not cosine) | `c/tools/convert_fp8_to_int4.py:148` | per-tensor precision floor keyed on downstream accept-rate | `internal/model/v4quant_admit.go:105` (`V4ClassMTP`→nil, "dropped→SKIP") | PARTIAL | **#4353** |
| Runtime MTP self-spec **acceptance governor** (auto-off charged vs MoE page-in) | `c/glm.c:1606` | adaptive speculation-disable on rolling accept floor | `internal/model/config.go:789` (`SelfSpeculationSubstrateReady`, no consumer) | PARTIAL | **#4354** |
| **Batch-union** MoE across the draft/verify window | `c/glm.c:1619` | read each unique routed expert once per batch, not per position | `internal/model/verify.go:69` (MoE excluded from batched verify) | PARTIAL | **#4355** |
| **Latent-resident** MLA KV cache for GLM-DSA | `c/glm.c:1023` | store 576-float latent, reconstruct per-head K/V at read; dedup shared RoPE key | `internal/model/glm_dsa_session.go:435` (up-projects + dups per head at write) | PARTIAL | **#4364** |
| **Absorbed** MLA decode (W_UK into query) | `c/glm.c:1088` | O(T·kv_lora) attend-the-latent vs O(T·H·hd) reconstruct-then-attend | `internal/model/kvlayout.go:108` (`reconstructKV` "NAIVE MLA read path") | PARTIAL | **#4356** |
| **Value-aware** pagedRing residency (hysteresis + LFU-decay) | `c/tier.h:22,27` | anti-thrash promotion margin + phase-tracking decayed heat vs pure LRU | `internal/model/paging_ring.go:107`; `internal/compute/kvcost.go:199` | PARTIAL | **#4357** |
| **Persistent** expert-usage histogram → warm-start pins + between-turns re-pin | `c/glm.c:2152,1835` | personalize the pinned hot-set to observed workload, online, no profile run | `internal/model/expert_replay.go:145` (in-memory only) | PARTIAL | **#4358** |
| **MADV_WILLNEED** readahead + **per-expert sub-range pread** | `c/glm.c:1235`, `c/st.h:219` | read/compute overlap + read-amplification avoidance on demand-paged experts | `internal/model/mmap_unix.go:34`; `gguf_glm_tensors.go:376` | PARTIAL | **#4359** |
| Native **FP8 e4m3 128×128 block-scale dequant** (quant-on-load) | `c/tools/convert_fp8_to_int4.py:99` | ingest GLM-5.2-FP8/DeepSeek-V4 block-scaled weights directly, ragged-crop correct | `internal/model/safetensors.go:514` (`decodeMXFP4Blocks`, no e4m3 path) | PARTIAL | **#4360** |
| `fak serve --plan-json` **inspectable memory-plan dry-run** | `c/resource_plan.py:160` | versioned header-derived plan you read before committing 370 GB + RAM | `internal/compute/capacity.go:97` (`MemoryPlan`, no dry-run emit) | PARTIAL | **#4361** |
| **OS page-cache reserve** as a first-class host-mem budget term | `c/glm.c:2365` | absolute page-cache floor (measured cliff), not a headroom fraction | `internal/compute/capacity.go:482` (`BudgetAfterHeadroom` fractional) | PARTIAL | **#4362** |
| **Loglikelihood MC** quant-fidelity scorer (`acc_norm`) + offline fixture | `c/tools/eval_glm.py:79` | behavioral MMLU/HellaSwag/ARC accuracy of a quant lane, not just numeric parity | `internal/model/forward.go:12`; `cmd/q8bench` (parity only) | PARTIAL | **#4363** |

**Enriched (no re-file):** **#1469** (restore-on-access — colibri validates byte-identical warm resume via `DeserializeSpan`-into-live, `c/glm.c:1920`) · **#4202** (lossless-under-temperature spec accept — `carry_ban` rejection-sampling, `c/glm.c:1626`).

## Recorded, file-later PARTIAL candidates (decomposed, not injected in one wave)

Grounded at `@1bdaeee`, dropped from this filing pass to avoid a tracker flood — promote from here when a lane picks them up:
`/v1` admission overload semantics (Retry-After + queue_timeout + scheduler-close, `internal/gateway/admission.go`) · reject unhonorable OpenAI params with 400 (`internal/gateway/wire.go:313`, enriches #1705) · single-model identity gate + `/v1/models/{id}` (`internal/gateway/http.go:1008`) · per-MoE-layer partitioned LRU (`paging_ring.go:45`) · mlock the never-evict tier vs OS memory compression (`mmap_unix.go:34`) · int2 as a cold/rare-expert precision tier (`quant_q2.go:37`) · lm_head above the body quant tier / I-O-boundary precision (`quant_q4.go:245`) · quant-aware default temperature (`dynamic_precision.go:36`) · sign-trick maddubs int8 dot (`quant_amd64.s:5`) · shape-conditioned Q4_K int-dot-vs-f32 crossover (`quant_amd64_q4k.go:66`) · tokenizer `ignore_merges` whole-piece fast path (`tokenizer.go:420`, good-first) · `fak model pull hf://…#prefix` shard filter (`hfhub.go:465`) · segmented Range download + resume (`hfhub.go:346`) · DSA presence-gated activation + `FAK_DSA_FORCE_DENSE` equivalence lever (`glm_dsa_session.go:408`) · append-only data-before-count crash-safe KV checkpoint (`l3kv/store.go:160`) · exclude the self-rebuilding MTP draft window from KV snapshots (`spanserialize.go:33`) · grow-only pagedRing slab recycle (`paging_ring.go:106`) · compile-time int-width assertion for >2 GB tensors (`gguf_weightsource.go:100`) · resident RSS on the per-turn stats line (`cmd/simpledemo/stats.go:142`) · achieved-quant-compression-ratio readout (`resident_report.go:44`) · bench-admission honesty: require warmup/reps/baseline + fail-loud on missing number (`run-manifest.v1.json`, `nightrun/run.go:440`, enriches #4101) · expert-paging disk microbench in the production read shape + buffered-vs-O_DIRECT ceiling (`loadpath.go:113`, cites #2722/#2726/#1062) · community hardware-benchmark contribution table (`bench_catalog.py:402`) · first-contact "reframe the need-a-GPU assumption" (`computeauto.go:139`).

## DIVERGENT ledger (28 — earned dismissals, borrow-blocked with a stated tradeoff)

The load-bearing ones (all reconstruct colibri's single-user-owns-the-box world vs fak's audited-fleet world):
- **O(1) truncate-rollback KV file** (`c/glm.c:1888`) vs fak's **append-only tamper-evident hash chain** (`journal.go:140`, no Truncate — EU AI Act Art.12). Borrow the warm-resume *load* (#1469), not the rollback.
- **Self-describing-by-size quant container** (`c/glm.c:696`) vs fak's **explicit format discriminant** (`l3kv/store.go:38`) — colibri's is unambiguous only in a closed one-scheme world.
- **Silent per-tensor CUDA→CPU fallback** (`c/glm.c:482`) vs fak's **refuse-not-diverge** on a multi-tenant `fak serve` (`compute/cuda.go:399`).
- **Single-kernel runtime-fmt-switch GEMM** (`c/backend_cuda.cu:49`) vs fak's **per-format kernels** (`cuda.go:834`) — fak trades kernel count for zero warp-divergence + steerability.
- **Delete-source-after-convert** (`convert_fp8_to_int4.py:428`) vs fak's **keep-both** blob world (`blobfs/store.go:367`) — colibri's only pays off for a one-shot conversion on a box that can't hold both.
- **Contiguous gate/up/down single-pread coalescing** (`c/glm.c:919`) vs fak's **per-tensor-offset, any-layout-correct** reads (colibri re-lays the container; fak loads once into resident RAM and never re-reads experts per token).
- **Browser-localStorage no-backend endpoint config** (`web/src/App.tsx:53`) vs fak's **bearer-secret cross-trust gate** (`http.go:318`).
- Plus: median vs mean expert-byte cost (`estimate.go:236`); probe-vs-formula expert footprint; naive-len continuation split (fak prevents the merge-crossing hazard structurally); degenerate-oracle-hyperparameter to green a not-yet-built path.

## PRESENT ledger (70 — deep-witnessed "already have it on-axis", dropped)

Representative on-axis PRESENTs (fak code read, not assumed from the capability name):
- **MLA sparse indexer** with provable dense-equivalence below threshold — `internal/model/dsa_index.go:20,99`.
- **Per-user batch sessions** sharing one weight set — `internal/model/batch.go:127`.
- **Radix longest-token-prefix reuse** for stateless full-history POSTs — `internal/radixkv/radixkv.go:334`; suffix-only prefill `internal/agent/inkernel_planner.go:1173`.
- **Q8_0 int8 dot** + fused nibble-unpack **Q4_K** kernels — `internal/model/quant_amd64.s`, `quant_amd64_q4k.s:68`.
- **Oracle argmax-parity** exact gate + tiny regen-locally fixture — `internal/model/oracle_test.go:1552`, `export_oracle.py`.
- **Header-only offset-validated** safetensors map; **thread-safe 64-bit pread**; **single-seam platform quarantine** (`internal/flock`).
- **Honest-fence worldview** already lived: inline loss-regime disclosure (`gateway.go:200`), self-invalidating stale benchmarks (`benchcli.go:440`, #416), source-citation-required ingest (`benchcatalog/ingest.go:129`), measured-vs-projection typographic split (`benchmark_authority.py:160`).

## Honest limits

- Witness is **lexical + a 2026-07-11 snapshot** (`fak_feature_query`/`fak index` + raw grep + `gh` dedup); re-witness before building.
- **0 fully-from-scratch borrows** — every leaf is a *specific delta* on a seam fak already owns, because colibri and fak solve the same GLM-5.2 problem. Expected shape for the closest-neighbor engine, not a thin read.
- The 12 seams the leaves anchor were re-grepped and confirmed present at this SHA; the ~30 recorded/file-later seams were not each re-verified — verify before promoting.
- colibri's worldview (single-user-owns-the-box) is a reconstruction from its code/defaults/README, kept falsifiable via cited config defaults and the honest loss-regime fence it publishes; a DIVERGENT resting on a guessed motive was ablated and filed instead.
- License read is a good-faith Apache-2.0 ↔ Apache-2.0 compatibility check, not legal advice.

## Companions

- Sibling engine studies: `CONCEPT-STUDY-KTRANSFORMERS-2026-07-10.md`, `CONCEPT-STUDY-DEDICATED-MOE-CACHING-2026-07-10.md`, `CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md`, `CONCEPT-STUDY-EPLB-2026-07-10.md`, `CONCEPT-STUDY-DYNAMO-2026-07-08.md`.
- Skills: `.claude/skills/study-repo/SKILL.md` (this pass) → hands off to `.claude/skills/field-borrow/SKILL.md` (per-capability witness+file).
- Epics the leaves anchor: #4352 (this study), #2236 (memory-first superset), #3174 (expert residency M11), #2726/#2722 (native/Mac streaming), #974 (host-fit), #4102 (tool-call speculation).
