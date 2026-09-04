---
title: "Fast-Cadence OSS Stream Study Radar (study-radar): Dynamic Pacing, Non-Trunk Intelligence, and Zero-Waste Intake"
description: "Architecture, dynamic pacing, non-trunk PR/issue tracking, deduplication funnel, and safe continuous intake for the fak stream study radar."
---

# Fast-Cadence OSS Stream Study Radar (`study-radar`)

> **Executive Summary:** Existing study mechanisms (`study-monitor`, `study-forge`, `scout-loop`) operate on static, batch schedules (14-day reviews or daily crawlers). In high-velocity open-source ecosystems (vLLM, SGLang, OpenCode, PyTorch, llama.cpp, Triton), crucial architectural innovations, kernel algorithms, and critical bug discoveries emerge in **Pull Requests, issue RFCs, and non-trunk branches weeks or months before landing on main or appearing in release tags**.
>
> `study-radar` is an autonomous, event-driven stream intelligence engine that tracks known OSS repositories on a **super-fast cadence (hourly down to 15 minutes for volatile hubs, backing off to daily/weekly for quiescent research repos)**. It couples zero-token HTTP conditional polling (ETags), deterministic surface/AST delta filtering, cross-receipt deduplication, license fencing, and anti-storm intake controls to generate grounded, contract-clean GitHub issues for fak without exhausting rate limits or burning token budgets.

---

## 1. Problem Formulation & Operational Reality

### The 1-Hour vs 14-Day Blind Spot
Today, `fak study-monitor` checks tracked repositories on a default `--due-days 14` interval, while `scout-loop` runs once a day to select one lead. This cadence introduces severe latency:
1. **Missed Early-Stage PR Innovation:** The best architectural designs (e.g. KV-cache management, custom GEMM kernels, speculative decoding state machines, prefix caching) undergo weeks of iteration in open PRs. Waiting for a merge or release causes fak to fall behind the field.
2. **Ignored RFC Discussions:** Open design issues and RFCs reveal why an incumbent's architecture failed and what tradeoffs they chose before any code is committed.
3. **Wasted Effort on Stale Scans:** High-velocity repos generate thousands of events monthly; running a full, unindexed study pass every two weeks requires parsing massive corpora, whereas micro-delta indexing over the last hour is lightweight and immediate.

### The Traps of Fast Cadence
Running an hourly or sub-hourly scanner naively creates four lethal failure modes:
1. **GitHub API Rate-Limit Exhaustion:** Polling 50+ repositories every hour naively (fetching PR lists, commits, issues) requires thousands of REST calls, instantly exceeding GitHub’s 5,000 req/hr rate limit.
2. **Token Economy Collapse:** Running LLM inference over every commit, PR synchronize, or review comment burns millions of tokens on noise (typos, CI adjustments, docs, formatting).
3. **Issue Storming & Human Drowning:** If an upstream repo merges 30 PRs or receives 50 comments in an hour, generating 30 issues on fak destroys maintainer attention and pollutes the backlog.
4. **Security & Prompt Injection:** Upstream PR descriptions and issue bodies are untrusted external input; malicious or adversarial text could attempt prompt injection or jailbreak our autonomous dispatch workers.
5. **License Contamination:** Uncontrolled automated borrowing risks dragging GPL/AGPL copyleft code into fak's Apache-2.0 codebase.

---

## 2. Core Architecture: The 5-Tier Zero-Waste Funnel

To ensure high velocity at near-zero cost, `study-radar` routes every upstream event through a strict, multi-tier funnel:

```
[Upstream GitHub Repositories]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 0: Zero-Token Conditional Network Ingestion       │
│ • GitHub Events API (/repos/{owner}/{repo}/events)     │
│ • ETag caching (If-None-Match) -> 304 = 0 rate limit   │
│ • Adaptive Velocity Pacing (15m to 7d)                 │
└────────────────────────────────────────────────────────┘
           │ (Only unread, new events)
           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Deterministic Surface & Delta Filter           │
│ • Globs: *kernel*, *scheduler*, *kv*, *cache*, *mmu*   │
│ • Ignore: docs/*, .github/*, *.css, translations/*     │
│ • Compare head SHA (detect no-op rebases/typos)        │
└────────────────────────────────────────────────────────┘
           │ (Only architecturally relevant deltas)
           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 2: Cross-Receipt Semantic Deduplication           │
│ • Dedupe against local study.Store receipts            │
│ • Dedupe against existing fak issues (Source borrow)   │
│ • Hash semantic mechanism identity                     │
└────────────────────────────────────────────────────────┘
           │ (Only novel, unstudied candidate deltas)
           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 3: Micro-Triage (Fast Model / Local Kernel)       │
│ • Structured JSON input schema                         │
│ • P1-P4 relevance score + novelty check (0..10)        │
│ • Threshold gate: Score >= 7.0                         │
└────────────────────────────────────────────────────────┘
           │ (Only high-value, actionable candidates)
           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 4: Grounded Issue Synthesis & License Fence       │
│ • Pin upstream path:line@sha                           │
│ • Witness fak seam in internal/                        │
│ • License gate (Apache/MIT vs AGPL concept-only)       │
│ • Anti-storm throttle (max 2/repo/day, max 5/day fleet)│
│ • Output: Contract-clean GitHub issue on fak           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Mechanisms

### A. Dynamic Adaptive Cadence (Velocity Pacing Engine)
Repositories exhibit vastly different activity levels:
- **Hotspot Hubs (e.g., `vllm-project/vllm`, `anomalyco/opencode`, `sgl-project/sglang`):** 5-20 events per hour.
- **Active Projects (e.g., `OpenRouterTeam/go-sdk`, `ggerganov/llama.cpp`):** 1-5 events per day.
- **Quiescent Research Artifacts (e.g., `microsoft/vidur`, niche hardware kernels):** 0-1 events per month.

Instead of a fixed hourly cron, the radar maintains a per-repository dynamic interval $\tau_i \in [\tau_{\min}, \tau_{\max}]$:
- Bounds: $\tau_{\min} = 15\text{ minutes}$, $\tau_{\max} = 7\text{ days}$.
- **Acceleration Rule:** When a poll returns new relevant events (HTTP 200 with actionable deltas), the interval cuts in half:
  $$\tau_i \leftarrow \max\left(\tau_{\min}, \frac{\tau_i}{2}\right)$$
- **Decay / Backoff Rule:** When a poll returns HTTP 304 (Not Modified) or 0 relevant events, the interval increases with exponential backoff and jitter:
  $$\tau_i \leftarrow \min\left(\tau_{\max}, \tau_i \times 1.5 + \text{jitter}\right)$$
- **State Persistence:** Stored in `docs/research/monitored-radar-state.json` or `.fak/radar/state.json`.

### B. Tier 0: Zero-Token Conditional Ingestion (ETags & Events)
1. **GitHub Events API (`/repos/{owner}/{repo}/events`):**
   - Emits a unified stream of the 300 most recent events across commits, PRs, issues, comments, and releases.
   - Polling passes send `If-None-Match: "<ETag>"`.
   - **Key Invariant:** GitHub does **not** count HTTP 304 responses against the primary authenticated REST rate limit (5,000 req/hr). 50 quiescent repos polled every 15 minutes consume 0 rate limit quota.
2. **Head SHA & Ref Tracking:**
   - PR events track `headRefOid` (the immutable commit hash). If a PR is re-synchronized with the same commit SHA, it is instantly discarded.
3. **Public Atom Feeds (Fallback):**
   - For non-authenticated or secondary monitoring, `https://github.com/{owner}/{repo}/commits.atom` and `releases.atom` provide rate-limit-free change notifications.

### C. Non-Trunk Intelligence: PRs, Issues, and Reviews
Trunk (`main`) is only the final landing strip. The radar actively extracts signal from:
1. **PullRequestEvent (`action: opened | synchronize | ready_for_review`):**
   - Inspects candidate PRs proposing new algorithms, kernels, or memory layouts.
   - Evaluates review decisions (`APPROVED` or review comments highlighting novel technical insights).
2. **IssuesEvent (`action: opened | labeled`):**
   - Filters for issue labels matching: `RFC`, `design`, `architecture`, `performance`, `kernel`, `bug:correctness`.
   - Ignores user support, questions, setup help, or documentation issues.
3. **ReleaseEvent:**
   - Tracks version tags, changelogs, benchmark claims, and breaking ABI modifications.

### D. Tier 1: Deterministic Relevance & Surface Filters (Zero LLM Tokens)
Before spending a single model token, raw event deltas are filtered using deterministic pattern matching:
- **Filepath Globs:**
  - Positive match required: `*kernel*`, `*scheduler*`, `*kv*`, `*cache*`, `*prefix*`, `*quant*`, `*mmu*`, `*route*`, `*dispatch*`, `*engine*`, `*serving*`, `*memory*`, `*attention*`, `*gemm*`, `*cuda*`, `*metal*`, `*rocm*`, `*.cu`, `*.metal`, `*.hip`.
  - Negative exclusions: `docs/**`, `*.md`, `.github/**`, `tests/e2e/ui/**`, `*.css`, `translations/**`, `assets/**`.
- **Commit Message & PR Title Classifier:**
  - Excludes PRs matching: `chore:`, `docs:`, `ci:`, `bump version`, `fix typo`, `lint`, `format`.
  - Advances PRs matching: `feat:`, `perf:`, `fix(kernel):`, `refactor(scheduler):`, `optimize`.

### E. Tier 2: Cross-Corpus Deduplication
To prevent studying the same PR across multiple turns:
- **Identity Triplet:** Keyed by `(repository, entity_type, entity_id, head_sha)`.
- **Local Receipt Store:** Queries `internal/study` (`fak study search`) for past receipts.
- **GitHub Issue Index:** Queries existing repo issues using `gh issue list --search "Source borrow: <repo>#<id>"` to verify whether this PR or issue was already addressed or dismissed.
- If an exact head SHA or identical borrow was already filed, the event is marked `STUDY_DEDUP_SKIPPED` and archived.

### F. Tier 3: Micro-Triage via Compact Model / Local Kernel
Events clearing Tiers 0-2 are evaluated by a fast, low-cost model (e.g. Qwen-3.8-Flash or local embedding classifier):
- **Input:** JSON payload with PR/issue title, sanitized body summary, list of changed files, and top-diff excerpt.
- **Scoring Dimensions (0..3 each, total 0..12):**
  1. *Fak Relevance:* Does it solve P1 (context), P2 (efficiency), P3 (adaptation), or P4 (operations)?
  2. *Novelty:* Is this a novel mechanism or standard boilerplate?
  3. *Actionability:* Can fak implement a clean-room equivalent?
  4. *Evidence:* Are there benchmarks, tests, or reproducible profiles?
- **Decision:**
  - Score $\ge 8$: Advance to Tier 4 (Deep Synthesis & Issue Creation).
  - Score $5-7$: Log as `WATCH` in radar telemetry (re-evaluate if PR merges or gains approvals).
  - Score $< 5$: Discard as `REJECT`.

### G. Tier 4: Grounded Issue Synthesis & Licensing
When an event passes Tier 3, the engine crafts a contract-clean GitHub issue following the strict `study-repo` / `field-borrow` format:
- **Upstream Citation:** Pinned `path:line@sha` in the upstream repository.
- **Fak Seam Witness:** Exact target file and line in `internal/` where the borrow lands.
- **Tradeoff Ablation:** Reconstruct the upstream author's worldview without ego; contrast against fak's architectural invariants.
- **License Boundary Check:**
  - Permissive (MIT, Apache-2.0, BSD): Clean-room adaptation allowed.
  - Copyleft (GPL, AGPL, CC-BY-NC): STRICTLY CONCEPT-ONLY. No code copying or translation; clean-room implementation from specification.
- **First Checkable Step:** Concrete unit test or benchmark to verify the borrow.

---

## 4. Hard Safeguards & Circuit Breakers

1. **Anti-Storm / Backpressure Rate Limits:**
   - **Per-Repo Cap:** Maximum 2 issues filed per upstream repository per 24-hour window.
   - **Fleet-Wide Cap:** Maximum 5 study issues filed per day across all monitored repos.
   - **Batch Clustering:** If an upstream repo merges a major 20-PR wave, the radar clusters them into a single "Spine Radar Review" epic rather than filing 20 fragmented tickets.
2. **Untrusted Content Containment (Prompt Injection Shield):**
   - External PR and issue text is treated as untrusted user input.
   - Bodies are wrapped in strict structural XML/JSON boundaries with instructions forbidding role switches or system-prompt modifications.
   - Markdown links, images, and raw HTML are stripped of executable/script payloads.
3. **DOS Lease & Worktree Isolation:**
   - The radar runs under a dedicated DOS lane (`study-radar`).
   - Before writing or updating local files, it adjudicates via `dos_arbitrate` to ensure it never collides with active developer sessions or super-loop worker trees.
4. **Offline & Dry-Run Guarantee:**
   - Default mode is `--dry-run`: indexes events, scores candidates, and prints JSON plans without creating GitHub issues.
   - Issue creation requires `--live` or explicit service configuration.

---

## 5. Integration into fak Services and Super-Loop

### Service Daemon (`fak service`)
- Runs as a systemd unit (Linux) or launchd daemon (macOS) via `fak service`:
  ```bash
  fak-dev study-radar daemon --interval 15m --state-file docs/research/monitored-radar-state.json
  ```
- Uses `internal/servicewatchdog` and `internal/serviceledger` for crash recovery, restart telemetry, and heartbeat monitoring.
- On Windows devboxes, registered via `tools/register_study_radar.ps1` in Windows Task Scheduler.

### Super-Loop Backlog Feeder
- Issues created by `study-radar` carry standard labels: `research`, `class:dev`, `priority/P1` or `priority/P2`, and `(fak <leaf>)`.
- The `/super-loop` and `/dos-dispatch` engines automatically discover these tickets during candidate audits (`dos-next-up`), pricing and dispatching them to headless workers as capacity permits.

---

## 6. Implementation Milestones

1. **Phase 1 (Ingestion Spine):** Implement `internal/studyradar/collector.go` with GitHub Events API, ETag conditional GETs, and velocity-based adaptive pacing.
2. **Phase 2 (Delta Filter & Deduplication):** Build deterministic file/title regex filters and local receipt/issue dedup checking.
3. **Phase 3 (Synthesis & Safety):** Implement structured micro-triage, license verification, anti-storm throttles, and contract-clean issue creation.
4. **Phase 4 (Daemon & Super-Loop Integration):** Deliver `fak-dev study-radar` CLI verb, `fak service` definitions, and automated backlog feeding.

---

## 7. Monitored OSS Performance Portfolio (23 Repositories)

The stream study radar actively monitors and indexes the following portfolio of 23 high-velocity open-source repositories tracking frontier hybrid architectures (**Qwen 3.8 Flash / Flash-Next 125B–180B MoE**, **Qwen 3.8 27B**, and **GLM 5.3 Flash 300B–320B MoE**). Individual deep concept studies are linked below:

| # | Repository | Target Model | Hardware Platform | Serving Engine | Precision / Quant | Key Mechanism / Breakthrough | Dedicated Concept Study |
|---|---|---|---|---|---|---|---|
| 1 | **`hasso5703/dgx-spark-qwen38`** | Qwen3.8-27B / Flash-Next | 1× DGX Spark (GB10) | SGLang (patched) | NVFP4 + FP8 KV | NVMe mmap for 51B PLE; kills FlashInfer autotune lottery; keepalive proxy | [`CONCEPT-STUDY-HASSO5703-DGX-SPARK-2026-09-03.md`](../notes/CONCEPT-STUDY-HASSO5703-DGX-SPARK-2026-09-03.md) ([#10952](https://github.com/anthony-chaudhary/fak/issues/10952)) |
| 2 | **`adrienbrault/qwen3.8-27b-rtx5090`** | Qwen3.8-27B | 2× RTX 5090 (GDDR7) | vLLM v0.28.0 | NVFP4 + FP8 KV | Linear V-scale store overlay on sm120; XQA decode; 200 GB direct-I/O disk KV tier | [`CONCEPT-STUDY-ADRIENBRAULT-RTX5090-2026-09-03.md`](../notes/CONCEPT-STUDY-ADRIENBRAULT-RTX5090-2026-09-03.md) ([#10951](https://github.com/anthony-chaudhary/fak/issues/10951)) |
| 3 | **`albond/SingleSpark-Qwen3.8-Flash-Next`** | Qwen3.8-Flash-Next | 1× DGX Spark (GB10) | vLLM (dev build) | NVFP4 / Block-FP8 | Hardware instruction profiling: 6.50% issue slot utilization roofline bound | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 4 | **`airawatraj/dgx-spark-qwen38-flash-agent`** | Qwen3.8-Flash-Next | 1× DGX Spark (GB10) | SGLang (patched) | NVFP4 + HashK PLE | HashK 4× PLE compression (12.8 GB VRAM); zero SSD I/O; Mamba DeltaNet state rollback | [`CONCEPT-STUDY-AIRAWATRAJ-HASHK-PLE-2026-09-03.md`](../notes/CONCEPT-STUDY-AIRAWATRAJ-HASHK-PLE-2026-09-03.md) ([#10954](https://github.com/anthony-chaudhary/fak/issues/10954)) |
| 5 | **`cglab-public/dgx-spark-flashnext`** | Qwen3.8-Flash-Next | 2× DGX Spark (TP2) | SGLang (PR #36497) | NVFP4 + BF16 KV | Forensics of token-0 (`!`) collapse; mapped QSA indexer transient memory spikes | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 6 | **`MindLab-Research/ferrite`** | GLM-5.3-Flash | 1× B300 (sm_100a) | Ferrite (Rust native) | BF16 / FP8 / F32 | Static PDAF disaggregation; exact MHC with Sinkhorn normalization; WYF parallel chunking | [`CONCEPT-STUDY-FERRITE-GLM53-2026-09-03.md`](../notes/CONCEPT-STUDY-FERRITE-GLM53-2026-09-03.md) ([#10950](https://github.com/anthony-chaudhary/fak/issues/10950)) |
| 7 | **`gitcommit90/glm-5.3-one-spark`** | GLM-5.3-Flash | 1× DGX Spark (GB10) | vLLM + ExLlamaV3 | EXL3 2.05 bpw | 76% cold boot latency cut (14m to 3m21s) via DNS loopback & anonymous mmap staging | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 8 | **`vcruz305/GLM-5.3-Flash-EXL3-K2`** | GLM-5.3-Flash | 1× DGX Spark (GB10) | vLLM (v0.3.1 native) | EXL3 K2/K4 | Fixed K-pool tail slot mapping OOB bug; super fat GEMM scatter; cooperative MoE decode | [`CONCEPT-STUDY-VCRUZ305-GLM53-EXL3-2026-09-03.md`](../notes/CONCEPT-STUDY-VCRUZ305-GLM53-EXL3-2026-09-03.md) ([#10953](https://github.com/anthony-chaudhary/fak/issues/10953)) |
| 9 | **`marksunner/dgx-spark-glm52`** | GLM-5.2 / GLM-5.3 | 4× DGX Spark (QSFP) | vLLM + Triton | QuantTrio Int4/Int8 | Resolved FlashInfer mbarrier livelock via Triton sparse MLA; complete RoCE fabric blueprint | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 10 | **`punkjazz-labs/glm-5.3-flash-exl3-4x`** | GLM-5.3-Flash | 4× DGX Spark (TP4) | vLLM + ExLlamaV3 | EXL3 4 bpw | Parameter sweep; fixed 330s prefill starvation by turning off mixed chunking | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 11 | **`alexellis/glm-5.3-flash-4x-switchless`** | GLM-5.3-Flash | 4× DGX Spark (Ring) | vLLM + Marlin MoE | NVFP4 + BF16 KV | Switchless RoCE ring with patched NCCL `skip-tree-connect`; 58:1 prompt-to-completion ratio | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 12 | **`Dyluhn/R9V`** | Qwen3.8-Flash-Next / Muse | Dual AMD R9700 (64GB) | Adapted vLLM + HIP | UD-IQ4_XS + block-FP8 | Asymmetric PCIe Gen5 x16 + Gen4 x4 handling; graph-safe LRU16 expert cache | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 13 | **`Gr33n93/llama.cpp-qwen3.8-mtp`** | Qwen3.8-Flash-Next | AMD Strix Halo (8060S) | llama.cpp (Vulkan/RADV) | UD-IQ4_XS + Q8 MTP | Decoupled draft ubatch (`--spec-draft-ubatch-size 512`) eliminating compute ring timeouts | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 14 | **`PieBru/Qwen-3.8-27B_Strix-Halo`** | Qwen3.8-27B | AMD Strix Halo (8060S) | strix-halo-llamacpp | UD-Q5/Q6/Q8_K_XL | Forensics proving 137k crash is AMDGPU watchdog (`amdgpu.lockup_timeout=-1`) | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 15 | **`davidcanar/vllm-strix-halo`** | GLM-5.3-Flash / DS-V4 | 2× AMD Strix Halo (TP2) | vLLM + Ray | AWQ W4A16 + FP8 KV | Thunderbolt-4 / USB4 RoCE-RDMA transport with `tbv` modules and ~105 µs all-reduce hook | [`CONCEPT-STUDY-DAVIDCANAR-STRIX-HALO-ROCE-2026-09-03.md`](../notes/CONCEPT-STUDY-DAVIDCANAR-STRIX-HALO-ROCE-2026-09-03.md) ([#10955](https://github.com/anthony-chaudhary/fak/issues/10955)) |
| 16 | **`carloslfu/slotstream`** | Qwen3.8-Flash-Next | Apple M5 Pro (48 GB) | Swift 6 + MLX / Metal | 4-bit (group 64) | Streams routed experts from SSD via QD32 `pread` (17.3 GB/s); 9-tensor slot decomposition | [`CONCEPT-STUDY-CARLOSLFU-SLOTSTREAM-2026-09-03.md`](../notes/CONCEPT-STUDY-CARLOSLFU-SLOTSTREAM-2026-09-03.md) ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 17 | **`kiojuvr/glm53-flash-mlx`** | GLM-5.3-Flash | Apple M3 Ultra (512GB) | Python + MLX | Native FP8 E4M3 | Compact NoPE DSA cache cuts state memory 86%; 100k continuous soak without drift | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 18 | **`Azhu9701/ninfer-4090d`** | Qwen3.8-27B | 1× RTX 4090 D (48 GB) | NInfer standalone | 16.67 GiB INT + E8 KV | 114-SM wave grid alignment; dynamic draft controller; DirectStorage 1.3 DMA cache | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 19 | **`halt95/qwen38-flash-next-3090s`** | Qwen3.8-Flash-Next | 4× RTX 3090 (96 GB) | vLLM 0.28.0 (TP4+EP) | W4A16 + FP8 KV | Full CUDA graphs amortize 13ms Python host loop; in-place tensor repacking recovers 1.3 GiB | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 20 | **`shyringo/qwen3.8-flash-next-in-c`** | Qwen3.8-Flash-Next | Intel Core i5 Laptop | Standalone Native C | Unsloth UD-IQ1_S GGUF | 8.99 GiB peak RSS; AVX2 `vpshufb` vector lookups; on-demand 51B PLE gather; zero Python/CUDA | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 21 | **`thadreber-web/llama.cpp-qwen38`** | Qwen3.8-Flash-Next | 1× NVIDIA GB10 (128 GB) | llama.cpp fork | IQ3_M + turbo3 KV | Shape-aware CUDA graph cache keying stops eviction thrash; grouped RMSNorm boosts MTP | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 22 | **`feifeidu-max/Qwen3.8-FlashNext`** | Qwen3.8-FlashNext | 2× Quadro RTX 8000 (96GB) | ik_llama.cpp | Unsloth UD-IQ4_XS | Hoisted 12 PLE CPU round trips to 1 batch gather (eliminates 113ms floor); wired orphaned kernels | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |
| 23 | **`pctablet505/glm53-flash-single-gpu`** | GLM-5.3-Flash | 1× RTX PRO 6000 (96 GB) | vLLM PR #53906 + Marlin | NVFP4 (181.3 GiB model)| Serves 181 GB model on 96 GB card via 52 GB/s Triton gather, 54 hot slots, and grouped DMA | Portfolio Index ([#10960](https://github.com/anthony-chaudhary/fak/issues/10960)) |

### Cross-Architecture Synthesis

For systematic translation vectors and cross-architecture blueprints across Apple Silicon (MLX/Metal), AMD (Strix Halo/RDNA4/Vulkan/ROCm), and NVIDIA (Blackwell sm120/GB10 sm121), see:
- [`docs/research/CROSS-ARCHITECTURE-INNOVATION-MATRIX.md`](CROSS-ARCHITECTURE-INNOVATION-MATRIX.md)
- [`docs/research/qwen38_glm53_deep_subagent_inventory.md`](qwen38_glm53_deep_subagent_inventory.md)
