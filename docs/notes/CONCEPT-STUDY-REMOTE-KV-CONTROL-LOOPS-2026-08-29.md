# Remote-KV reference v1.5.7: borrow the control loops around remote KV, not the cache server

> Studied 2026-08-29. Source: `a private remote-KV reference`, pinned at `179c3db7696a77aa94e9e6441b73f6778f82763f` (`v1.5.7`, committed 2026-05-28). Tracker: [#10261](https://github.com/anthony-chaudhary/fak/issues/10261). Durable receipt: `study_14e714453b97715392e456c478f7b575dabf855901af6cd10e98607b5686264e`.

## Verdict

The remote-cache reference's useful lesson for FAK is not "run this external cache." It is that a fast optional KV tier needs explicit control loops around every optimistic operation: readiness must mean data is usable, prewarm state must be claimable and configuration-bound, a mixed batch must preserve per-item truth, remote faults must stop taxing the request path, and dedup checks must continuously justify their own round trip.

FAK already owns stronger native foundations on several adjacent axes. It has value/frequency-aware pressure admission, typed OK/MISS/FAULT residency receipts, integrity- and scope-checked remote snapshots, partial restore accounting, cache-aware routing, model warmup readiness, and active topology/transport work. Five narrow gaps survived source comparison and issue deduplication:

- [#10273](https://github.com/anthony-chaudhary/fak/issues/10273): bounded open/half-open recovery for an unhealthy optional remote L3.
- [#10274](https://github.com/anthony-chaudhary/fak/issues/10274): self-disable remote existence checks when their measured cost exceeds avoided transfer.
- [#10275](https://github.com/anthony-chaudhary/fak/issues/10275): make successful prewarm state single-claim, expiring, and bound to the complete effective configuration.
- [#10276](https://github.com/anthony-chaudhary/fak/issues/10276): add an optional ordered KV batch receipt without weakening the per-span ABI.
- [#10277](https://github.com/anthony-chaudhary/fak/issues/10277): resolve and probe an optional remote KV backend before heavyweight model initialization.

All five are clean-room, **inspire-only** effects. The pinned repository has no root license for its reference-specific code. Vendored SGLang carries Apache-2.0 material and the vLLM package declares Apache-2.0, but neither grants a license over the surrounding custom remote-cache implementation.

## Value frame

- **For:** operators using FAK-owned native inference with an optional remote/shared KV performance tier.
- **Problem:** the transport can be correct while its surrounding readiness, failure, identity, and economics state remains optimistic or stale.
- **Today:** FAK's per-span transfer truth is strong, but several multi-operation and startup boundaries are still implicit.
- **Better because:** small typed control loops keep optional-cache overhead net-positive and preserve recomputation as the safe fallback.
- **Witness:** each surviving issue names one fail-before/pass-after test at an existing FAK seam; no issue claims the reference system's benchmark results as FAK performance.

Centrality is **Core** for claimable prewarm and per-item batch truth, and **Enabling** for breaker, dedup economics, and startup preflight. P1 preserves content and credential boundaries; P2 includes checks, faults, retries, and initialization in net cost; P3 makes every feedback loop bounded and reversible; P4 exposes configuration identity, phase, freshness, and fallback reason.

## What the remote-cache reference optimizes

The remote-cache reference is a disaggregated KV-cache system aimed at read-heavy shared-prefix serving. Its assumed production envelope is SGLang or an experimental vLLM connector, a separate large-DRAM cache server, NVIDIA GPU hosts, ConnectX-class networking, TCP for development, RDMA for production, and CXL as an experimental tier. The architecture is availability-first: the serving engine remains authoritative, cache failures become misses and recomputation, and no optional-cache error should make inference unavailable.

The design optimizes time-to-first-token for repeated prefixes and multi-consumer reuse, not universal model execution. Its policy documentation treats a few microseconds of metadata work as acceptable when it avoids a much larger KV transfer. That makes its feedback mechanisms workload-specific: a useful check on a warm, high-hit workload can become pure overhead on a cold or high-latency one.

FAK should keep this worldview only at the optional performance-cache boundary. FAK-native execution still owns kernels, memory, scheduling, cache, adaptation, and operations. No external remote-cache system, SGLang, vLLM, or llama.cpp fallback is proposed.

## Source state and evidence denominator

The complete committed tree at the pin contains **3,668 files**, **573 directories**, **41,771,717 bytes**, and **988,104 text lines** across 12 inventory groups. The largest group is a vendored SGLang snapshot with 3,313 files; reference-specific load-bearing areas are the Go server, Python client, SGLang connector/patch layer, experimental vLLM connector, deployment/configuration, dashboards, docs, and root installer/history.

The local checkout also contained untracked binaries, `.orig` backups, and a `VERSION` file. They were excluded from every shipped-source claim. The study used a committed-source archive at the exact pin.

Forge census through `2026-08-30T01:37:23Z` found:

| Surface | Complete census result |
|---|---:|
| Issues | 0 |
| Pull requests | 0 |
| Discussions | unavailable/disabled |
| Releases | 103 |
| Labels | 9 |
| Milestones | 0 |

The repository's selected history does contain useful implementation transitions: `9a26d31...` introduced early connection/readiness progress, `26c783...` pinned and manifested the SGLang patch stack, `a355fbe...` added restore and vLLM work, and `23717b...` developed the installer. The latest hosted Actions run at study time executed zero steps because of an account billing refusal. That is neither a green nor red code witness.

Machine companions: the source inventory and forge census remain outside the public tree; this note preserves the bounded technical findings and reproducibility pin.

## Load-bearing mechanisms

### Server ownership and pressure

The Go server serializes shard allocation per NUMA owner while allowing different NUMA owners to progress concurrently (`remote-cache-server/internal/cache/shard_manager.go:393-480@179c3db7696a77aa94e9e6441b73f6778f82763f`). Its batch handler gives the cross-shard operation one caller deadline rather than multiplying timeout by shard count (`remote-cache-server/internal/cache/handle_batch.go:15-60@179c3db7696a77aa94e9e6441b73f6778f82763f`).

Under pressure, allocation proceeds through best fit, promotion, TTL expiration, eviction, and finally OOM; a probabilistic gate rejects low-value writes without disabling reads (`remote-cache-server/internal/cache/shard_ops.go:82-180,215-274@179c3db7696a77aa94e9e6441b73f6778f82763f`). FAK already has the stronger on-axis default in `internal/radixkv/admission.go`: bounded frequency evidence and compute-value comparisons protect hot victims, reject losing writes, and let repeated demand recover a previously rejected candidate.

The remote-cache reference can retune its allocator from observed hugepage headroom and migrate a bounded amount of live state (`shard_ops.go:1010-1225`). That shape does not transfer directly to FAK's owned native memory layers, and the reference's restore path frees an existing entry before successor allocation at `shard_ops.go:1669-1679`; an allocation failure can therefore destroy a live value. The study excludes this as a correctness pattern.

### Transport truth and degradation

The RDMA MGET path returns coordinates per shard and leases them for five seconds, while the client performs one-sided reads and acknowledges completion (`remote-cache-server/internal/transport/rdma_cgo.go:1562-1676`; `remote-cache-client/remote_cache_client/_batch_ops.py:74-184`). A shard-level failure can degrade independently instead of turning the whole request into an invented success.

The SGLang connector has the strongest result fold: both K and V must succeed, per-page outcomes survive, deduplicated writes merge exact positions, and timeouts become misses while fault counters remain visible (`remote-cache-sglang-connector/remote_cache_storage.py:1428-1458,1545-1609,1652-1696`). FAK already represents a single span truthfully in `internal/abi/kvbackend.go:46-69` and accounts a warm resume in exhaustive restored/missed/faulted buckets in `internal/l3kv/warmresume.go:113-162`. The missing axis is an additive ordered batch contract, filed as #10276.

The vLLM connector is negative evidence for why this matters. It can mark all requested hashes present after individual load failures (`remote-cache-vllm-connector/remote_cache_vllm_connector/worker.py:197-216`), ignores MSET results (`:274`), and declares an error-block set that is never populated (`:73`, read at `:302`). Its scheduler deletes an LRU negative but leaves the Bloom hint stale, and the unit test permits that stale match (`scheduler.py:249-269`; `tests/unit/test_scheduler.py:76-108`). FAK must keep authoritative outcomes above approximate membership hints; existing #4303 already owns verified pulls behind a probabilistic directory.

### Readiness, prewarm, and identity

The remote-cache reference's warmup waits for the first real cache pages and can restart progress after reconnection (`remote-cache-client/remote_cache_client/warmup.py:90-167,289-357`). FAK's `internal/gateway/readiness_warmup.go` already arms before listener readiness and distinguishes `warmup_pending` from ready after a real model warmup. External-cache data readiness is adjacent, but #8532 already owns live external/shared-KV state receipts, so this study does not create a duplicate readiness issue.

The remote-cache reference's prewarm registry publishes completed warm work, bounds it by TTL, and lets one matching consumer claim it (`remote-cache-client/remote_cache_client/prewarm.py:108-223,429-537`). The effective-configuration fingerprint at `:252-257` is incomplete: it omits NIC striping, password presence, and page geometry. FAK's `internal/compute/prewarm_admit.go:144-242` already makes the earlier security/economics decision, including secret-class default-deny and whether a warm will land hot. #10275 adds only the missing producer/consumer artifact and requires a complete behavior-affecting digest without raw secret bytes.

Topology-aware pool replacement closes old registered memory before opening the new pool, avoiding a transient double-pinning event (`remote-cache-sglang-connector/remote_cache_storage.py:528-576,1057-1083`; `remote-cache-client/remote_cache_client/_pool.py:449-679`). That becomes an acceptance invariant for existing FAK transport work #3199/#9918 rather than a new owner.

### Self-revising economics

The remote-cache reference skips dedup existence checks while cold or explicitly disabled, observes their real latency and hit rate, disables them after a bounded losing streak or when check cost exceeds avoided-transfer savings, and periodically probes for recovery (`remote-cache-sglang-connector/remote_cache_storage.py:1740-1868`). This is the most generally useful mechanism in the repository: an optimization must pay rent continuously, not only in the benchmark that justified its initial default. FAK's cache admission prices residency, not a remote lookup before a write. #10274 isolates the pure controller from any future store protocol.

The optional vLLM connector also includes a closed/open/half-open circuit breaker (`remote-cache-vllm-connector/remote_cache_vllm_connector/circuit_breaker.py:1-17,36-125`). FAK's remote snapshot path counts each fault at `internal/radixkv/remote_l3.go:126-176` but calls the backend again on the next restore. #10273 retains the availability-first effect: an open optional cache returns a typed miss/recompute without new I/O, then admits exactly one bounded recovery probe.

### Startup, deployment, and maintenance

The remote-cache reference's SGLang patch stack has a machine-readable manifest with an exact upstream base, ordered patches, changed paths, and checksums (`remote-cache-sglang-connector/patch_manifest.json:1-54`; `upgrade-sglang.py:106-190`). FAK should adopt this recipe only if it begins carrying an ordered source fork. Today its owned Go seams and explicit engine adapters are the better fit.

The connector performs a backend preflight before model load (`remote-cache-sglang-connector/remote_cache_sglang_connector/preflight.py:38-104`; patched `sglang/srt/server_args.py:1934-1955`). Its invalid JSON path silently falls back, which FAK must not copy. #10277 requires a typed configuration refusal, explicit required-versus-optional policy, bounded probe, sanitized receipt, and proof that the model loader was not called first.

The systemd unit waits on `/ready`, carries long startup bounds, restarts on failure, raises memlock/no-file limits, and disables OOM scoring (`deploy/systemd/remote-cache-server.service:22-51`). It also runs as root with `NoNewPrivileges=false`; those security choices are not adopted. The installer records installed components and provenance (`install.py:1-56,703-769`) but some fast paths accept binary existence without verifying identity (`:380-442`). FAK's version/receipt doctrine remains stronger.

## Candidate matrix

`fak capabilities` returned only broad cache, routing, and receipt cards for the candidate-specific queries; those cards were treated as navigation, not proof. `fak dev index`, raw source inspection, and open/closed GitHub searches supplied the on-axis decisions.

| Effect | FAK status | Disposition | Decision |
|---|---|---|---|
| First-real-data external-cache readiness after reconnect | **PARTIAL** | **WATCH** | Keep #8532 as owner; FAK model readiness already exists. |
| Reject low-value writes under pressure while preserving reads | **PRESENT** | **DEFAULT** | Keep `radixkv` admission and shipped #5290. |
| Per-NUMA owner serialization with cross-owner concurrency and one deadline | **PARTIAL** | **WATCH** | Existing #3199/#9913/#9920 own the transferable pieces. |
| One-sided coordinates plus bounded read leases | **PARTIAL** | **WATCH** | Existing #3199/#3310/#3316/#9919 own native transport and topology work. |
| Optional remote-cache open/half-open recovery | **ABSENT** | **OPTIONAL-MODULE** | File #10273. |
| Topology-scoped keys and compatibility remap | **PARTIAL** | **WATCH** | Existing #5273/#9919/#9929 own the seams. |
| Ordered upstream patch manifest | **ABSENT** | **RECIPE** | Apply only if FAK carries an upstream source fork. |
| Installer receipt validates reused binary identity | **PARTIAL** | **WATCH** | Preserve FAK's version/receipt doctrine; no external remote-cache installer belongs in kernel. |
| Adaptive dedup-check ROI with recovery probes | **ABSENT** | **OPTIONAL-MODULE** | File #10274. |
| Single-claim, TTL/config-bound warm artifact | **PARTIAL** | **OPTIONAL-MODULE** | File #10275. |
| Close-before-replace registered-memory pool | **PARTIAL** | **WATCH** | Acceptance invariant for #3199/#9918, not a duplicate issue. |
| Ordered per-item KV batch outcomes | **PARTIAL** | **OPTIONAL-MODULE** | File #10276. |
| Phase/freshness-bound cache telemetry | **PARTIAL** | **WATCH** | Existing #8532/#9951 own the relevant receipts/events. |
| Remote-cache preflight before model load | **ABSENT** | **OPTIONAL-MODULE** | File #10277. |
| Live hugepage allocator retune and migration | **PARTIAL** | **EXCLUDE** | Product mismatch plus a non-atomic restore failure in the reference. |
| Authoritative negative result overrides approximate membership | **PARTIAL** | **WATCH** | Keep as an invariant for verified-pull issue #4303. |

## Negative knowledge and drift

- Documentation says some inline fallback is whole-batch while the implementation degrades per shard; do not cite the prose as behavior.
- `DOCUMENTATION.md` describes a migration wait while current code is nonblocking.
- Server `VERSION` says 1.5.4 while code/changelog identify 1.5.7.
- The root changelog and README patch counts lag the actual tree.
- The bundled SGLang snapshot reports v0.5.7 while client compatibility metadata requires at least v0.5.9.
- Several advertised configuration knobs are parsed but not load-bearing; the changelog explicitly leaves some sweeper/per-layer behaviors unimplemented.
- Batch RDMA can return coordinates even when lease-enqueue fails; replication is best effort and may drop work under queue pressure.
- Compression alternatives and contiguity coalescing are proposal material, not shipped behavior.
- Installer `auto` chooses RDMA or TCP from the host. That is acceptable convenience for an external cache installer but forbidden as a silent engine substitution in FAK-native performance evidence.
- No committed test closes restore-allocation failure atomicity, replication under failure, or a production vLLM end-to-end truth path.

## License and provenance

There is no root `LICENSE`, `COPYING`, or equivalent grant for the reference-specific repository at the pin. The vendored `sglang/` tree includes Apache-2.0 license and notice material. The vLLM connector package metadata declares Apache-2.0, but a declaration in one package is not assumed to cover every custom file or the repository as a whole.

No source was copied into FAK. The five issues describe effects, invariants, and witnesses in new Go terms. A future implementation must retain the pinned attribution and perform an exact-file license review if it proposes direct reuse.

## Completeness critic and refresh boundary

Covered: every committed path through a full-tree inventory; all reference-specific server, client, connector, deployment, dashboard, docs, plans, installer, tests, and history seams; targeted vendored SGLang integration points; all six forge census surfaces available for the repository; tags/releases and selected mechanism commits; FAK capability/index/raw-code query; open/closed issue deduplication; license/provenance; and independent source-anchor reads across three disjoint subsystem packets.

The vendored SGLang implementation was not re-studied as a general serving engine because only the reference's patch and call sites bear on this decision. Generated XHTML, binary assets, local untracked files, external hardware artifacts, raw private logs, and private infrastructure details were excluded. There were no issue or PR bodies to traverse, discussions were disabled, and every one of 103 release payloads was counted but not manually re-read. Dependency-by-dependency license validation and live hardware reproduction remain outside this concept study.

Refresh on a new reference release or source pin, a newly declared root license, implementation of the experimental vLLM paths, a change to FAK's remote snapshot/batch/prewarm seams, or closure of #10273-#10277. This note does not claim that the reference's performance reproduces in FAK, that the reference executed inside FAK, or that any filed follow-up is implemented.

Companions: [field-borrow landscape](CONCEPT-FIELD-BORROW-LANDSCAPE-2026-07-08.md), earlier coarse [remote-cache re-imagining note](L3-DISAGGREGATED-CACHE-REIMAGINED.md), study tracker [#10261](https://github.com/anthony-chaudhary/fak/issues/10261), and serving epic [#50](https://github.com/anthony-chaudhary/fak/issues/50).
