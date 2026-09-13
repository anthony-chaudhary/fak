# Concept Study: Azhu9701/ninfer-4090d  -  Windows RTX 4090 D Qwen3.8 Serving Dossier, Restart-Based Draft-Depth Control, and Tool-Call-Drift Salvage

**Source:** https://github.com/Azhu9701/ninfer-4090d  
**Pinned Revision:** `3eaa163cdac422da64fed5d42d17c31174a62206`  
**Study Date:** 2026-09-03  
**License:** NONE  -  the repository ships no `LICENSE`, `COPYING`, `NOTICE`, or `AUTHORS` file at `3eaa163`; the GitHub API `license` field is `null`. All content is all-rights-reserved, which makes every candidate here **INSPIRE-ONLY**: no bytes may be copied into `fak`.  
**Tracking Issue:** [#10959](https://github.com/anthony-chaudhary/fak/issues/10959)  
**Parent Epics:** [#10193](https://github.com/anthony-chaudhary/fak/issues/10193) (Qwen3.8 native performance critical path)  
**Portfolio Index:** [#10960](https://github.com/anthony-chaudhary/fak/issues/10960)  
**Study Depth:** Deep (exhaustive inspection of all 21 tracked files  -  README, tuning log, research notes, batch launchers, PowerShell controllers, and patch/test sources)  
**Study Receipt:** `study_dbf8d303ad1ccc4ba93ef32e69ad905ac313e34696689924e1f79d793905d32d`
**Completeness Critic:** Verified  -  the complete tracked set was inspected (21 files): `README.md`, `TUNING_LOG.md`, `RESEARCH_NOTES.md`, `serve_task.bat`, `draft_controller.ps1`, `ninfer_console.ps1`, `ninfer_console.bat`, `patches/0001-tool-call-drift-salvage.patch`, `patches/tool_call_parser.cpp`, `patches/test_tool_call_parser.cpp`, `portable/README.md`, `portable/config/settings.cmd`, `portable/console.bat`, `portable/draft_controller.ps1`, `portable/install.ps1`, `portable/ninfer_console.ps1`, `portable/serve.bat`, `portable/start.bat`, `portable/status.ps1`, `portable/stop.bat`, `portable/uninstall.ps1`. There is no `CHANGELOG.md`, no `src/` tree, and no build system. Skips are justified: there is **no `src/` tree**, no build system, and no engine source, so the remaining files are deployment plumbing (configs, scheduled-task XML, small launch wrappers) with no borrowable mechanism. README.md:172 states explicitly (in Chinese) that there is **zero code diff from upstream** (`src/` full diff identical) versus `UDPSendToFailed/ninfer-4090@3d11fd64`.

---

## Executive Summary

`Azhu9701/ninfer-4090d` is **not an inference engine**. It is a Windows RTX 4090 D **deployment/config dossier** for Qwen3.8 serving: 21 tracked files, no `src/` directory, no C++ build despite the GitHub "C++" language field, and a README whose headline numbers are **prose descriptions of an uncloned upstream repository**, not artifacts verifiable in this tree.

The honest finding is therefore mostly negative-knowledge:

1. **Headline claims are unanchored.** The `-DNINFER_TARGET_SM_COUNT=114` / "112 CTA = 0.98 wave" occupancy story (README.md:16,25,145,174), the DirectStorage 1.3 disk-cache "1.8s / 1.9s TTFT" story (README.md:19,38,39,83; `TUNING_LOG.md:51-52`; `serve_task.bat:10,32`), and the `--kv-dtype rk4v4-e8` / `rk2v4-e8` E8-lattice INT8 KV fidelity story (README.md:78; `RESEARCH_NOTES.md:65-88`; `serve_task.bat:26`) are all CLI-flag strings or build-checklist prose with **no grid/occupancy, DMA/DirectStorage, or dequant kernel code in this repository**. They cannot be filed as grounded borrows because there is no `path:line@sha` code anchor.
2. **Only two artifacts are genuinely code-anchored**, and both are *outside* the engine:
   - A **restart-based, load-adaptive draft-depth controller** (`draft_controller.ps1:31-51`; `portable/draft_controller.ps1:62-80`) that polls `/metrics` and rewrites the server's `--draft-tokens` flag between 7 and 2, restarting the process through the Windows task scheduler to apply it.
   - A **tool-call-drift salvage patch** (`patches/0001-tool-call-drift-salvage.patch`, 241 lines, plus `patches/tool_call_parser.cpp` 366 lines and `patches/test_tool_call_parser.cpp` 326 lines) that recovers bare/malformed function calls with a safety gate.
3. **The draft-depth axis is already PRESENT in `fak`**  -  `fak` switches speculative depth under acceptance hysteresis and emits receipts *inside the engine*, so the controller is a cruder, non-native variant of a property `fak` already owns. It is recorded for comparison only.
4. **The tool-call salvage axis is PARTIAL in `fak`**  -  `fak` already has fail-closed Qwen function parsing and an explicit prose-precedence guard, but lacks lax recovery across missing `</parameter>` tags and the truncated-block refusal gate. This is the **single real borrow** and is filed as **#12944**.

The repository also carries a security hazard: `draft_controller.ps1` contains a **hardcoded API key** (fully redacted here) and Windows-local absolute paths (referred to here as the *operator-local checkout*). **None of those literals, paths, hostnames, or credentials appear in this note or the registry row.**

---

## Fan-Out Coverage & Subsystem Map

| Subsystem / Component | Key Files Inspected | Responsibility & Engineering Focus |
|---|---|---|
| **Deployment dossier / claims** | `README.md` | Prose description of upstream serving setup: occupancy target, disk cache, KV dtype; the claims are not backed by in-repo code. |
| **Tuning & research narrative** | `TUNING_LOG.md`, `RESEARCH_NOTES.md` | Self-reported TTFT and KV-fidelity numbers with no raw artifacts; CLI flag references only. |
| **Load-adaptive draft-depth controller** | `draft_controller.ps1`, `portable/draft_controller.ps1` | Real code: poll `/metrics` `llamacpp:requests_processing`, asymmetric debounce, rewrite flag file, restart via scheduled task. |
| **Tool-call-drift salvage patch** | `patches/0001-tool-call-drift-salvage.patch`, `patches/tool_call_parser.cpp`, `patches/test_tool_call_parser.cpp` | Real code: salvage bare function calls, lax parameter extraction, malformed-block recovery, and a safety gate refusing unsafe lead-ins. |
| **Launchers / configs** | `serve_task.bat`, `portable/serve.bat`, `portable/start.bat`, `portable/stop.bat` | Windows scheduled-task launch wrappers carrying the CLI flags named in the README claims. |
| **Remaining plumbing** | task/config/asset files | No engine mechanism; skipped. |

With 21 tracked files and no `src/`, the fan-out is complete: everything borrowable in this repository is in the controller and the patch.

---

## Detailed Technical Analysis

### 1. Load-Adaptive Draft-Depth Controller (Restart-Based)

Located at `draft_controller.ps1:31-51@3eaa163cdac422da64fed5d42d17c31174a62206` (root) and `portable/draft_controller.ps1:62-80@3eaa163cdac422da64fed5d42d17c31174a62206`:

The controller is a polling loop with an **asymmetric hysteresis debounce**:
- Every 10 seconds it reads `/metrics` and extracts `llamacpp:requests_processing`.
- When the concurrency load satisfies `$load -ge 2` for **3 consecutive** samples, it rewrites the server argument `--draft-tokens` to `2` and **restarts the server via `schtasks /Run`**.
- It reverts `--draft-tokens` to `7` only after `$load -le 1` for **6 consecutive** samples  -  a deliberately longer window so the system does not oscillate at the boundary.

The mechanism is **file rewrite + process kill + process restart**, not an in-engine depth change. The server's own runtime is unaware of the change until it is relaunched, and the restart drops any in-flight KV/prefix state. It is a genuine piece of engineering, but it operates at the orchestration layer, not the scheduler.

Security note: this file also embeds a **hardcoded API key** (fully redacted) and **operator-local absolute paths**; neither is reproduced anywhere in this study.

### 2. Tool-Call-Drift Salvage Patch

Located at `patches/0001-tool-call-drift-salvage.patch` (241 lines), `patches/tool_call_parser.cpp` (366 lines), and `patches/test_tool_call_parser.cpp` (326 lines) at `3eaa163`:

This is the repository's one cleanly engineered, test-backed subsystem. It adds:

- `try_salvage_bare_function_calls`  -  detects function-call invocations emitted without the expected wrapping delimiters.
- `salvage_parameters`  -  **lax parameter extraction** that recovers arguments even when a `</parameter>` close tag is missing.
- `try_salvage_malformed_block`  -  recovers a function block whose structural delimiters are damaged.
- `lead_in_may_precede_salvaged_call`  -  the **safety gate** that decides whether prose may legally precede a salvaged call. It admits a lead-in only when it is `< 400` characters, contains no triple-backtick code fence, and has no `\n\t` / `\n    ` indentation; truncated blocks are **refused** rather than salvaged.
- A **streaming filter** that holds output when it sees the `<function=` marker, so a partially streamed call is not emitted prematurely.

This is a production-quality drift-salvage design: recovery is broad, but the safety gate is conservative and fail-closed on the structurally suspicious cases.

### 3. Unanchored Headline Claims (Negative Knowledge)

Three headline claims have **no code anchor in this repository** and describe the uncloned upstream `UDPSendToFailed/ninfer-4090@3d11fd64`:

- **114-SM wave-grid alignment:** `README.md:16,25,145,174` claim `-DNINFER_TARGET_SM_COUNT=114` gives a "GQA INT8 decode grid = 112 CTA = 0.98 wave"; README.md:145/174 present the build step as an unfinished manual checklist item. There is no grid or occupancy code here.
- **DirectStorage 1.3 disk cache:** `README.md:19,38,39,83`, `TUNING_LOG.md:51-52`, `serve_task.bat:10,32` reference `--disk-cache --disk-cache-gb 200` and self-report "1.8s / 1.9s TTFT". There is no DMA/DirectStorage code here  -  only the CLI flag.
- **E8-lattice INT8 KV:** `README.md:78`, `RESEARCH_NOTES.md:65-88`, `serve_task.bat:26` reference `--kv-dtype rk4v4-e8` / `rk2v4-e8` and self-report 98.7% / 96.2% fidelity with no raw artifact. There is no dequant kernel here.

These are recorded strictly as **unverified upstream claims**, not as borrows.

---

## Upstream Worldview Reconstruction & Systems Tradeoffs

The author's world is a **single Windows workstation with an RTX 4090 D**, a consumer card with fewer SMs than the datacenter parts the upstream tooling was tuned for. That worldview explains the three headline obsessions:

1. **Squeeze the SM count.** The 114-SM / "0.98 wave" number is an attempt to right-size a decode grid to the exact device so no wave is wasted. On a fixed consumer card, grid misalignment is pure lost throughput.
2. **Trade disk for VRAM.** With memory pressure on a 24 GB card, DirectStorage 1.3 is treated as a way to keep a 200 GB disk cache hot and cut TTFT to ~2 s, accepting PCIe/NVMe bandwidth as the new bottleneck.
3. **Push fidelity into the KV format.** E8-lattice INT8 KV is an attempt to shrink cache footprint without the usual quality collapse.

The **tradeoff** this dossier embodies is *orchestration over engine integration*: the draft-depth controller restarts the process rather than asking the engine to change depth, because the author does not own the engine internals. That yields a working result but loses in-flight state and cannot reuse the prefix cache segment across a depth change. It is exactly the tradeoff `fak` avoids by owning the scheduler.

The second tradeoff is **recovery versus safety in tool-call parsing**: the upstream parser is too strict and drops calls, so the author loosens extraction  -  but wraps the loosening in a lead-in gate so recovery cannot be hijacked by prose or fenced examples. That balance is the genuinely reusable idea.

---

## Borrow Candidates for the fak Agent Kernel

### Candidate 1: 114-SM Wave-Grid Alignment

- **Source Anchor:** NONE  -  prose only. The claim lives in `README.md:16,25,145,174@3eaa163` and has no supporting code in this repository (no grid/occupancy source).
- **One-Line Technique:** Size the decode grid to the device's exact SM count so the GQA INT8 decode launch fills an integer number of waves (claimed 112 CTA = 0.98 wave at 114 SMs).
- **Specific Axis:** Decode grid/occupancy alignment to a fixed device SM count.
- **Comparison with fak:** `fak` models this axis with `internal/compute/decode_occupancy.go:256` `(a NVArch) Occupancy(l DecodeLaunch, deviceSMs int)`, grid views `IdleSMs/Waves/WaveQuantWaste/DeviceOcc` at `:181-199`, `const A100SMs = 108` at `:109-112`, the `NVArch` table `:74-84` (sm_80/90/100 only  -  **no Ada/sm_89 row**), and `DecodeGapReport` + `decodeTailGapFloor = 0.05` at `:390-399`; workgroup geometry also appears at `internal/compute/coopmat.go:266-275`. `fak`'s model is arch-parameterized and computes waste, but it is a **calculator, not an active grid-launch planner**, and it lacks an Ada/sm_89 (4090-class) row. (**PARTIAL on-axis**).
- **Worldview Rationale:** On a fixed consumer card, a grid that does not divide evenly into the device's SM count strands SMs on the tail wave; sizing to the exact SM count trades headroom for utilization.
- **Disposition:** `RECORD-ONLY`  -  no code anchor upstream, so it cannot be a grounded borrow. Note `fak`'s missing Ada/sm_89 row in `decode_occupancy.go` as a *possible* independent follow-on, but do **not** file a second issue from this pass (the coordinator filed exactly one grounded borrow; avoid the file-nothing / mega-ticket anti-patterns).

### Candidate 2: Restart-Based Dynamic Draft Depth (MTP 7<->2)

- **Source Anchor:** `draft_controller.ps1:31-51@3eaa163cdac422da64fed5d42d17c31174a62206`; `portable/draft_controller.ps1:62-80@3eaa163cdac422da64fed5d42d17c31174a62206`
- **One-Line Technique:** Poll server concurrency every 10 s and switch speculative draft depth between 7 and 2 with asymmetric debounce (3 high-load samples down, 6 low-load samples up), applying the change by rewriting the launch flag and restarting the process.
- **Specific Axis:** Load-adaptive speculative/MTP draft depth control.
- **Comparison with fak:** **PRESENT on-axis.** `fak` already changes draft depth inside the engine under acceptance hysteresis and emits receipts: `internal/model/adaptive_draft_depth.go:10` `AdaptiveDraftDepthController` + `:40` `Observe` (`RaiseAcceptance/LowerAcceptance/HysteresisWindows`) + `:69` `EvaluateAdaptiveDraftDepth`; `internal/model/metal_mtp.go:180` `Qwen38MTPAdaptiveDepthGovernor`, `:44` `DraftDepth`, `:617` `AdaptiveReceipt`; `internal/polymodel/specdepthconfidence.go:9` `SpecDepthConfidenceGate.DraftDepth`; `internal/polymodel/acceptance_profile.go`. `fak`'s variant is superior in kind: it changes depth **without a restart**, so prefix/KV state survives, and it records the decision.
- **Worldview Rationale:** On a memory-constrained single GPU the author cannot afford deep drafting when the server is already saturated; the restart is the only lever available without engine internals.
- **Disposition:** `RECORD-ONLY` (comparison). Do **not** file  -  the property already exists natively; the controller is a cruder orchestration-layer variant.

### Candidate 3: DirectStorage 1.3 Disk Cache / Fast Restore

- **Source Anchor:** NONE  -  CLI flag + prose only. `README.md:19,38,39,83`, `TUNING_LOG.md:51-52`, `serve_task.bat:10,32@3eaa163` reference `--disk-cache --disk-cache-gb 200` and self-report "1.8s / 1.9s TTFT"; no DMA or DirectStorage code is present.
- **One-Line Technique:** Keep a large disk-backed cache to cut first-token latency to ~2 s by streaming weights through the Windows DirectStorage 1.3 API.
- **Specific Axis:** Disk-resident model/KV tier and cold-restore latency.
- **Comparison with fak:** **PARTIAL/PRESENT on-axis**, but **not on Windows**. `fak` has disk-direct and KV-paging seams: `internal/compute/disk_direct.go:11-27` (O_DIRECT aligned pread + `POSIX_FADV_DONTNEED`), `internal/compute/amd_gpudirect_storage.go:232` `DirectStorageMemorySlab`, `:344` `DirectNVMeSwapIn`, `:397` `PrefetchBlocks`; `internal/compute/amd_gpudirect_kvpaging.go` `Pin` (`:789`), `OffloadBlock` (`:894`), `RestoreBlock` (`:915`), `ReadBlock` (`:861`); `internal/compute/strix/mmap_loader.go:174` `MmapModelLoader`; `internal/compute/pagecachefloor.go:15-29`. The Windows DirectStorage 1.3 API itself is **ABSENT**  -  every existing seam is POSIX/Linux or AMD.
- **Worldview Rationale:** Disk as a cheap extension of VRAM is attractive only when the accelerator is memory-poor and the workload tolerates NVMe latency; it reframes the bottleneck from VRAM to storage bandwidth.
- **Disposition:** `RECORD-ONLY`  -  no code anchor upstream, so no grounded borrow this pass. The Windows DirectStorage gap is noted for a possible independent future ticket.

### Candidate 4: Tool-Call Drift Salvage (`salvage_parameters` + `lead_in_may_precede_salvaged_call`)

- **Source Anchor:** `patches/0001-tool-call-drift-salvage.patch` (241 lines), `patches/tool_call_parser.cpp:1-327@3eaa163cdac422da64fed5d42d17c31174a62206`, `patches/test_tool_call_parser.cpp:1-292@3eaa163cdac422da64fed5d42d17c31174a62206` at `3eaa163`.
- **One-Line Technique:** Recover function calls that drift from the expected format  -  bare calls, a block missing a `</parameter>` close tag, or a malformed block  -  while a lead-in safety gate refuses salvage when prose or a fenced example precedes the call or the block is truncated.
- **Specific Axis:** Agent tool-call parsing robustness across model-format drift, with fail-closed safety.
- **Comparison with fak:** **PARTIAL on-axis.** `fak` already parses the Qwen function dialect and guards against example prose, but is stricter on malformed input: `internal/agent/toolcall_fallback.go:92` `qwen_function_parameter` dialect, `:156-160` `extractQwenFunctionBlocks`, `:230` `parseQwenFunctionToolCall` (**fail-closed on missing `</parameter>`**), `:377-391` `isPrecededByExampleProse` (**blocks lifting after prose**); negative tests at `internal/agent/toolcall_fallback_test.go:86,101,356`. `fak` lacks the **lax recovery across a missing close tag** and the explicit **truncated-block refusal** gate.
- **Worldview Rationale:** A strict parser silently discards a real call that a model emitted with minor drift, losing an agent turn; the salvage path must therefore be broad  -  but broadness without a lead-in gate would let prose or a documented example be mistaken for an invocation, so the gate bounds recovery to short, unfenced, unindented lead-ins.
- **Disposition:** `INSPIRE-ONLY` (**no license**)  -  implement an equivalent in `fak`'s own words; do not copy bytes. Filed as **#12944**.

### Candidate 5: License / Provenance Finding

- **Source Anchor:** Repository root at `3eaa163cdac422da64fed5d42d17c31174a62206`  -  absent `LICENSE`/`COPYING`/`NOTICE`/`AUTHORS`; GitHub API `license` is `null`.
- **One-Line Technique:** Treat an unlicensed repository as all-rights-reserved and constrain every extraction to inspiration, never verbatim reuse.
- **Specific Axis:** Legal/provenance disposition.
- **Comparison with fak:** **ABSENT on-axis**  -  `fak`'s other study rows carry explicit MIT/Apache-2.0 provenance and permit direct port; this source does not.
- **Worldview Rationale:** A hobby deployment dossier needs no license to run locally; once its ideas enter an Apache-2.0 product, the absence of a grant becomes the binding constraint.
- **Disposition:** `INSPIRE-ONLY` for all five candidates; all copied bytes forbidden.

---

## Candidate Summary Table

| Candidate | Source Location | Axis | fak Comparison | Worldview Rationale | Disposition | Target Seam |
|---|---|---|---|---|---|---|
| **1. 114-SM Wave-Grid Alignment** | NONE (prose: `README.md:16,25,145,174@3eaa163`) | Decode grid/occupancy | **PARTIAL** (arch-parameterized calculator; no Ada/sm_89 row) | Fixed consumer card needs grid sized to exact SM count | `RECORD-ONLY` | `internal/compute/decode_occupancy.go` |
| **2. Restart-Based Dynamic Draft Depth (7<->2)** | `draft_controller.ps1:31-51@3eaa163`; `portable/draft_controller.ps1:62-80@3eaa163` | Load-adaptive MTP depth | **PRESENT** (in-engine hysteresis + receipts) | No engine internals, so restart is the only depth lever | `RECORD-ONLY` | `internal/model/adaptive_draft_depth.go` |
| **3. DirectStorage 1.3 Disk Cache / Fast Restore** | NONE (flags/prose: `serve_task.bat:10,32@3eaa163`) | Disk-resident tier & TTFT | **PARTIAL** (POSIX/AMD seams; Windows DirectStorage ABSENT) | Disk as cheap VRAM extension under memory pressure | `RECORD-ONLY` | `internal/compute/disk_direct.go`, `amd_gpudirect_kvpaging.go` |
| **4. Tool-Call Drift Salvage** | `patches/0001-tool-call-drift-salvage.patch`; `patches/tool_call_parser.cpp:1-327@3eaa163`; `test_tool_call_parser.cpp:1-292@3eaa163` | Tool-call parse robustness | **PARTIAL** (fail-closed parse + prose guard; no lax close-tag recovery) | Broad recovery needs a conservative lead-in gate | `INSPIRE-ONLY` (filed **#12944**) | `internal/agent/toolcall_fallback.go` |
| **5. License / Provenance Finding** | repo root `@3eaa163` (no license) | Legal disposition | **ABSENT** (no grant) | Local hobby runs need no license; product reuse does | `INSPIRE-ONLY` | n/a |

---

## License, Provenance & Attribution Disposition

- `Azhu9701/ninfer-4090d@3eaa163cdac422da64fed5d42d17c31174a62206` ships **no license**  -  no `LICENSE`, `COPYING`, `NOTICE`, or `AUTHORS`; the GitHub API `license` field is `null`. Default copyright applies and all rights are reserved.
- Consequently **every** candidate is **INSPIRE-ONLY**. Implement equivalent behavior in `fak`'s own code and words; do **not** copy source text, patch bytes, comments, or test cases into `fak`.
- The tool-call salvage candidate (#12944) must be implemented from the *idea* (lax close-tag recovery + lead-in safety gate), not from the upstream patch.
- **Security hazard:** `draft_controller.ps1` embeds a hardcoded API key and operator-local absolute paths. Nothing of the kind - no key, no absolute user path, no hostname, no credential - appears in this note or in `docs/research/monitored-repositories.json`.

---

## Concrete Follow-up Implementation Tickets

- **[#12944](https://github.com/anthony-chaudhary/fak/issues/12944)**  -  `feat(agent): lax tool-call parameter salvage with a lead-in safety gate` (Candidate 4, INSPIRE-ONLY). This is the single grounded borrow from this study.
- Candidates 1, 2, 3, and 5 are recorded for comparison and negative knowledge; no further issues are filed from this pass. Study receipt `study_dbf8d303ad1ccc4ba93ef32e69ad905ac313e34696689924e1f79d793905d32d`.

---

**Companions:** [`docs/notes/CONCEPT-STUDY-VCRUZ305-GLM53-EXL3-2026-09-03.md`](docs/notes/CONCEPT-STUDY-VCRUZ305-GLM53-EXL3-2026-09-03.md) - [`docs/notes/CONCEPT-STUDY-ADRIENBRAULT-RTX5090-2026-09-03.md`](docs/notes/CONCEPT-STUDY-ADRIENBRAULT-RTX5090-2026-09-03.md) - `field-borrow` witness discipline - parent epic [#10193](https://github.com/anthony-chaudhary/fak/issues/10193) - [portfolio index #10960](https://github.com/anthony-chaudhary/fak/issues/10960)
