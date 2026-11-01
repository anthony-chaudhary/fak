# RESULTS — maya + Strata on owned NVIDIA HW (2026-10-30/31)

Goal `GOAL-maya-strat-hw-config-sweep`. All perf numbers below are **[HW-WITNESSED]**
on named nodes with argv + binary-sha + model-oid pins (full cell objects:
[raider-fak/RECEIPT.json](raider-fak/RECEIPT.json),
[aegis-fak/RECEIPT.json](aegis-fak/RECEIPT.json)); pins and node envelopes:
[README.md](README.md). Oracle prompt every cell:
`What color is the sky on a clear day? Answer in one sentence.` (greedy, temp 0, top_p 1).

## a) TL;DR — winners

| engine | model | node | best config (argv-delta) | decode tok/s | ratio vs same-day llama.cpp baseline | label |
| --- | --- | --- | --- | --- | --- | --- |
| **strata-dsv41 0.1.53** (@46c152a1) | DS4.1-Flash GSQ-RCO IQ2_XXS | raider-fak | BASE defaults (`--pack <ds41 pack> --max-context 512 --tp 1 --threads on --vision off`, warm cache) | **28.91 @ctx512** (10.46 @ctx2048) | **≈36x** vs 0.803 (same-day pinned T0); 0.025x vs historical 1160.104 | [HW-WITNESSED] GREEN-BASELINE-STRATA |
| **maya-ds4 1.1.39** (@376b4b87) | DS4.1-Flash GSQ-RCO IQ2_XXS | raider-fak | BASE defaults, warm page cache (M5 control; first request pays ~18 GB pre-read) | **12.7 @ctx512** (3.6 @ctx2048) | ≈15.8x vs 0.803 | [HW-WITNESSED] GREEN-BASELINE-MAYA-WARM-CONTROL |
| llama.cpp b11870 (baseline) | DS4.1-Flash GSQ-RCO IQ2_XXS | raider-fak | pinned T0 (`-ngl 40 --n-cpu-moe 36`) | 0.803 @ctx512 / 0.56 @ctx2048 | 1.0x (baseline; collapse attributed DISK-BOUND-STRUCTURAL, §d) | [HW-WITNESSED] COLLAPSE-REPRODUCED-ATTRIBUTED |
| **strata 0.1.53** (@46c152a1) | Qwen3.8-Flash-Next GSQ-RCO Q2_0 | aegis-fak | `--expert-cache auto --mmap-experts --prefill auto --kv int8 --greedy`, warm page cache (S2) | **30.55 @ctx2048** (12.37 @ctx512 cold) | **≈3.6x** vs same-day llama.cpp b11870 `--cpu-moe` (8.556 @ctx2048) | [HW-WITNESSED] GREEN-BASELINE-STRATA |
| llama.cpp b11870 (baseline) | Qwen3.8-Flash-Next GSQ-RCO Q2_0 | aegis-fak | `-ngl 9999 --cpu-moe --lazy-mode auto --flash-attn auto -np 1 --batch-size 2` (pinned full-offload OOMs at load — 35.6 GB > 16 GB card; minimal deviation `--cpu-moe`, typed OOM_LOAD_LLAMA_PINNED_ARGV) | 8.812 @ctx512 warm / 8.556 @ctx2048 cold | 1.0x (baseline) | [HW-WITNESSED] OK-DEVIATION-DOCUMENTED |
| maya v1.1.39 (@376b4b87) | CYBER-FROST-3.8-PS (its documented Qwen path) | aegis-fak | — | — | — | ABSENT-IN-WINDOW: CYBER-FROST download 9.8% at window close, resumed detached (PID 31936); binary staged + sha-verified, unused (§g) |

## b) Scope & pins

Engines: maya `mw00/project-maya` @ `376b4b87…` (v1.1.39), strata `Niko1221/Strata`
@ `46c152a1…` (0.1.53), baseline llama.cpp b11870 — binary shas and model shard oids
per [README.md](README.md) §Pinned artifacts. Node envelopes (boot ids, GPU UUIDs,
driver 615.78.08, raider AVX512F_ABSENT) per README.md §Nodes. Campaign: ABBA
(L → M → L′ → S) under measure lease `hwl-c7e36f713353b619`, 2026-10-30T17:51–~19:45Z,
profile-OFF, 300 s inter-block cooldowns, kill-after-every-cell (VRAM back to 2 MiB
each time).

## c) Raider DS4.1 ABBA — full 11-cell table

Copied faithfully from `raider-sweep-cells.json` (cell objects verbatim in
[raider-fak/RECEIPT.json](raider-fak/RECEIPT.json)). All 11 cells verdict `ok`;
greedy text byte-identical across L/M/S at ctx512; S2 differs in word order at
ctx2048 (argmax divergence between engines, coherent, not an error).

| cell | engine | argv-delta | ctx/maxtok | decode tps | TTFT ms | VRAM MiB | verdict |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| L1 | llama.cpp b11870 | T0: `-ngl 40 --n-cpu-moe 36`; ctx512 | 512/16 | 0.803 | 34048.0 | peak 19380 / steady 19360 | ok — COLLAPSE-REPRODUCED |
| L2 | llama.cpp b11870 | T0; ctx2048 | 2048/64 | 0.56 | 42120.9 | peak 19376 / steady 19368 | ok — COLLAPSE-REPRODUCED |
| M1 | maya-ds4 1.1.39 | BASE: `--pack <ds41>/maya-ds4 --max-context 512 --tp 1 --threads on --vision off`; serve stack port 18091; reasoning off; greedy; **cold** cache | 512/16 | 2.8 | 6908.6 | steady 23676 | ok — GREEN-BASELINE-MAYA |
| M2 | maya-ds4 1.1.39 | BASE; `--max-context 2048` (rest = M1); warm | 2048/64 | 3.6 | 925.2 | — | ok — GREEN-BASELINE-MAYA |
| M3 | maya-ds4 1.1.39 | ABLATION: env `MAYA_DS4_STREAM=0` (PCIe stream-fetch off); ctx512; warm | 512/16 | 8.6 | 486.6 | — | ok — GREEN-ABLATION-STREAM-OFF |
| M4 | maya-ds4 1.1.39 | ABLATION: `--mmap-experts on` (in-place, no pinned-RAM tier); stream default ON; ctx512; warm | 512/16 | 5.8 | 2977.1 | — | ok — GREEN-ABLATION-INPLACE |
| M5 | maya-ds4 1.1.39 | BASE (no ablations); warm-cache control for M3/M4 | 512/16 | 12.7 | 772.1 | — | ok — GREEN-BASELINE-MAYA-WARM-CONTROL |
| L1′ (L1p) | llama.cpp b11870 | T0; ctx512; B-bracket with `/proc/pid/io` attribution; warm | 512/16 | 1.074 | 26492.1 | steady 19380 | ok — COLLAPSE-REPRODUCED-ATTRIBUTED |
| L2′ (L2p) | llama.cpp b11870 | T0; ctx2048; B-bracket with `/proc/pid/io` attribution; warm | 2048/64 | 0.737 | 49136.0 | — | ok — COLLAPSE-REPRODUCED-ATTRIBUTED |
| S1 | strata-dsv41 0.1.53 | BASE: `--pack <same ds41 pack> --max-context 512 --tp 1 --threads on --vision off --ids <19 ids> --n 16`; generate mode; warm | 512/16 | 28.91 | — (prefill 63.95 tok/s engine line) | — | ok — GREEN-BASELINE-STRATA |
| S2 | strata-dsv41 0.1.53 | BASE; `--max-context 2048` (rest = S1); warm | 2048/64 | 10.46 | — | — | ok — GREEN-BASELINE-STRATA |

Decode sources: L/M cells `server_timings` (stream clock agrees within ≤0.24 tok/s);
S cells `engine_timing_line`. Per-cell NVMe reads: L2 ~105 GiB device-level;
L1′ 68.1 GB / L2′ 73.2 GB process-level; M1 21.35 GB, M2 7.1 GB, M3 2.76 GB,
M4 8.73 GB, M5 2.31 GB; S1 19.5 GB (27.3 MB/token), S2 23.4 GB (132.8 MB/token).

## d) Baseline anomaly — historical regime non-reproducible, attributed DISK-BOUND-STRUCTURAL

Historical llama.cpp b11870 arms on this node/model (same boot id, same binary sha,
same model oids): `docs/benchmarks/raider-5090/ds41-gsq-iq2xxs-20261028/RECEIPT.json`
@ 2026-10-29T04:42Z — ARM-1 (T1) 712.715, **ARM-2 (T0, ctx512) 1160.104**, ARM-3
(T2) 1055.597, ARM-4 (best-T0, ctx2048) 1014.489 tok/s. A same-arm rerun the same
night (`RECEIPT-20261030.json`, 2026-10-29T20:25Z) already showed 1.05 tok/s.
Same-day 2026-10-30 re-measurement under the identical pinned argv: L1 0.803,
L2 0.56, L1′ 1.074, L2′ 0.737 tok/s.

Attribution (L1′/L2′ bracket, `/proc/pid/io`): the llama-server process itself pulled
**68.1 GB (L1′) / 73.2 GB (L2′) from NVMe per 11-token generation** —
`rchar=0`, `syscr=0` → zero `read()` syscalls, pure mmap demand paging of mapped
expert weights at **~1.2–1.9 GB/s** sustained; `majflt_delta=0`, `minflt_delta`
19,624/22,082; GPU idle at **1% util / 19 W** throughout. A warm page cache moved
the result only ~30% (0.803 → 1.074): the touched working set (~68–73 GB per
generation incl. prefill) exceeds the ~28 GB RAM page-cache budget, so re-reads
persist. Root cause: the ~123 GB expert pool cannot be resident in 24 GB VRAM +
30 GB RAM, so decode streams experts from NVMe through CPU mmap every token.

Labels: the historical 1160.104 tok/s regime is
**REGIME-NONREPRODUCIBLE-under-pinned-argv-on-this-node**. This is NOT claimed to be
a hardware fault: Xid count unchanged at 12 across the whole window, greedy text
byte-identical to yesterday, VRAM steady 19,360–19,380 MiB (same as yesterday),
clean kills to 2 MiB. It is a memory-budget regime shift: whatever host-memory
state made the 10-29 morning run fast (resident expert working set) is absent now,
and the pinned argv does not change it back.

## e) Config-space ablations (raider, maya-ds4 × DS4.1)

Knob sets and results (all warm except M1; NVMe reads during the request shown to
separate the mechanisms):

| knob set | cell | decode tok/s | req-time NVMe reads |
| --- | --- | ---: | ---: |
| BASE defaults: PCIe stream-fetch ON (single-GPU default), mmap-experts auto with ~18 GB most-routed experts copied to a pinned-RAM tier ahead, `--tp 1 --threads on --vision off` — **cold** page cache (first request pays the ~18 GB pre-read) | M1 | 2.8 | 21.35 GB |
| `MAYA_DS4_STREAM=0` (PCIe stream-fetch OFF; reader thread stops re-pulling experts, CPU lane works off the pinned RAM copy) | M3 | 8.6 | 2.76 GB |
| `--mmap-experts on` (experts kept in place in the model files; no pinned-RAM tier) | M4 | 5.8 | 8.73 GB |
| BASE defaults, **warm** page cache (control) | M5 | **12.7** | 2.31 GB |

Reading: with the cache warm the default BASE config wins (12.7 > 8.6 > 5.8) — the
pinned-RAM tier (+~120% vs in-place-only) and stream-fetch (+~48% vs stream-off)
both help on this node. The raw M3 > M1 gap (8.6 vs 2.8) is a cold/warm confound,
not the ablation effect; M3 is interpretable only against M5. Run-to-run variance
within BASE is large (M2 3.6 vs M5 12.7, both warm; pre-read/RAM-copy completion
timing suspected) — report ranges, not point values. Recommended winning config
for maya × DS4.1 on this node: defaults, warm page cache.

## f) Arch-refusal data points (committed evidence)

The originally pinned target model `Qwen3.8-27B-UD-Q2_K_XL.gguf`
(`general.architecture = qwen35`, dense) is **rejected by both engines at their
pins** — hard code gates, witnessed verbatim on both nodes (typed failures in
[raider-fak/RECEIPT.json](raider-fak/RECEIPT.json)):

- **Maya** (`tools/iq_pack.py --gguf <file>` smoke, raider + aegis):
  `general.architecture is 'qwen35'; the packer knows qwen4exp and glm5-next and glm5next`
  — code gate `include/strata/artifact/gguf_reader.hpp:500@376b4b87` requires
  `general.architecture == "qwen4exp"`; no fallback loader. Control with the
  fallback Q4_K_M file (same arch) gave the identical rejection → arch-based, not
  quant-based.
- **Strata** (documented import path `python3 strata.py models import <gguf> --plan`,
  aegis):
  `this model type is not supported by the Strata engine (add it as a Custom model API with llama-server instead); this is not a model size Strata knows.`
  — code gate `include/strata/artifact/model_family.hpp:21-23@46c152a1` accepts
  only `{qwen4exp, qwen35moe}`.

**AVX-512 expectation REVISED.** RAIDER-BUILD predicted `BLOCKED-CPU-AVX512` for
Strata on this node (the raider APU lacks AVX512F; the canonical-pack CPU expert
pool path did refuse during smoke attempt 2: "this CPU cannot run the expert
kernel: missing AVX512F …"). In the sweep, `strata-dsv41` (built
`-DSTRATA_BUILD_DSV41=ON`) ran both S-cells green via **runtime dispatch** on the
AVX512F-absent host — the AVX-512 gate binds the canonical-pack CPU pool path, not
the DS4.1 build.

## g) Aegis — Flash-Next Q2_0 (the qwen4exp home turf both OSS engines accept)

Campaign 2026-10-31T03:34–~04:20Z under measure lease `hwl-b79c191a981ccf24`
(first acquire `hwl-f4d6752f699840e6`; fak.service stopped via maintain
`hwl-990a03f12fc92dee`, restored active at close, MainPID 32556). Boot
`31fc7046…` unchanged (no reboot); Xid delta 0 on every cell; VRAM back to
10 MiB after every kill. Artifact: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
@ ed59f920, Q2_0 (66.4 GB, Apache-2.0) — all three files sha-verified against
the `src/strata/setup.py:370-373` pins. Full cell objects:
[aegis-fak/RECEIPT.json](aegis-fak/RECEIPT.json).

| cell | engine | argv-delta | ctx/maxtok | decode tps | TTFT ms | VRAM MiB | verdict |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| L1 | llama.cpp b11870 | `-ngl 9999 --cpu-moe --lazy-mode auto --flash-attn auto -np 1 --batch-size 2`; **warm** (cold probe discarded first) | 512/16 | 8.812 | 96.3 | steady 3682 | ok — OOM_LOAD_LLAMA_PINNED_ARGV deviation documented |
| L2 | llama.cpp b11870 | same, ctx2048; **cold** fresh process | 2048/64 | 8.556 | 3933.5 | steady 3726 | ok |
| S1 | strata 0.1.53 | `--pack pack-q20 --native shard1 --ple-gguf shard2 --expert-cache auto --mmap-experts --prefill auto --kv int8 --greedy`; **cold** (5208 experts streamed from NVMe in 6.0 s) | 512/16 | 12.37 | 6904.0 | expert-cache 9.13 GiB (7090 slots) | ok — OOM_HOST_STRATA_ARENA deviation documented |
| S2 | strata 0.1.53 | same, ctx2048/64; **warm** page cache (stream 393 ms) | 2048/64 | 30.55 | 1094.4 | expert-cache 9.11 GiB (7075 slots) | ok |

Typed deviations (verbatim errors in the receipt):
- **L-cells** — pinned full-offload (`-ngl 9999`, no cpu-moe) OOMs at load:
  `allocating 35609.80 MiB on device 0: cudaMalloc failed: out of memory`
  (35.6 GB weights > 16 GB card) → `--cpu-moe` (MoE experts CPU-resident, rest
  GPU), the minimal feasibility fix. OOM_LOAD_LLAMA_PINNED_ARGV.
- **S-cells** — strata's default 31 GiB pinned expert arena was oom-killed on
  this 30 GB host (`Killed process 30942 (strata) total-vm:53769752kB,
  anon-rss:29639964kB`; setup.py itself declares Q2_0 wants 48 GB RAM) →
  `--mmap-experts`, which is setup.py's own documented low-RAM choice for Q2_0
  on <33 GB RAM. OOM_HOST_STRATA_ARENA.
- **Prompt-path caveat** — strata's pack chat-template renders 40–41 prompt ids
  vs llama.cpp's 27 server-side (oracle text identical, both greedy): decode
  rates stay comparable; TTFT/prefill are NOT strictly cross-engine comparable.
  Answer wording differs ("blue" vs "typically blue") — expected across
  quant/kernel paths, both coherent one-sentence answers.

Reading: strata wins steady-state single-stream decode on aegis —
**30.55 tok/s @ctx2048 warm = 3.6x the same-day llama.cpp baseline** (8.556),
and even cold S1 (12.37) beats warm L1 (8.812). The mmap-experts +
expert-cache-auto regime parks ~9.1 GiB of the hottest experts in VRAM and
streams the rest from NVMe; the cold/warm spread (12.37 → 30.55) is strata's
documented mmap A/B behavior. Recommended winning config for strata ×
Flash-Next Q2_0 on this node: `--expert-cache auto --mmap-experts --prefill
auto --kv int8`, warm page cache.

Maya × aegis: **ABSENT-IN-WINDOW**. Maya's documented Qwen path is
CYBER-FROST-3.8-PS (112.5 GB, Qwen Community License 1.0); its download reached
9.8% (~10 MB/s) by window close and was resumed detached (PID 31936,
cf.done/cf.failed markers) for a future campaign. Raw Q2_0 through Maya is
UNPROVEN (arch gate, §f) and was not attempted.

Incident note (host stability): the OOM attempt took systemd-journald/networkd
down via watchdog and hung the sshd banner ~6 min; boot id unchanged (no
reboot); a watchdog covered all subsequent runs (zero events). Treat aegis as
OOM-sensitive at 30 GB RAM + 1 GB swap: no pinned arenas ≥ host RAM without
`--mmap-experts` or equivalent.

## h) Cross-engine verdict + implication

For DS4.1-Flash on raider-fak, **strata-dsv41 is preferred over llama.cpp**:
28.91 tok/s @ctx512 = **36x the same-day pinned-argv baseline** (0.803) and
**2.3x maya warm BASE** (12.7). The differentiator is strata's profile-driven
RAM-store prefetch of the hottest experts (91.0% store hit at ctx512; 27.3 MB/token
disk traffic vs ~6.2 GB/token for llama.cpp and ~1.9 GB/token for maya BASE).
llama.cpp b11870 under its pinned T0 argv is structurally disk-bound on this
node/memory budget (§d) and is not a usable reference for future climbs here.
For Flash-Next Q2_0 on aegis-fak, **strata 0.1.53 is likewise preferred over
llama.cpp**: 30.55 tok/s @ctx2048 warm = 3.6x the same-day `--cpu-moe`
baseline (§g) — there the llama.cpp baseline is healthy, so the win is a clean
engine delta, not a baseline pathology. Strata therefore wins both witnessed
(model, node) pairs of the campaign — DS4.1-Flash × raider (36x, disk-bound
baseline) and Flash-Next Q2_0 × aegis (3.6x, healthy baseline) — with maya
trailing on raider (12.7 warm BASE) and unrun on aegis (CYBER-FROST artifact
absent in window; download resumed for a follow-on).

Mechanism half of the goal: multi-candidate default-engine selection
(commit `1ada7a1ab1` "feat(strix): select default engine across native/reference/OSS
candidates", lane `ticket-maya-strat-default-engine-preference-20261030`, lands in
this campaign as step 5d). The data-driven flip it drives will key off receipts
like this one — a witnessed-winning OSS engine becomes preferred over llama.cpp
only when a same-day, same-node, argv/sha-pinned receipt shows it.

### Label discipline

- [HW-WITNESSED]: every verdict-ok cell above (argv + binary sha256 + model oid pins
  in [raider-fak/RECEIPT.json](raider-fak/RECEIPT.json) and
  [aegis-fak/RECEIPT.json](aegis-fak/RECEIPT.json)); historical arms cited in §d
  carry their own committed receipts.
- ESTIMATE: none outstanding — the aegis tiering figures that were planning
  estimates in the probe are now witnessed (§g: mmap-experts path green, default-
  arena OOM witnessed); probe planning numbers are superseded.
- UPSTREAM-REPORTED: any engine-doc performance claims not re-measured here (none used
  as a denominator in this document).
- No invented numbers: every figure traces to `raider-sweep-cells.json`,
  `raider-sweep.md`, `raider-build.md`, `aegis-build.md`, `aegis-sweep-cells.json`,
  `aegis-sweep.md`, `flashnext-probe.md`, or the cited historical receipts.
