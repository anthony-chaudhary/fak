# Mac Subagent Prefix-Cache Reuse: Head-to-Head vs llama.cpp `llama-server`

Status: **PILOT RECEIPT (fak arm blocked)**, issue fak#13023. The OSS reference arm has a live pilot receipt (§5.1); the fak arm and the full-geometry run are pending.

This document is the methodology + repro-protocol skeleton for a Mac-native head-to-head
between fak and the best practical OSS reference — **llama.cpp `llama-server`** (its prefix
cache / continuous batching) — on Apple Silicon. No measured number appears below yet: the
results table is entirely `<PENDING — filled by the live run>`. This page defines *what* will
be measured, *how*, and *what must be pinned* so the eventual receipt is apples-to-apples.

---

## 1. What is measured and why

`fak up` claims that concurrent coding subagents **share warm prefix KV blocks** instead of each
re-prefilling the shared context. `docs/subagents-guide.md` states this directly:

> "`fak up` serves an OpenAI-compatible endpoint where concurrent workers share warm prefix KV
> blocks for instant token generation." — *In-Kernel Prefix Cache Reuse (RadixKV)*.

That claim is the object under test. The question is not "is caching better than no caching"
(trivially yes) but **"does fak's shared-prefix reuse beat the strongest practical OSS
alternative on the same machine, model, quant, and trace?"** The strongest practical reference
on a Mac is llama.cpp's `llama-server`, which already has (a) a prompt-prefix cache and
(b) continuous batching — it is *not* a naive cold-reprefill strawman.

Two facts constrain how the result may be read today:

- The current Mac comparison is **`MODELED`**, not physical: `cmd/fak/macbench_manyagent.go`
  carries `DefaultManyAgentProvenance = "MODELED"`, `IsPhysicalSilicon: false`, and the
  `macbench many-agent --compare-llama` arm is a **projection** grounded in measured
  single-stream rates, with unmodeled `thermal_dvfs_throttling`, `memory_bus_contention`, and
  `queue_sync_jitter`. It is not a receipt for the workload in this document.
- `docs/benchmarks/PREFIX-KV-REUSE-COMPARISON.md` is **`Status: INCOMPLETE`** and, until all its
  required arms are present, "no net-true prefix-reuse winner is claimable."

This document does not resolve either; it defines the live protocol that would.

### Provenance label of results, when produced

- Results from `--live` against real `llama-server` and `fak up` processes on this host are
  **`[SW-VERIFIED]`** by default: they are software executions over HTTP with real model files.
- They are **`[HW-WITNESSED]` only if a physical-place lease is held** for the named appliance
  and the receipt records the physical system state (thermal/clocks/backend) per
  `docs/benchmarks/MOMENT-IN-TIME-PROVENANCE-SPEC.md` §2.2. Absent that lease, do **not** stamp
  `[HW-WITNESSED]`, and do not cite an in-memory/socket-simulated run as physical evidence.

---

## 2. The two live arms

Both arms load the **same on-disk weights**. There are exactly two arms here; the four-arm
apples-to-apples contract (`no_reuse`, `sglang`, `vllm`, `fak`) in
`docs/benchmarks/SUBAGENT-FANOUT-APPLES-TO-APPLES-HARNESS.md` is a superset and is not
overridden by this page.

| Arm | Identifier | System | Prefix cache under test |
|:---:|:---|:---|:---|
| 1 | `llamacpp` (reference) | llama.cpp `llama-server` | native prompt-prefix cache + continuous batching |
| 2 | `fak` (candidate) | `fak up` (in-kernel RadixKV / `ctxmmu`) | shared-prefix KV reuse across concurrent subagents |

Pinned model (identical both arms — enforced, not asserted):

```text
~/Library/Caches/fak-models/hub/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf
```

### Arm 1 — llama.cpp `llama-server` (reference), port 8081

```bash
# Pinned build is recorded by `llama-server --version` (see §4 provenance).
llama-server \
  --model ~/Library/Caches/fak-models/hub/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8081 \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --parallel 8 \
  --cont-batching
```

Notes: `--parallel 8` gives one server slot per canonical N=8 subagent; the reference keeps its
**prompt-prefix cache enabled** (this is the tuned OSS baseline — see §6 strawman warning).
Record the exact `llama-server --version` string in the receipt.

### Arm 2 — `fak up` (candidate), port 8082

```bash
fak up \
  --model ~/Library/Caches/fak-models/hub/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf \
  --addr 127.0.0.1:8082 \
  --headless
```

Notes: `fak up` defaults to `--addr 127.0.0.1:8080`; the explicit `--addr 127.0.0.1:8082` keeps
the two arms on disjoint ports (8080 stays free for a normal interactive `fak up`). `--model`
accepts a raw `.gguf` path override alongside the tier aliases (`7B`, `27B`, …). Verify liveness
with `curl http://127.0.0.1:8082/healthz` before starting a live cell.

---

## 3. Frozen trace geometry

Geometry is frozen as mandated by the apples-to-apples contract
(`docs/benchmarks/SUBAGENT-FANOUT-APPLES-TO-APPLES-HARNESS.md` §3.4), restricted to the
Mac-scale N sweep:

| Parameter | Symbol | Value | Meaning |
|:---|:---:|:---:|:---|
| Shared root prefix | `P` | `4096` tokens | coordinator system prompt + tool schemas + repo orientation |
| Private subagent suffix | `S` | `512` tokens | per-subagent task leaf |
| Decode tokens | `D` | `64` tokens | generated per subagent |
| Fanout sweep | `N` | `1, 4, 8` | concurrent subagents |
| Trials per cell | `T` | `5` | repetitions for variance control |

Exact live invocation (writes the receipt into the owned receipts directory):

```bash
go build -o /tmp/fak-dev ./cmd/fak-dev

/tmp/fak-dev bench-subagent-fanout --live \
  --model "Qwen2.5-Coder-3B-Instruct-Q4_K_M" \
  --quant "Q4_K_M" \
  --prefix-tokens 4096 \
  --suffix-tokens 512 \
  --decode-tokens 64 \
  --fanout 1,4,8 \
  --trials 5 \
  --arms fak \
  --fak-url http://127.0.0.1:8082 \
  --json \
  --out docs/benchmarks/receipts/mac-subagent-cache-reuse/m3pro.json
```

The `llamacpp` reference arm is exercised on port 8081 with the same harness geometry and the
same flags. Issue fak#13023 added a **first-class `llamacpp` arm** to the harness (selectable via
`--arms llamacpp --llamacpp-url <url>`); it is recorded under its own identity in the receipt and
is **never** placed in the `no_reuse` slot and never relabeled `sglang`/`vllm`. Its `llama-server`
prompt-prefix cache and continuous batching are left **on** (llama.cpp defaults), so this is the
tuned reference, not the disabled-cache diagnostic baseline.

---

## 4. Provenance to pin

Every cell in the receipt must be condition-bounded on the following tuple; any cell that
changes one of these is not comparable (`MOMENT-IN-TIME-PROVENANCE-SPEC.md` §2.2, §4.1):

| Field | What to pin |
|:---|:---|
| Model | `Qwen2.5-Coder-3B-Instruct` (exact filename above) |
| Quantization | `Q4_K_M` (bit-identical both arms) |
| llama.cpp build | full `llama-server --version` string (+ commit if printed) |
| fak commit | git SHA of the `fak` binary under test |
| Trace digest | content hash of the exact `P`/`S`/`D` prompt+decode trace |
| Concurrency | `N` and the server slot/parallel setting per arm |
| Warmup | warmup requests + cache-state policy before the measured trials |
| Host state | machine id, macOS build, backend (`metal`/`cpu-ref`), thermal/clock if available |
| Sample count | `T` (and per-cell distribution or explicit `N=1` outlier warning) |

---

## 5. Results

**The `fak` arm cells remain `<PENDING>`.** The `llamacpp` (OSS reference) arm has
produced a **real, live** baseline on this M3 Pro (2026-09-14); the `fak` arm was
blocked (see below) and no number is estimated, projected, or extrapolated for it.

### 5.0 What is measured vs. what is accounted (read before the table)

Two columns in the receipt are **not** live measurements and must not be read as such:

- **`reused_tokens` / `prefix_hit_rate` are ACCOUNTING, not observation.** The harness
  computes `reused = (N-1)*P` arithmetically for every non-`no_reuse` arm
  (`cmd/fak-dev/bench_subagent_fanout.go`, live path) without querying server cache
  stats; the same value is produced for any prefix-caching arm. It states the
  *expected* geometry, not a witnessed cache hit. Treat it as a prediction to be
  confirmed by the server's own cache counters, not as evidence.
- **`output_equivalence` / `output_hash` are likewise not witness-grade** in the current
  harness (pre-existing): equivalence is not compared across arms in the live path.
  Do not cite output equivalence from this receipt. Filed as a follow-up
  (see §8 cross-references).

Only **TTFT (p50/p95), decode tok/s, and wall-clock** are live-measured per cell.
Note that `decode tok/s` is computed as `1000/itl_mean * N`, so it already carries the
fanout multiplier — a rise from N=1 to N=4 is partly that multiplier, not purely a
per-request speedup.

**Geometry note.** The frozen protocol in §3 mandates D=64, N∈{1,4,8}, T=5. The
first live run reported below used a narrower pilot geometry (D=16, N∈{1,4}, T=1)
to bound wall-clock under a busy box; it is labeled as a **pilot** and is not a
contract-satisfying cell. The full-geometry receipt remains `<PENDING>`.

### 5.1 Live OSS reference pilot — `llamacpp` `[SW-VERIFIED]`

Measured against a real `llama-server` (llama.cpp, Metal) on
`Qwen2.5-Coder-3B-Instruct-Q4_K_M`, port 8081, `--parallel 8`. Pilot geometry
P=4096, S=512, D=16, N∈{1,4}, T=1 (single trial — distribution not tightened;
not yet the §3 contract geometry).

| arm | N | TTFT p50 ms | TTFT p95 ms | decode tok/s | reused tokens (accounting) | prefix hit (accounting) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `llamacpp` | 1 | 86.7 | 86.7 | 57.4 | 0 | 0.00 |
| `llamacpp` | 4 | 175.5 | 176.3 | 92.4 | 12,288 | 0.667 |
| `fak` | 1 | `<PENDING>` | `<PENDING>` | `<PENDING>` | `<PENDING>` | `<PENDING>` |
| `fak` | 4 | `<PENDING>` | `<PENDING>` | `<PENDING>` | `<PENDING>` | `<PENDING>` |
| `fak` | 8 | `<PENDING>` | `<PENDING>` | `<PENDING>` | `<PENDING>` | `<PENDING>` |

Committed receipt: `docs/benchmarks/receipts/mac-subagent-cache-reuse/m3pro.json`.
The N=4 `reused 12,288 = (4-1)*4096` and `prefix hit 0.667` are the harness's
**accounting expectation** (§5.0), not a witnessed llama.cpp cache hit — the
cross-agent shared-prefix behavior that fak must beat, still to be confirmed
against the server's own cache counters.

### 5.2 `fak` arm status — endpoint gap RESOLVED; one environmental blocker remains (honest, not a number)

Two blockers were found while attempting the live `fak` arm. Blocker 2 is now resolved;
blocker 1 is environmental. Neither was ever a performance claim:

1. **Peer Metal lease.** A concurrent peer `fak serve` (27B, Metal) held the GPU
   lease `/var/folders/.../T/fak-gpu.lease`, so `fak up` correctly refused with
   `Metal residency admission refused ... lease held by pid <peer>`. Killing a
   peer is out of policy; the 3B `fak up` was therefore not run on Metal.
2. **Endpoint-path gap (filed as fak#13027) — RESOLVED.** The harness posts live cells to
   `/v1/completions` (`cmd/fak-dev/bench_subagent_fanout.go`). `fak serve` always exposed
   that route (`internal/gateway/completions.go`); the gap was the turnkey `fak up` server,
   which served only `/v1/chat/completions` and returned `404 page not found`. That half
   landed in `bf099b20a` (`cmd/fak/up.go`), so `fak up` now serves `/v1/completions` too —
   same native `StreamingPlanner` per-token path, only the JSON envelope differs. The live
   fak arm is no longer blocked by the endpoint path; blocker #1 (peer Metal lease) still
   governs whether a run may execute on a shared host.

Receipt path (once produced): `docs/benchmarks/receipts/mac-subagent-cache-reuse/m3pro.json`.

---

## 6. Honest verdict — how to read a lift / no-lift

Read the result against these rules, decided **before** the run:

- **A "lift" is fak beating the tuned reference**, i.e. lower TTFT p50/p95 or higher
  decode tok/s at equal-or-lower peak memory, **at the same N**, with both arms sharing model,
  quant, trace, concurrency, warmup, and host state. A win at N=1 is not a cache-reuse result;
  the claim is about *cross-agent* reuse, so the signal should appear (or grow) as N rises.
- **A "no-lift" is a valid, publishable outcome.** If `llamacpp` matches or beats `fak`, say so
  plainly and attribute it to the measured mechanism — do not soften it, and do not re-run until
  a favorable number appears.
- **Ratios are point-in-time and condition-bounded.** Report absolute units (ms, tok/s, MB) as
  the primary figures; any speedup ratio must cite its receipt, date, commit, and conditions and
  must not be amplified into a permanent capability (`MOMENT-IN-TIME-PROVENANCE-SPEC.md` §5).
- **Distribution, not best-case.** With `T=5`, report p50 and p95 (not the single best sample);
  flag any `N=1` cell with the mandated outlier warning.

### Strawman warning (mandatory)

The baseline is the **tuned OSS prefix cache** (llama.cpp `llama-server` with its prompt-prefix
cache and continuous batching **enabled**) — **not** a naive cold re-prefill stack. Comparing
fak's cache against a cache-disabled arm would manufacture a large, meaningless ratio (the
`180.97×`-on-a-degraded-baseline failure class the provenance spec exists to prevent). If a
disabled-cache arm is reported at all, it must be labeled the **degraded diagnostic baseline**,
never the comparison baseline. Do not attribute a blended compound effect (cached + batched +
concurrent) to caching alone; isolate one lever at a time.

---

## 7. Repro commands

```bash
# 0. Build the harness
go build -o /tmp/fak-dev ./cmd/fak-dev

# 1. Pinned model (identical both arms)
MODEL=~/Library/Caches/fak-models/hub/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf

# 2. Reference arm: llama.cpp llama-server (port 8081), prefix cache ON
llama-server --version
llama-server --model "$MODEL" --host 127.0.0.1 --port 8081 \
  --ctx-size 8192 --n-gpu-layers 99 --parallel 8 --cont-batching

# 3. Candidate arm: fak up (port 8082)
fak up --model "$MODEL" --addr 127.0.0.1:8082 --headless
curl -sf http://127.0.0.1:8082/healthz

# 4. Live sweep -> receipt (geometry P=4096 S=512 D=64, N in {1,4,8}, T=5)
/tmp/fak-dev bench-subagent-fanout --live \
  --model "Qwen2.5-Coder-3B-Instruct-Q4_K_M" --quant "Q4_K_M" \
  --prefix-tokens 4096 --suffix-tokens 512 --decode-tokens 64 \
  --fanout 1,4,8 --trials 5 \
  --arms fak --fak-url http://127.0.0.1:8082 \
  --json --out docs/benchmarks/receipts/mac-subagent-cache-reuse/m3pro.json

# 5. Fill §5 from the receipt only; stamp [SW-VERIFIED] (or [HW-WITNESSED] under a lease).
```

---

## Cross-references

- Four-arm apples-to-apples contract: [`SUBAGENT-FANOUT-APPLES-TO-APPLES-HARNESS.md`](SUBAGENT-FANOUT-APPLES-TO-APPLES-HARNESS.md)
- Incomplete prefix-reuse contract: [`PREFIX-KV-REUSE-COMPARISON.md`](PREFIX-KV-REUSE-COMPARISON.md)
- Claimed shared-prefix behavior: [`../subagents-guide.md`](../subagents-guide.md)
- Moment-in-time provenance & anti-amplification: `MOMENT-IN-TIME-PROVENANCE-SPEC.md` (fak-private)
- Mac many-agent cache-value background (measured, separate workload): [`../notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md`](../notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md)
- Modeled Mac comparison source: `cmd/fak/macbench_manyagent.go` (`DefaultManyAgentProvenance = "MODELED"`)
- fak#13023 — this head-to-head (arm + receipt + this README)
- fak#13027 — `/v1/completions` gap on the turnkey `fak up` server, blocking the live fak arm; the `fak up` half is RESOLVED in `bf099b20a` (`fak serve` always served the route), so only the peer-lease blocker remains
- fak#13029 — harness reports accounted reuse / output equivalence as if measured (honesty fix)
