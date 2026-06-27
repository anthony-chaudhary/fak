# STATUS — fak v0.2.1: proven with DOS concepts

> The deliverable's whole point: this status is **not a self-report**. Every line
> below is closed by a witness the author did not write — a `go` exit code, a
> benchmark field, a git tag, or the DOS truth syscall reading git ancestry.
>
> **Product standing** (which concepts a person can pick up and use today, and what's
> next): [`docs/PRODUCT-STATUS.md`](docs/PRODUCT-STATUS.md) — 10 durable products, 100%
> concept-catalog coverage, cross-checked against the tree by `tools/product_scorecard.py`.

## 0. 2026-06-18 benchmark/status refresh

The benchmark front door is now
[`../VISUALS-benchmarking-status-2026-06-18.md`](docs/notes/VISUALS-benchmarking-status-2026-06-18.md):
it collects the refreshed plot deck plus the current overall read. The plots were
regenerated from checked-in CSV/JSON with `tools/fanout_plot.py`,
`tools/fleet_heatmap.py`, `tools/fleet_compare.py`, and `tools/fleet_eraser.py`.

Current benchmark read:

- **Fleet sweep:** the read-fleet 50x50 corner deletes 2,344/2,500 calls, with
  **+370** cross-agent turns over isolated worlds; no-share controls are exactly
  zero.
- **Write invalidation:** global invalidation turns negative around 1% writes,
  while resource-scoped invalidation keeps **+313** uplift at 1% writes and
  **+235** at 10% writes.
- **Fan-out:** at N=1024, the shared fan-out path has **+1,005** sibling-only
  tool-result saves; the prefix-cache model claws back **61.7%** of the
  multi-agent token tax and exposes the fold-bound latency knee.
- **Realistic workload:** the transcript-derived profile is **82 sessions / 952
  logical turns**, with median prefix **4,671** tokens and **94.4%** tool-call
  turns. This keeps the benchmark shape tied to observed agent sessions rather
  than only synthetic constants.

Fresh verification pass on 2026-06-18:

| Witness | Result |
|---|---|
| `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ci.ps1` | **PASS** — `go build`, `go vet`, `go test ./...`, and claims lint all green; claims lint found 54 tagged claim lines, 0 violations. |
| `python -m pytest tools` | **PASS** — 289 Python/tooling tests passed. |
| `go run ./cmd/fak bench --suite tau2-smoke --out report.json` | **PASS as a subsystem sentinel** — `gate_primary=pass`; in-process p50 2,427 ns; spawned-hook p50 6,913,458 ns (n=100); p50 speedup about 2,849x; vDSO hit-rate 0.500. It proves the adjudicator path is not accidentally paying a per-call process boundary; it does not prove production readiness. |
| `python tools\fak_phase0_gate.py fak\experiments\fleet-nodes\phase0-local-uncapped --json` | **FAIL, correctly open** — no clean-node provenance and peak batched-decode speedup is 40.975x, below the 45x Phase 0 bar. |
| `go run ./cmd/modelbench -backend cpu-ref -require-non-reference ...` | **FAIL, correctly closed** — `cpu-ref` is rejected as a reference backend for Phase 1. |
| `go run ./cmd/paritybench ... --require-phase1` | **FAIL, correctly open** — missing live local-GPU 7-9B rung. |

## 1. The truth syscall (`dos_verify`) — shipped from evidence, not say-so

Both phase commits were confirmed by the DOS kernel's `dos_verify` against the real
git history of `C:\work\fleet` (source `grep-subject`, rung `direct`):

| Phase | `dos_verify` verdict | sha | interpretation |
|---|---|---|---|
| `fak abi-v0.1` | **shipped: true** | `6be91d4` | "confirmed by evidence … not self-reported" |
| `fak v0.1.0` | **shipped: true** | `c72ddf1` | "confirmed by evidence … not self-reported" |

`dos_commit_audit` on both commits returned **non-forgeable diff evidence**: the
`v0.1.0` commit touched **28 source files incl. 12 `_test.go` files**; the
`abi-v0.1` commit touched the 5 ABI source files + the golden. (Both `ABSTAIN` on
the *subject* grammar — the commit messages don't use this workspace's ship-stamp
grammar — but the diff witness proves they are real code commits, not empty/README
stamps. The honest distinction the truth syscall is built to make.)

**The v0.2.x line (tags `v0.2.0`, `v0.2.1`) extends this — with a caveat the truth
syscall itself surfaces, recorded here rather than hidden.** Running `dos_verify fak
v0.2.0` / `v0.2.1` against this repo today returns **`shipped:false, source:none`**:
the ship oracle finds the tagged commits (`a8b10c3`, `3c2a1eb`) but **demotes** them as
*release-bump* commits — the `/release` skill stamps the version on its own commit,
which carries no ship-stamp grammar and touches only `VERSION` + release notes (#399).
That demotion is **correct, not a regression**: the *code* of each release ships across
the commits the bump caps (29 for v0.2.0, 4 for v0.2.1), and the per-lane ship evidence
is each lane's own witnessed results doc (`MODEL-BASELINE-RESULTS.md`,
`RECALL-RESULTS.md`, `CDB-RESULTS.md`, `KV-QUARANTINE-BRIDGE-RESULTS.md`,
`TURN-TAX-RESULTS.md`) plus the green `go test ./...` at HEAD. A truth syscall that
refused to credit a version-bump commit as a code ship is doing precisely its job.

## 2. Syscall subsystem check — useful, not the product KPI

`fak bench --suite tau2-smoke` → current `report.json`:

```
in-process adjudication p50 : 2,427 ns
spawned-hook        p50     : 6.913 ms  (process-per-decide, this machine, n=100)
SUBSYSTEM CHECK (gate_primary): pass
boundary-tax delta            : ~2,849x    (varies with machine load; always >>1)
```

What it proves: the syscall/adjudicator path is resident and not accidentally
shelling out through `fak hook` on every decision. That is a useful regression
sentinel for the reference-monitor subsystem.

What it does **not** prove: production readiness, model quality, real serving
throughput, the 45x fleet claim, or a win over a long-lived policy sidecar. The
spawned baseline is intentionally a worst-case boundary-tax control; an
in-process function beating a process spawn is expected. The production gates
remain Phase 0 clean-node reproduction and Phase 1 non-reference backend plus
7-9B local-GPU evidence.

The vDSO hit-rate (~0.5 on the cache-favorable demo trace; ~0.7% addressable on
real tau2-airline) and the 47% token delta are reported as **soft secondaries**,
never production gates (`report.json` `token_delta_pct`, `kpis.vdso_hit_rate`).

## 3. The witness set (`scripts/ci.ps1` / `make ci`)

| Witness | Result |
|---|---|
| `go build ./...` | exit 0 |
| `go vet ./...` | exit 0 |
| `go test ./...` | **all packages green, 200+ test functions** (`internal/model`, `internal/turnbench`, and the cache metadata/radix/recall lanes carry the heavy or newly refreshed paths) |
| `claims-lint.ps1` | 54 claim lines, **0 violations** |
| ABI golden freeze (`TestABIGoldenFreeze`) | green (the additive-only freeze is machine-checked) |
| `report.json` / `baseline.json` | refreshed in this worktree, `gate_primary == "pass"` for the syscall subsystem check, `baseline p50 > 1ms` |
| `go test -race ./...` (cgo) | **0 data races**, full suite green — runs in the `race-detector` CI job (E-001 / issue #12) |

> Race-detector caveat (honest): `go test -race` requires cgo + a C compiler. The
> canonical Windows dev box has `CGO_ENABLED=0` and no gcc/clang, so the detector
> still cannot build there — it runs **uninstrumented** on that box only. It now
> executes wherever cgo is available: the `race-detector` CI job (`ubuntu-latest`)
> runs `go test -race -count=1 -timeout=25m ./...` on every push/PR, and it can be
> run locally via WSL/Linux/macOS (see `../docs/testing/race-detector.md`). Under
> instrumentation the full suite is race-clean: **0 data races detected**.

## 4. DOS concepts used to build AND prove it

| DOS concept | Where it shows up in fak |
|---|---|
| **witness-grounded adjudication** | `dos_verify` over both ship commits; the whole "prove not self-certify" discipline |
| **structured refusal (closed vocabulary)** | fak's closed 12-reason `ReasonCode` mirrors `dos_refuse_reasons`; **`SELF_MODIFY`** appears in BOTH (DOS: "a live loop must not rewrite the kernel adjudicating it") and is enforced by fak's adjudicator (`SelfModifyGlobs`) + the shipgate's worktree isolation |
| **bounded-disclosure witness** | a `SELF_MODIFY` deny returns only the offending glob (the unsat-core move) |
| **lease arbitration / plan-price** | `partition-price.json` (collisions=0, disjoint by construction) + `lease-ledger.json`; `dos_arbitrate` demonstrated the reactive collision floor (refuses a contended lane, surfaces free ones) |
| **deny-as-value** | a refusal carries a derived disposition (RETRYABLE/WAIT/ESCALATE/TERMINAL) the loop consumes |
| **context-MMU / KV checkpoint** | write-time result quarantine + page-out to a pointer; the addressable-`Ref` seam is the zero-copy/checkpoint future |
| **RSI as ship-gate** | keep-or-revert on a non-forgeable keep-bit, candidate applied in an isolated git worktree, escalation breaker |
| **adversarial verification** | a 7-skeptic workflow re-checked each headline claim from raw evidence (§5) |

## 5. Adversarial verification (independent skeptics)

7 independent read-only (Explore) skeptic agents each tried to REFUTE one headline
claim by reading the real code and running the real commands. Their default was
REFUTED unless their own evidence confirmed it. **Result: 7/7 CONFIRMED.**

| Claim | Verdict | Decisive evidence the skeptic gathered |
|---|---|---|
| C1 syscall subsystem A/B is real + apples-to-apples | **CONFIRMED** | re-ran bench: 1295 ns vs 7,358,600 ns = **5,682×** (an earlier ad-hoc re-run; the canonical refreshed boundary-tax is ~2,849× at n=100 — see `BENCHMARK-AUTHORITY.md`), `gate_primary="pass"`; verified both arms call `kernel.Fold(abi.Adjudicators())` and the comparison is computed (`on < base`), not hardcoded. Scope: subsystem boundary-tax check, not product throughput. |
| C2 deny never reaches dispatch | **CONFIRMED** | `TestDenyNeverReachesDispatch` PASS: engine `n==0` on deny, `Meta["disposition"]` set; Reap returns `DenyResult` before the engine call |
| C3 MMU quarantines the poison fixture | **CONFIRMED** | manually admitted all 3 `poison.json` payloads: injection + secret → `Quarantine` (rewritten payload contains **zero** offending bytes), benign → `Allow` |
| C4 no `os/exec` on the hot path | **CONFIRMED** | kernel imports only `{context,errors,fmt,sync,sync/atomic}`; `os/exec` appears only in `bench`+`shipgate` (not the dispatch path); `TestNoOsExecOnHotPath` PASS |
| C5 full suite green | **CONFIRMED** | `go build`/`go vet` clean; **200+ test functions all PASS** across 30 packages; `claims-lint: 0 violations` |
| C6 vDSO soundness + invalidation | **CONFIRMED** | canonicalized keys (reordered args hit), `worldVer` bump invalidates stale reads, tier-2 hit == fresh call (`TestUnit38_Soundness…` PASS) |
| C7 CLAIMS.md honesty ledger accurate | **CONFIRMED** | 4 `[SHIPPED]` claims have backing code+tests; 3 `[STUB]/[SIMULATED]` (zero-copy KV, decode-time mask, metrics-service) confirmed NOT on the critical path |

The single caveat (C3): the in-suite fixture test exercises the injection + benign
payloads explicitly and secrets via a separate test; the skeptic manually confirmed
the `secret_leak` payload also quarantines. No claim was refuted or downgraded.

The v0.2 lanes each carried their **own** skeptic pass on the same default-refute
discipline: recall **5/5 CONFIRMED**, cdb **5/5**, the KV-quarantine bridge 3/5→fixed→
green, and MODEL-BASELINE's numbers survived a **4-skeptic** pass (two methodology
defects caught and fixed, not papered over). The SECURITY-BENCHMARKS run went the other
way on purpose — 9 independent agents *re-derived* the detector's ~100% evasion rate and
confirmed it, which is why detection is reported as non-load-bearing.

## 6. Honest residue (see `CLAIMS.md`)

> Live update (2026-06-17): `fak agent` drives this kernel with a **real model**
> (Gemini OpenAI-compat + local Qwen2.5) over a turn-counting A/B — see
> `LIVE-RESULTS.md` (turns ≈ equal on the happy path; the win is the deterministic
> injection-quarantine floor) and `TICKETS.md` for the surfaced issues.
>
> **v0.2 grew four more witnessed organs** on the v0.1 syscall skeleton, each with its
> own results doc + adversarial pass: a **real model fused into the kernel** (pure-Go
> SmolLM2-135M, proven bit-for-bit vs HF, then parity-fast incl. an int8 SIMD lane —
> `MODEL-BASELINE-RESULTS.md`); a **security substrate** (ifc / provenance / plan-CFI /
> witness / normgate — the kernel stops believing the model); a **gateway** (`fak serve`,
> OpenAI + MCP); and a durable **session core-dump + debugger** (`recall` / `fak debug` —
> a quarantine that survives the process boundary). The honest security finding the
> substrate forced is in a private transcript-derived security benchmark:
> the architecture is sound (0 leaks after quarantine) but the *detector* it inherits is
> ~100% evadable and FP-prone — so the load-bearing guarantee is the capability floor +
> containment, not detection (which `normgate` improves but does not make a guarantee).

What is STILL deferred (labeled, not hidden): no LIVE transport mapping an *external*
serving engine's KV region into the now-shipped cross-engine co-residence arena (the
zero-copy SEAM itself landed in #448 — `internal/xenginekv`, opt-in `FAK_XENGINE_KV`, the
region-addressed Evict/Clone quarantine behind the frozen `Ref`/`RegionBackend` seam; what
remains is the CUDA-IPC / shared-memory import of a real vLLM/SGLang KV region, a backend
plug-in behind that ABI with no further change); **GPU device compute is now witnessed real** (`cuda` on RTX 4070,
`vulkan` on a Radeon RX 7600 — argmax-exact, cosine 1.0; `GPU.md`, `VULKAN-AMD-RESULTS.md`),
while token-per-watt / metrics-service KV telemetry stays SIMULATED (no power meter on the
box); rung-2/3 probes, decode-time logit-mask, SNAPSHOT/ROLLBACK wrap, and the
fine-tuned *syscall/adjudication* model are STUB (the fused model is a stock reference,
not a tuned adjudicator; `internal/harvest` now folds the verdict stream into its
training corpus, but the model that consumes it is unbuilt). Consistent with the
cluster's 0/29-NOVEL posture (0 of 29 audited prior-art primitives are novel): the contribution is the **assembly** (a fused, fail-open,
witness-gated kernel with the tool call promoted to an in-process syscall), not any
single primitive.
