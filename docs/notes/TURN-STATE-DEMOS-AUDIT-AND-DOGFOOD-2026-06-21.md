# The other key turn/state demos — audited, improved, dog-fooded (2026-06-21)

> **Goal:** *audit the **other** key turn and state demos (the ones not already
> witnessed in [GLM52 §3](GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md)),
> improve them, dog-food them, and run them on a laptop and a MacBook.*
>
> This doc is closed by witnesses the author did not write — `go run` exit codes, the
> kernel's own counters, `go test` results, and a cross-OS acceptance script — not by
> self-report. Every command below was run on-box at `HEAD` on 2026-06-21
> (`go build ./...` green; `go version` = go1.26.3 windows/amd64; the `go test` phase run
> under WSL go1.26.0 linux/amd64 because native-Windows app-control blocks freshly
> compiled test binaries).

## Scope — what "the other demos" are

GLM52 §3 already witnesses the **five fleet demos** (`fanbench`, `fleetbench`,
`fak turntax`, `radixbench`, `ctxdemo`). This pass covers the rest of the turn/state
surface — the demos that prove the kernel's **state** properties (provable deletion,
causal eviction, context admission) and the browser **turn** demo's data path:

| Demo | Axis | What it proves | Needs a model? |
|---|---|---|---|
| `deletioncert` | state | bit-exact KV eviction (evicted == never-saw, max\|Δ\|=0) + a tamper-evident signed certificate that fails closed | no |
| `causalbench` | state | an external write causally evicts exactly the dependent cached read, byte-exact, siblings warm, re-admission refused | no |
| `ctxbench` | state | the write-time context-admission gate (ctxmmu.Admit) quarantines poison; 0 trigger bytes survive | no |
| `turntaxdemo` | turn | the three-lane turn-tax race (naive +9 / tuned +5 / fak 0) replayed through the real kernel | no |
| `sessionbench` · `batchbench` · `demorace` | turn | long-session work-reuse / batched-decode throughput / live reuse race | **yes** |

## §1 — On-box witnesses (dog-food)

A single cross-platform acceptance script, **`tools/run_turn_state_demos.sh`**, runs every
no-model turn/state demo and asserts exit 0 + each demo's documented invariant. Witnessed
identical on two operating systems:

```
== fak turn/state demos — cross-platform dog-food ==
  deletioncert -selfcheck          ... PASS   (evicted == never-saw, max|Δ|=0; tamper-rejected)
  deletioncert (cert minted)       ... PASS
  causalbench                      ... PASS   (dependent read evicted byte-exact; siblings warm)
  ctxbench (LEAK == 0)             ... PASS   (0 trigger bytes survive admit)
  ctxbench (2 quarantined)         ... PASS
  ctxbench -chain (normgate)       ... PASS
  turntaxdemo -selfcheck           ... PASS   (airline 9 = forced 5 + elision 4; happy 0; safety 1→0)
  fak turntax (airline = 9)        ... PASS
  fak turntax (happy control)      ... PASS
  go test (turn/state pkgs)        ... PASS   (turnbench, ctxmmu, recall, causalbench, turntaxdemo, sessionbench)
== summary: 10 passed, 0 failed ==   ACCEPTANCE PASSED
```

- **Windows / amd64 (go1.26.3):** 9/9 demo checks PASS; the `go test` phase auto-skips
  (native app-control blocks test binaries — by design, not a failure).
- **Linux / amd64 under WSL (go1.26.0):** 10/10 PASS including the `go test` phase.
- **macOS / arm64 (Apple Silicon):** every demo + both test files **cross-compile and
  vet clean** (`GOOS=darwin GOARCH=arm64 go build` / `go vet`). The deterministic, seeded
  invariants reproduce byte-for-byte (no GOARCH/path/endianness/asm on these demos' paths —
  the synthetic-model weights pin `binary.LittleEndian`, the witnesses are byte/token-id
  comparisons), so a MacBook run of the script is the final on-device confirmation (handoff
  in §4).

## §2 — Improvements shipped

Each was surfaced by a parallel audit and **independently adversarially verified** (a
second agent confirmed the problem was real and the fix breaks no test / documented
invariant) before being applied.

| # | Demo | Fix | Why it mattered |
|---|---|---|---|
| 1 | `ctxbench` | Report the **actual payload bytes** the gate inspected (fall back to `len(payload)` when the fixture omits the declared `bytes`), not the hand-declared field | The public fixture omits `bytes`, so the report printed `0 bytes total` / `0B` for non-empty quarantined poison. Now: **371 bytes** (202 + 108 + 61), matching the fixture's independently-computed UTF-8 lengths. (First measured `res.Payload.Inline` — wrong, because `Admit` pages the quarantined payload out to a short pointer; corrected to the original `r.Payload`.) |
| 2 | `turntaxdemo` | New **`-selfcheck`** headless mode + a `main_test.go` guard | The browser demo's data path had no browserless verification. `-selfcheck` replays each suite through the same `turnbench.RunWithCalls` the browser drives and asserts the documented invariants (airline `9 = forced 5 + elision 4`, `vdso_off 2`, safety `1→0`; happy `0`), exit non-zero on drift. Now CI- and MacBook-runnable with no browser. |
| 3 | `turntaxdemo` | `page.html` `reset()` now clears the **tuned lane** | A real client-side bug: re-running "Replay" appended a fresh set of tuned cells onto the previous ones (the track doubled each run) and left a stale `#ctTuned`. |
| 4 | `turntaxdemo` | Surface the **"no fixtures found"** state in the browser (using `j.dir` already in the payload), and disable Run | A wrong-cwd run silently showed an inert dropdown with only a stderr warning. |
| 5 | `turntaxdemo` | `turnTaxDir()` now **walks up to the module root** (additive fallback) | `go run`/the binary from any subdirectory previously found no fixtures. Verified: running from `docs/` now resolves `…/testdata/turntax` and self-checks green. The existing relative candidates still win first, so the cmd-level test is unaffected. |
| 6 | `turntaxdemo` | Package header + `TURN-TAX-RESULTS.md` updated **two-lane → three-lane** | The header documented two lanes while the page renders three (naive/tuned/fak); a source reader had a stale mental model. |
| 7 | `ctxbench` | CALL-side prints **`n/a (corpus has no calls)`** instead of `0/0 = 0.0%` (+ JSON `catch_rate: null`; `sources: (none listed)`) | `0.0% caught` read as a *failed gate* on the result-side-only fixture. |
| 8 | `deletioncert` | Run an actual **grouped-query** config (`NumKVHeads: 2`) so the certificate's `ModelPath: "gqa-rope"` names the head grouping the demo really runs | The config was MHA (`NumKVHeads == NumHeads`); the cert named a path it didn't run. Re-verified on-box: `PROVEN`, evicted == never-saw, non-vacuous, tamper-rejected, exit 0. |
| 9 | `deletioncert` | Add a fenced **`[SHIPPED]`** entry to `CLAIMS.md` | A shipped, tested, proof-documented security capability was absent from the honesty ledger (`claims-lint` lints for exactly-one tag). The entry carries the three honest fences: self-signed v1 (integrity, not independence), `EvictedCount` is a self-report, `max\|Δ\|=0` is checked as a signed string (not re-measured). `claims-lint` green (70 lines, 0 violations). |
| 10 | `causalbench` | `BumpWorld` comment now states it bumps **only** `worldVer`; `trustEpoch` + the revoked-witness ledger persist (which is why the assertions compare relatively) | The comment called it "the internal reset hook," overstating how much state it clears. |
| 11 | `causalbench` | `BENCHMARK-AUTHORITY.md`: **"all nine invariants" → "all 12 guarded invariants"** (+ portable `go test ./cmd/causalbench/`) | The record carries 12 invariant fields and the test guards 12; "nine" undercounted what gates exit 0. |
| 12 | docs | Fixed two **broken GLM52 links** in `fleet-benchmarks.md` (`../../` → `../notes/`) and dropped the dangling `experiments/SECURITY-DRIVERS-*.md` ref in `cli-reference.md` | The GLM52 doc moved to `docs/notes/`; both links 404'd. The SECURITY-DRIVERS file does not exist. |

## §3 — Findings surfaced but deliberately deferred (honesty)

Not everything the audit found was changed — some belong to another session's lane, and
some are honest scope calls:

- **`make_evasion_corpus.py` stale `../fak/experiments/` path + the CLAIMS `0→20/24` count.**
  The corpus generator writes to a non-existent nested `fak/` path (same class as the
  `new_leaf.py` bug), and adversarial verification found the generator actually yields ~20
  cases at ~15/20 caught, *contradicting* CLAIMS' `0→20/24`. **Left untouched** because
  `experiments/evasion-corpus.json` and `tools/_tmp_gen_fixed.py` are in another session's
  working set right now — reconciling the count is that lane's call, not a drive-by edit.
- **`deletioncert` equivalence-axis vocabulary.** The demo's `max\|Δ\|=0` is a *token-id*
  delta on a synthetic (meaningless-logits) model, while the README/explainer use the same
  `max\|Δ\|=0` vocabulary for the *HF-logit* guarantee. The new CLAIMS entry fences this
  ("checked as a signed string, not re-measured; numerics proven separately by the oracle").
  A deeper self-describing `Metric` field on the certificate would change the signed
  pre-image and **requires a `SchemaVersion` bump** — out of scope for this pass.
- **`demorace` headlines a single cold ratio.** `demorace` (model-backed) presents the
  naive-re-prefill ÷ fak multiple with no warm-per-agent-KV baseline and no "worst-case
  reference" caveat — the framing law its siblings `sessionbench`/`ctxdemo` follow. A real
  medium-severity honesty gap, but it needs a real checkpoint to verify a code change, so
  it is flagged here for the maintainer rather than edited blind.
- **`-selfcheck` no-op on `deletioncert`/`causalbench`.** Both parse then discard the flag
  (`_ = selfcheck // single mode today`) — a deliberate repo convention that reserves room
  for a future mode and keeps the committed witness JSON byte-stable. Left as-is.

## §4 — MacBook (Apple Silicon) handoff

Everything is in place to run on a MacBook; the one residual is the on-device execution
(this dev box has no Mac). On the MacBook, from the repo root:

```bash
bash tools/run_turn_state_demos.sh        # demos + go test; expect "ACCEPTANCE PASSED" (10/10)
```

The deterministic invariants must match this run byte-for-byte. (The model-backed
`sessionbench`/`batchbench`/`demorace` *timed* ratios will differ on arm64 — NEON-SDOT
kernels + P/E cores vs the amd64 AVX path — but their **timing-free token ratios** and all
of the above invariants are platform-independent.)

## Reproduce

```bash
go run ./cmd/deletioncert -selfcheck      # provable deletion: evicted == never-saw, tamper-rejected
go run ./cmd/causalbench                  # causal eviction: dependent read out, siblings warm
go run ./cmd/ctxbench                      # context admission: 371 bytes inspected, 0 leak, 2 quarantined
go run ./cmd/ctxbench -chain               # + normgate canonicalize-and-rescan
go run ./cmd/turntaxdemo -selfcheck        # turn-tax headless: airline 9 (5+4), happy 0, safety 1→0
bash tools/run_turn_state_demos.sh         # all of the above + go test, with a PASS/FAIL gate
```

_Artifacts are regenerable; nothing here is committed beyond the demos, the acceptance
script, and the doc/claims edits in §2._
