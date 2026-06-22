# Build your optimization on fak

**For researchers and teams who want to make a fak subsystem faster or smarter — a new
quantization kernel, a GPU / NPU (neural-processing-unit) backend, a cache-eviction
policy, an admission rung, a KV (key/value attention cache) layout — and have it land as
a peer of the built-ins without forking the core.**

If you've ever shipped a clever kernel and then watched it rot because the upstream you
patched moved underneath you, this document is the contract that stops that here. You
attach through a **named registration seam**; the kernel *walks* the registry and never
imports your code. So your optimization (1) survives core refactors, (2) **composes** with
every other optimization at zero hot-path cost, and (3) is **provably correct and provably
faster before it ships** — by a witness the harness checks, not a claim you make.

This is the researcher-facing companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the
extension model in full), the layering gates documented inline below, and the
repo-wide [`CONTRIBUTING.md`](CONTRIBUTING.md) (how to land a change). Read this one
first if your goal is "I want to optimize subsystem X."

---

## TL;DR — the three gates

Every optimization lands the same way. Each gate is mechanical: a tool or a test, not a
review opinion.

| Gate | Question | The mechanism | Run it |
|------|----------|---------------|--------|
| **1. Plug in** | Where does my code attach? | a `Register*` seam + `internal/architest` layering gate | `python tools/new_leaf.py …` / add a backend file |
| **2. Prove correct** | Does it preserve behavior? | the Reference/Approx correctness class + a deterministic witness test | `.\fak\test.ps1 ./internal/<pkg>/` |
| **3. Prove faster** | Is it actually a win? | the non-forgeable keep-bit (`shipgate.Evaluate` via `cmd/rsicycle`) | `go run ./cmd/rsicycle …` |

You do not get to skip a gate. That is the point: a contributor cannot land a kernel that
is plausible-but-wrong (Gate 2 catches it) or correct-but-slower (Gate 3 catches it). The
gates are what make it *safe* to accept optimizations from anyone — including autonomous
coding agents.

**Before you start, run the preflight** — one command tells you whether your environment
and all three gate entry points are wired, and prints this golden path with the exact
commands (add `--json` for a machine-readable answer an agent tool can parse):

```bash
python tools/extend_preflight.py
```

---

## Gate 1 — Plug in (don't fork)

fak is a **frozen minimal spine** (`internal/abi`) plus **real extension seams**. The
spine never changes after freeze; everything else attaches as *a new package + one
`Register*()` call*, never an edit to the core. `internal/architest` fails the build on an
upward/cross-tier import, so the layering can't silently erode.

### Pick your seam

There are two registries, by where your optimization lives.

**A. Device compute / quantization kernels → `internal/compute`** (the hardware
abstraction layer, HAL). This is the seam for the most common research optimization:
*"make this subsystem's math faster on my hardware."* A backend is a new **file** in
`internal/compute/` that self-registers in `init()`:

```go
//go:build mybackend            // build tag gates which backends COMPILE IN; the
package compute                 // registry picks which one RUNS (FAK_BACKEND / Pick).

func init() { Register(myBackend{}) }   // compute.Register — that's the whole wiring.

type myBackend struct{}
func (myBackend) Name() string             { return "my-q4" }
func (myBackend) Class() compute.CorrectnessClass { return compute.Approx } // see Gate 2
// …implement the small whole-op Backend interface (MatMul, Attention, RoPE, Argmax, …);
//   the forward loop targets the interface and never sees your bytes.
```

The CPU reference (`cpuref.go`) and the real `cuda` / `metal` / `vulkan` backends already
ride this seam — read them as your template. The forward loop in `internal/model` calls
*whole ops* (`MatMul`, `Attention`, `Argmax`), so a device is free to express its own
intra-kernel parallelism, async enqueue (`Caps.Async`), fused attention (`Caps.FusedAttn`),
or a tiled layout — and an older loop that doesn't know your capability falls back to the
synchronous core. **You never edit the forward loop.**

**B. Cross-worker policy / engine / cache / verdict → `internal/abi` `Register*`.** If your
optimization is a new admission rung, a serving engine, a zero-copy region backend, a
page-out codec, or a fast-path answer, it's a new **leaf package** that registers against
the frozen ABI from `init()`:

| To optimize / add… | Call | What you get |
|---|---|---|
| a smarter policy / admission rung | `RegisterAdjudicator(rank, impl)` | a new link in the LSM-style decision chain |
| a local fast-path (vDSO-style) answer | `RegisterFastPath(tier, impl)` | a cache hit that never enters the slow path |
| a serving backend (local/remote/multi) | `RegisterEngine(id, impl)` | a new engine behind the selector |
| zero-copy KV co-residence | `RegisterRegionBackend(impl)` | a Resolver swap (copy → shared arena) |
| a swappable page-out / compaction codec | `RegisterPageOutBackend(id, impl)` | a new MMU headroom backend |
| a new verdict kind | `RegisterVerdictKind(k>1023, …)` | an open-range kind with a declared fold position |

Scaffold one with the golden-path tool — it stamps a green-by-construction skeleton,
declares the tier in `architest`, and (with `--register`) wires the blank-import:

```bash
python tools/new_leaf.py myfastcache --tier mechanism --register --summary "prefix-aware fast cache"
```

See the full seam table and the "how a new idea bakes in" walkthrough in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

### The layering gate

Whichever seam you use, `internal/architest` enforces the five-tier layering
(`root → foundation → mechanism → composer → integrator`): a leaf may import lower tiers,
never higher. Confirm you're green before going further:

```powershell
.\fak\test.ps1 ./internal/architest/
```

A violation comes back as a structured DOS reason (`ARCH_LAYER_VIOLATION`) with the fix,
not a wall of compiler output.

---

## Gate 2 — Prove it's correct (a witness, not a claim)

fak's whole thesis is *verify, don't self-certify*. An optimization that "should be
equivalent" isn't accepted on your word — it ships a **witness** the test harness
re-checks. The correctness contract is typed, so it can't silently rot.

### The Reference / Approx contract

`compute.CorrectnessClass` makes the bit-identity scope mechanical:

- **`Reference`** — held to the exact rungs: **`max|Δ| = 0`** against the reference
  reduction order, plus the Hugging-Face argmax oracle. Use this only if your kernel
  reproduces the reference arithmetic byte-for-byte.
- **`Approx`** — every device backend and every quantized lane. Held to the *looser*
  gate: **argmax-exact** (same token out) plus a **logit-cosine** threshold you declare
  per backend. This is the honest class for a faster-but-not-bit-identical kernel.

The harness calls `compute.RequireReference(b)` before any `max|Δ|=0` assertion, so it is
*mechanically impossible* to hold a device to bit-identity it can't meet, or to promote a
device to reference. Declare your class honestly in `Class()` and the right gate applies
itself.

### Write the witness

The pattern is a deterministic, stdlib-only `proofs_witness_test.go` next to your code —
`internal/compute/proofs_witness_test.go` is a live example, and there are ~30 more under
`internal/*/proofs_witness_test.go`, each bound to a theorem in
[`docs/proofs/`](docs/proofs/) with a verdict (`PROVEN` / `REFUTED` / `OPEN` /
`SCOPED-OUT`). The witness is **deterministic** (no wall-clock, no RNG in the gate), so it
reproduces **bit-for-bit across architectures** — a Mac (arm64) and a Windows box (x86_64)
agree, which is what licenses the determinism claim. Run the weight-free metrics with
`-live=false` for an instant cross-platform check.

```powershell
.\fak\test.ps1 ./internal/compute/      # your package's witnesses + the model bit-identity rungs
```

Honesty rule (from [`docs/proofs/00-METHOD.md`](docs/proofs/00-METHOD.md)): **prove or
refute, never launder.** A `REFUTED` witness that records a real gap (e.g. a token-3
decode drift) is a first-class, mergeable result — it's the gap mapped, not hidden. Tag
every claim in your docs with exactly one of `[SHIPPED]` / `[SIMULATED]` / `[STUB]`
([`CLAIMS.md`](CLAIMS.md)).

---

## Gate 3 — Prove it's faster (the keep-bit)

A correct optimization still has to *earn its keep*. fak runs a real
propose → measure → keep-or-revert cycle, and the keep decision is **non-forgeable**: the
"improved" bit (`shipgate.Evaluate`, `internal/shipgate/shipgate.go`) is unexported and
set *only* from a measured witness — you cannot assert your way to a KEEP.

```bash
# one-shot: measure baseline vs your change, decide KEEP / REVERT from the witness
go run ./cmd/rsicycle …
```

The decision is **KEEP** only on *strict gain ∧ green suite ∧ clean tree*; it is **REVERT**
on a no-op, a regression, or a dirty-truth (a result that can't be reproduced from
committed state). So "no measurable win" lands as REVERT, automatically — your optimization
competes against the honest baseline, not against a story about it.

Trace your number to ground truth: every public benchmark figure is reconciled to a
committed JSON artifact in [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md). When you
report "1.4× on this kernel," the artifact + commit are the witness, and the
`bench`-family tools (`tools/bench_*.py`) write the run rows. Free the resource you're
measuring first (e.g. VRAM before a GPU bench) — contention, not architecture, is the
usual cause of a surprising regression.

---

## The scaling contract you must honor (and the reason to build here)

The registries are written **once** at `init()` and read on **every syscall**, so the
rule is: *writes may be expensive; reads must be O(1) and wait-free no matter how many
optimizations are registered.* The 1000th idea must cost the 1st syscall nothing in
framework overhead — reads load an immutable atomic snapshot (no lock, no alloc), event
fan-out is indexed by kind, and the policy folds are per-tool.

**This is the payoff, not just a constraint.** Because each optimization is its own
package / file-tree and attaches through a snapshot-read seam, *your* kernel composes with
everyone else's at zero marginal hot-path cost — and two teams optimizing two subsystems
edit **disjoint files** and never collide (the DOS arbiter leases one file-tree per leaf,
so parallel work is collision-free by construction). If your rung or observer only applies
to specific tools or events, declare it (`CallScope{ Tools() }` / `EventSubscriber{
Subscriptions() }`) and the 100th tool-specific policy costs an unrelated call nothing.

---

## Land it

Once all three gates are green, ship it the same way every contributor does — the rules
are enforced *below* the agent layer by git hooks + the DOS kernel, so a human, a Claude
Code session, and a Codex/Cursor/Aider run all land work identically:

1. **Stay on `master`** in the main worktree — never branch (the trunk guard refuses
   `OFF_TRUNK`). Install the guards once: `python tools/install_trunk_guard.py`.
2. **Commit small, by explicit path** (`git commit -- <paths>`, never `git add -A`).
3. **Stamp the subject** with a `(fak <leaf>)` trailer so the DOS verify-referee can bind
   your "done" to git evidence — e.g. `feat(compute): add q4_k Metal backend (fak compute)`.
   Use `docs(scope): …` for doc-only diffs.
4. **Tests run through WSL** on Windows hosts (`.\fak\test.ps1`); `go build` / `go vet`
   work natively. Never commit a red tree.
5. **DCO sign-off + CLA** on an external PR — see [`CONTRIBUTING.md`](CONTRIBUTING.md)
   and its CLA section (the CLA itself is a draft pending legal review).

That's it. Your optimization is now a first-class part of fak: the core can't break it,
the harness proves it stays correct and fast, and it composes with every other team's
work at O(1) on the hot path.

---

## Why build *on* fak instead of forking

- **A stable spine you can bet on.** The ABI is frozen and additive-only; your seam won't
  be yanked out from under you.
- **You own your file-tree.** One package per optimization → disjoint leases → your work
  never collides with another team's, even running concurrently.
- **Correctness and speed are mechanically gated, not trusted.** Your kernel ships with a
  witness the harness re-checks; it can't silently regress, and neither can the kernel
  underneath it.
- **It composes.** O(1) hot path means the 50th optimization is as cheap to the syscall as
  the 1st. Forking gives you one win; building here lets every team's wins stack.

Questions, or a seam you need that doesn't exist yet? Open an issue (the
`agent-tool-boundary-fixture` template is a good model for a precise, replayable ask), or
propose the new seam as an additive `Register*` in `internal/abi` — the answer to "the
seam I need is missing" is a new named seam, never a core edit.
