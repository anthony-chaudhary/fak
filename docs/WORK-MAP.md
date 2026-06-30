# WORK-MAP: optimizations, ongoing work, and dev, kept separate

**WORK-MAP is the index that separates fak's three kinds of work and routes each to its
own front door.** The three are easy to conflate because they share files and cross-link
constantly; this page tells you which door a task belongs to.

> **TL;DR.** Optimizations go through [`EXTENDING.md`](../EXTENDING.md)'s three gates;
> ongoing work is tracked in [`INDEX.md`](../INDEX.md)'s "Status & tracking" list plus the
> [issue tracker](https://github.com/anthony-chaudhary/fak/issues); the dev workflow
> starts at [`AGENTS.md`](../AGENTS.md).

The three:

- **Optimizations**: make a subsystem faster or smarter (a quantization kernel, a
  cache-eviction policy, an admission rung, a KV layout). Lands through a fixed,
  mechanical three-gate contract.
- **Ongoing work**: the in-flight efforts, epics, and backlog being driven right now.
- **Dev**: the core development and contributor workflow for building, testing,
  partitioning, and shipping any change.

This is a navigational map. It points at surfaces the repo already maintains, and it
does not block a commit. Where a category is well-organized it says so; where it drifts
it says that too (see [Overlaps & known drift](#overlaps--known-drift)).

The spine all three reconcile against is the claim ledger ([`CLAIMS.md`](../CLAIMS.md) plus
[`docs/claims-salience-register.md`](claims-salience-register.md)): every capability
carries one machine-checked tag (shipped, simulated, or stub).

## 1. Optimizations: "make subsystem X faster or smarter"

The best-organized of the three: a front door, a correctness proof, and a net-win proof,
in that order.

| Surface | Role |
|---|---|
| [`EXTENDING.md`](../EXTENDING.md) | The front door. Every optimization lands the same way, through three mechanical gates: (1) plug in (a `Register*` seam plus the `internal/architest` layering gate), (2) prove correct (the Reference/Approx correctness class plus a deterministic witness test), (3) prove faster (the non-forgeable keep-bit, `shipgate.Evaluate` via `cmd/rsicycle`). You do not get to skip a gate. |
| [`docs/INNOVATIONS-INDEX.md`](INNOVATIONS-INDEX.md) | The catalog of what has been built, grouped by subsystem family (safety/kernel, context/cache/memory, model/compute, serving/routing/scheduling), each row tagged `SHIPPED` / `SIMULATED` / `STUB` / `MIXED` and whether it has been generalized for reuse. |
| [`docs/rsi-loop.md`](rsi-loop.md), [`docs/perf-parity-rsi-loop.md`](perf-parity-rsi-loop.md) | The keep-or-revert loop. Gates an optimization on a measured net win, applies the candidate in an isolated worktree, and reverts on a keep-bit miss (`cmd/rsicycle`). |
| [`docs/standards/net-true-value.md`](standards/net-true-value.md) | The rubric every perf claim is judged by: a real baseline (the actual alternative), net of its own cost, scope stated, provenance-labeled, reproducible, on by default. |
| [`docs/CUDA-DEV-SCORECARD.md`](CUDA-DEV-SCORECARD.md), the `*-parity-tracking-*` notes | The perf measuring sticks that keep the optimization lane honest over time. |

## 2. Ongoing work: what is in flight right now

The weakest-organized of the three: real, but spread across three parallel surfaces with
no single live view.

| Surface | Role |
|---|---|
| [`INDEX.md`](../INDEX.md), "Status & tracking" | The hub that links every per-effort tracker, and the closest thing to a single roll-up today. |
| `docs/notes/*-tracking-*.md`, `*-status-*.md` | The per-effort trackers (dated by design, so they age out): Track B perf parity (#306), Track D agent-framework parity (#304), Track F integration/tooling (#302), GPU parity (#480), SIMD CPU parity (#400), the self-tax performance-assurance plane (#1147), the model-arch seam (#487), ultra-long context (#519), the [verification-ladder epics](notes/verification-ladder-epics.md). |
| [GitHub Issues](https://github.com/anthony-chaudhary/fak/issues) plus [`docs/dispatch-loop.md`](dispatch-loop.md) | The live backlog and the witness-gated loop that drives it (`cmd/dispatchworker`: spawn, ship #N, witness, close). The always-current open-issue count lives in the tracker, never hard-coded here. |
| [`docs/generation.md`](generation.md) | The generation-aware development contract for epic #1625: now/next/second-next/future stream labels, matching milestones, promotion/demotion evidence, intake rules, issue views, and the rule that generation is orthogonal to priority, shared trunk, and runtime feature gates. |
| [`docs/idea-scout.md`](idea-scout.md) | The inbound feeder: a daily arXiv and GitHub sweep that files triage-ready issues (`tools/idea_scout.py`), the complement to the dispatch loop. |
| [`docs/EXECUTIVE-ROLLUP.md`](EXECUTIVE-ROLLUP.md), [`docs/PRODUCT-STATUS.md`](PRODUCT-STATUS.md) | Point-in-time snapshots: the leadership roll-up and the tree-checked product-standing map. Current, regenerated from the tree. |
| [`operator-brief.md`](operator-brief.md) / `fak operator brief` | The human pacing layer: folds cadence, program, milestone, optional operator-heaviness JSON, and optional previous-brief JSON into source coherence, change delta, attention timebox/read-order, human-use guidance, and `human`, `agent`, `watch`, and `background` buckets so operators can see decisions separately from delegable work, surface pressure, and ambient telemetry. |
| [`human-operator-effectiveness.md`](human-operator-effectiveness.md) / `fak program report` | The ongoing human-steerability program: tracks operator attention, learning pace, source coherence, change compression, and surface pressure as a frontier + trend rather than a completion bar. |

## 3. Dev: the core development and contributor workflow

How any change is built, tested, partitioned across parallel sessions, and shipped.

| Surface | Role |
|---|---|
| [`AGENTS.md`](../AGENTS.md), [`CLAUDE.md`](../CLAUDE.md) | Orientation plus the working contract: build/test/run, the repo layout, and the hard rules enforced below the agent layer (trunk-only, commit-by-path, ship-stamp grammar). |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | How to land a change: the full contributor contract. |
| [Developer tooling](dev-tooling.md) | The hands-on practitioner layer: the commands you run *inside* the loop — the test runner (`make test*` + WSL), the debuggers (`fak debug` / `fak doctor`), profiling (Go pprof + the benchmark verbs), and the commit-and-ship loop. Honest about which capabilities are dedicated `fak` verbs today and which (`fak profile` / `fak test`) are planned. |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`PARTITION.md`](../PARTITION.md) | The structure: the registry seams, the frozen additive-only ABI (`internal/abi`), and the star-of-disjoint-leaf-trees model a feature attaches to. |
| [`dos.toml`](../dos.toml) `[lanes]` | The mechanism that makes parallel dev safe: one lane per leaf across the 115-tree `[lanes.trees]` roster, so two sessions editing disjoint leaves never collide. `internal/architest` fails the build on an upward or cross-tier import. |

## How the three connect

A unit of work usually moves through all three. It is born in dev, as a new leaf behind a
`Register*` seam. It is tracked as ongoing work, under an issue or epic in a tracking
note. Once it passes the [`EXTENDING.md`](../EXTENDING.md) gates it graduates into the
optimizations catalog ([`docs/INNOVATIONS-INDEX.md`](INNOVATIONS-INDEX.md)) with a
`CLAIMS.md` tag. The claim ledger is the spine the other two reconcile against, so a
concept cannot read "shipped" in one surface and "stub" in another without the lint
catching it.

## Overlaps & known drift

Where the separation is still implicit, or the surfaces have drifted:

- Three maturity ladders for one truth. The [`PRODUCT-STATUS.md`](PRODUCT-STATUS.md)
  verdict ladder (durable-product, usable-today, real-not-easy, stub), the
  [`INNOVATIONS-INDEX.md`](INNOVATIONS-INDEX.md) `SHIPPED`/`SIMULATED`/`STUB` tags,
  and the [`CLAIMS.md`](../CLAIMS.md) tags are three views of the same concepts. They agree
  today, yet are maintained separately.
- Ongoing work has no single live view. It is split across the `INDEX.md` "Status &
  tracking" list, roughly 15 dated tracking notes, and the GitHub issue tracker. No one
  page shows the current in-flight set.
- Status snapshots drift. The root [`STATUS.md`](../STATUS.md) still reads v0.2.1 while
  [`PRODUCT-STATUS.md`](PRODUCT-STATUS.md) is at v0.34.0. Treat `STATUS.md` as a
  historical witness record (its value is the per-claim witness table rather than the
  version number); [`PRODUCT-STATUS.md`](PRODUCT-STATUS.md) and
  [`docs/EXECUTIVE-ROLLUP.md`](EXECUTIVE-ROLLUP.md) carry the current standing.

## Where to go next

- New here and want to build something faster: [`EXTENDING.md`](../EXTENDING.md).
- Want to pick up an in-flight effort: [`INDEX.md`](../INDEX.md) "Status & tracking" plus the
  [issue tracker](https://github.com/anthony-chaudhary/fak/issues).
- Want to work in the tree at all: [`AGENTS.md`](../AGENTS.md) first.
- Want the full repo map (everything, not just work): [`INDEX.md`](../INDEX.md).
