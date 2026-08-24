# AGENTS.md — orientation for coding agents

> **Not the human contributor guide.** This file is operating instructions for *automated*
> contributors working inside the maintainers' shared checkout — trunk guards, lane leases,
> commit trailers, and shared-tree rules that have no meaning outside it. If you are a human
> evaluating or contributing to fak, read [`README.md`](README.md) for what fak is and
> [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to contribute; nothing below is required of you.

> You are an autonomous agent working in this repo. This file is the machine-read entry
> point (the [agents.md](https://agents.md) convention). It is intentionally
> command-dense and free of philosophy. For the *why*, read [`README.md`](README.md);
> for a curated doc map, read [`llms.txt`](llms.txt). Humans: see [`START-HERE.md`](START-HERE.md).

## Default scope when launched from this repo

When an agent is launched with this `fak` repository as its working directory,
default all broad or underspecified work requests to **FAK work in this
repository**. Requests such as “finish local WIP,” “work the backlog,” or “run
agents all night” mean inspect, prioritize, implement, test, and ship FAK work;
they do not authorize a workspace-wide sweep of sibling repositories.

Use FAK's guarded headless-agent and worker mechanisms for parallel or unattended
work, following the lane, lease, worktree, landing, and witness rules below.
Only inspect or modify another repository when the operator explicitly names
that repository or the FAK task itself has a concrete, evidenced cross-repo
dependency. If scope is ambiguous, stay in FAK.

## Native inference performance invariant

For any native-inference or performance task, keep model execution **fak-native all the
way**. The product path is intended to beat llama.cpp in matched, quality-constrained
envelopes while fak retains ownership of kernels, memory, scheduling, cache, adaptation,
and operations. llama.cpp is allowed only when explicitly selected for benchmarks,
parity/reference diagnosis, migration/interoperability, or ego-free borrowing. Never add
an `auto`, recovery, or convenience path that silently changes native/performance work to
llama.cpp. Before accepting evidence, ask whether the model executed inside fak and whether
the receipt names that engine. Canonical definitions, matched-envelope rules, and the
deterministic docs guard: [`docs/native-inference-goal.md`](docs/native-inference-goal.md).

## What this project is

**fak** is an *agent kernel*: one Go binary that sits between an AI agent and the tools
it calls, and handles every tool call before it runs — reuse the shared setup, route each
call to the right model, serve repeats locally, and shed old turns while the provider's
cache survives. It is first a **performance gate** (do the shared setup work once, not
every turn — cheaper, faster, longer-running sessions) and, on the very same checkpoint, a
**security gate** (a default-deny capability floor the model can't talk past). Performance
is the current focus; the security floor rides along for free on the same seam.

## Repo layout (where things live)

| Path | What it is |
|---|---|
| `go.mod` · `cmd/` · `internal/` | **The Go module is the repository root** (the kernel + the `fak` CLI). |
| `cmd/fak/` | The `fak` binary (every verb: `preflight`, `serve`, `agent`, `policy`, `bench`, …). |
| `internal/` | Kernel subsystems: `adjudicator`, `policy`, `vdso`, `engine`, `gateway`, `ctxmmu`, `model`, … |
| `examples/` | Policy manifests **and** runnable demos (`adjudication-demo/`, `agentdojo-redteam/`, `mcp/`). |
| `docs/` | Explainers, integration guides (`docs/integrations/`), benchmark methodology, proofs. |
| `docs/private-comms-channel.md` | **The private comms channel** (private control bridge to the lab GPU servers) — a public stub pointing to its home in the `fak-private` companion repo. Start here to reach the hardware. |

## Build / test / run

> **The Go module is the repository root** — run `go` commands from the clone root.
> Needs Go 1.26+ (`GOTOOLCHAIN=auto` self-fetches). Zero external deps, so no `go.sum`.

```bash
go build ./cmd/fak        # -> ./fak  (fak.exe on Windows).
make build                # debuggable ./fak via the dev profile (DWARF/symbols kept, Delve-ready)
make test-fast            # ~2s smoke gate: build + vet + `go test -short ./...`
make test-race            # fast LOCAL race gate (#1311): WSL `go test -short -race ./...`, cgo-preflighted
make build-race           # race-detector ./fak via the race profile (cgo-preflighted, like test-race)
make test                 # full suite incl. the weight-backed model witnesses
make ci                   # the full gate: build + vet + test + claims-lint  (Windows: scripts/ci.ps1)
```

The profiles behind `make build` / `make build-race` / `make release` (debuggable vs race-detector vs
stripped-shipped) are the single flag-delta table in [`docs/dev-tooling.md`](docs/dev-tooling.md#build-profiles),
all routed through the one `scripts/build.sh` entrypoint (#3709/#3710).

### Which build am I asking about? (shared trunk — name the question first)

This checkout is permanently *peer-dirty*: hundreds of uncommitted files and half-wired
untracked `.go` siblings from other live sessions. A bare `go build ./...` / `go vet ./...`
**run in place therefore answers no clean question** — a peer's broken WIP can fabricate a
red, and a peer's uncommitted fix can mask a real one. Name the question, run the one verb:

| Your question | Run |
|---|---|
| Is the **committed trunk** buildable + gofmt-clean? *(what CI gates — what "clean **git** build" means)* | `fak-dev ci-preflight` |
| Does the committed tip plus **only my explicit uncommitted paths** pass full build/vet and affected tests? | `fak validate --mine <path> [--mine <path>...]` |
| Does **my change** compile, ignoring peers' broken untracked WIP? | `fak-dev buildcheck [--vet]` |
| Does the **literal working tree** compile — *my own* untracked WIP included? | `fak-dev buildcheck --isolate=false --vet` |
| Will my **push** red another worker's build graph? | `fak hooks pre-push` |

`fak validate` archives the committed tip, overlays only repeatable explicit `--mine` paths, then runs
full `go build ./...`, `go vet ./...`, and dependency-aware affected tests; it never infers ownership
from the shared dirty tree.
`fak-dev ci-preflight` archives the tip to a throwaway checkout (immune to the dirty tree;
`--skip-build` = gofmt-only fast path). `fak-dev buildcheck` discards output to the null device
and, by default, `-overlay`s away untracked siblings so a peer's WIP can't red your compile —
an untracked `.go` in a package *you're* already editing is auto-kept as the matched new file,
`--mine <file>` force-keeps one wired in from elsewhere, and a red that's purely mask-induced is
re-checked against the live tree and reported OK before it reaches you; `--isolate=false` builds
the live tree as-is, which is the one that catches *your own* broken untracked `_test.go`. None
of the three writes an in-tree binary. `make test-fast` / `make ci` remain the local full-suite
gates. **When you report "the build is clean" from a raw `go build`/`go vet` in this working
tree, say which question you mean** — they give different answers here.

Or install the released binary directly — the module is at the repo root, so this resolves:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Windows:** `go build` / `go vet` / `go run` work natively, but native `go test` is
> blocked by an OS Application-Control policy on the freshly-compiled test binaries — run
> the suite under WSL: `./test.ps1` from the repo root. This is an OS quirk, not a code failure.
> For now, **use WSL/CI for tests and the GPU/cloud nodes for a real serve** — keep tests
> and long-lived local `fak serve` off the dev box. See
> [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md).

> **Build *verification* must never write the in-tree binary.** A bare `go build ./cmd/fak`
> drops `fak.exe` into the repo root; while a fleet process holds the old binary open, Windows
> throws `open fak.exe: The process cannot access the file because it is being used by another
> process` (#2373). The fleet-safe way to compile-check on the shared trunk is **`fak
> buildcheck [pkgs...]`**: it discards the output to the null device (never an in-tree binary,
> universal across lib/main/multi-pkg) *and* generates a `go build -overlay` that hides peers'
> untracked in-flight `.go` files (keeping any that sit in a package you're already editing, since
> those are almost always the matched new half of your edit) so a sibling's WIP can't red your
> compile — `--mine <file>` force-keeps one wired in from elsewhere, `--vet` runs `go vet`,
> `--json` reports. The raw fallback
> (when the binary is unbuildable) is `go build -o $env:TEMP\fak-verify.exe ./cmd/fak`
> (PowerShell) / `go build -o /tmp/fak-verify.exe ./cmd/fak` (bash), or `go vet ./cmd/fak` —
> but note a *fixed* temp name still collides when two agents run it at once, which is exactly
> what `fak-dev buildcheck`'s per-process temp overlay avoids. The in-tree `go build -o fak` above
> is for *producing* the binary you run, not for a compile check.

## The 60-second proof (no key, no model, no GPU — verified)

This is the canonical first command. Run it before anything else:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK): refused by structure, no model in the loop
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW: not a blanket block
./fak agent --offline                                                                                       # -> injection-in-context YES->no, destructive-op YES->no, task still booked
```

The first `go build` compiles the binary (~30–60s, plus a one-time Go-1.26 toolchain fetch);
later runs are instant. Full walkthrough: [`docs/repro-packet.md`](docs/repro-packet.md).

## The local machine is the control point, not the compute boundary

That the *kernel proof* needs no GPU (above) is not license to stop when a task **genuinely**
needs one. The workstation you run on plans, dispatches, and ledgers — it is the **control
point**. It is **not** the compute boundary. When a task needs a CUDA/GPU device, a specific
accelerator arch, DC-network reach, or heavy CPU, **dispatch it to a fleet node that has it**
and report the real result.

**Never end a turn with `not yet — no GPU on this host`** (or any "blocked for lack of local
hardware") as a terminal answer. There is always a route: a sanctioned compute node, or — if its
credential/bridge session is missing — the exact ready-to-run command sequence handed to the
operator. Both count as "used the lab"; only `no local hardware` does not. The nodes and how to
reach each are the SSOT in **[`docs/fleet-compute-nodes.md`](docs/fleet-compute-nodes.md)**
(GCP `fak-realmodel` L4 for a live CUDA serve; the GPU server via `dgxbridge` for a device-GEMM
witness; `da33` for DC/CPU; the nightrun pipeline for a nightly ledger).

This is guard-enforced, not honor-system: `fak hwgate-lint` scans a final turn for the stop
pattern and types each hit to a sanctioned-node redirect (the hardware-gate dual of
`fak headless-lint`), and `fak guard`'s Stop hook blocks a headless stop that declares a
local-hardware blocker as terminal (`--hardware-gate enforce`), feeding the redirect back so the
agent dispatches instead of stopping. `fak guard-stops` tallies the pattern for the soak → promote read.

## Proof by default (every issue fix ships its evidence)

This kernel exists because a self-report is not a fact. Hold your own fixes to the same bar:
**by default, an issue fix ships a captured proof artifact, matched to the failure class** — not
just a code change and a "looks fixed". Pick the witness the bug actually has:

- **TUI / visual** (a rendering, layout, or terminal-corruption bug — the kind you were sent a
  *screenshot* of): the proof is a **captured render**. Write a render-witness test that captures
  the bytes a surface emits and asserts the defect is gone — see
  [`cmd/fak/watchdog_autoheal_test.go`](cmd/fak/watchdog_autoheal_test.go)
  `TestWatchdogAutohealKeepsAgentPaneClean`, which captures the would-be agent pane and proves
  **zero bytes** reach it. When a live on-screen UI is in the loop, also attach a **before/after
  terminal screenshot** to the issue/PR (the same evidence the reporter gave you). A green unit
  test that never renders the surface is not proof for a visual bug.
- **Logic / behavior**: a test that **fails before the fix and passes after** — the repro is the
  proof. Land it in the same commit as the fix.
- **"Shipped / done" claims**: a witnessed commit (`dos verify`, the `(fak <leaf>)` trailer) — see
  the witness rules below. A subject line is forgeable; the diff and the registry are not.

The rule of thumb: reproduce the defect as a captured artifact *first*, then make that artifact
clean. If you cannot capture it, you cannot prove you fixed it — say `not yet` with the missing
witness instead of claiming a fix.

## New work defaults: spine first, then fan out

Two defaults fire for **any new unit of work** (feature, leaf, verb, demo, process change) —
full doctrine in [`docs/spine-first-defaults.md`](docs/spine-first-defaults.md), agent
checklist in the `/spine-fanout` skill. Before either, classify the work's **problem
centrality** (`Core`, `Enabling`, `Stewardship`, or `Peripheral`) and run all four rows of
the [`P1`-`P4` problem checklist](docs/problems-we-solve.md): managed context, net-true
efficiency, bounded adaptation, and integrated operations. These are all-work checks, not
competing priority buckets or a choose-one ID. Then pass the
[Feynman-simple value frame](docs/shift-left-task-organization.md): name **For / Problem /
Today / Better because / Witness** in plain language. Compare against the real next-best alternative,
not an imagined baseline. If a new operator cannot repeat why the smallest spine should
improve an observable user outcome, keep scoping; do not compensate with architecture,
fan-out, or exhaustive proof.

1. **Ship the minimal WORKING end-to-end spine first, in the same session the work starts** —
   the smallest runnable path through the real seam (LCD demo with `-selfcheck` for a
   user-facing surface; a test driving the real object plus one captured live run for a
   library/verb). **Order the work applied implementation → captured spine witness →
   exhaustive operating-envelope proof → measured optimization.** Do not lead with a broad
   comparison matrix, exhaustive edge proving, or component optimization while the primary
   end-to-end outcome is still "almost there"; those are follow-ons anchored to a working
   path. Safety and fail-closed behavior needed to run the path remain part of the spine.
   If that is not achievable this session with high confidence, **file the
   spine itself as the first issue** (`gen/now`, milestoned, missing witness named) — a spine
   is never silently deferred.
2. **File the follow-on backlog at creation time (3..50+ issues)** — the moment a spine
   ships, run `fak-dev issue fanout --title T --leaf L --spine <sha|cmd|doc> --json` to expand
   the QA / dogfood / productization / observability / integration / docs / release taxonomy
   into contract-ready candidates (every one dispatchable under `fak-dev issue contract`), then
   file them with milestone + labels at creation, or wave-plan first via
   `fak-dev issue cohort --from-plan`. The verb refuses to plan without a spine witness — that
   refusal is default 1 talking.

**End of run: file the leftovers, don't narrate them.** `never silently deferred` binds
every run, not just new-work spines. If a task finishes and surfaces remaining or
out-of-scope follow-ups — the "there are two more things worth doing" you'd otherwise
list at the end — **file each as an open gh issue** before you stop (dedupe →
done-condition → leak-check the body → label), then report those issue numbers. A
follow-up named in prose but left unfiled is silently-deferred work: it becomes an OPEN
issue or it does not leave the run. This binds headless workers, in-session loops, and
interactive turns alike; if there is genuinely nothing left, say so plainly.

**Promote scratch up the tooling ladder — a working artifact must not die with the
session.** The rungs are **scratchpad → committed tool → first-class verb → captured
knowledge**, and the failure mode is stopping one rung too low. A throwaway script that
*worked* is a deliverable, not a probe: left on rung 1 it evaporates with the session, and
the next agent re-derives it and re-hits the same traps. Promote **by default, without
being asked**, when any of these holds — it succeeded at a real task (not merely probed),
it would plausibly be re-run (same question next cycle, same environment next quarter), it
encodes a non-obvious operational fact the ecosystem already got wrong once, or its design
beats the committed equivalent. Rung 1 is correct *only* for a true one-off: a diagnostic
probe, a throwaway data pull, or logic a committed tool already covers. Two fences bind the
climb **here**: a promotion lands as a **Go leaf + a `fak` verb**, never a new `tools/*.py`
(the `internal/pythongate` ratchet reds the trunk — see "New tooling is Go, not Python"),
and a rung-4 operational fact goes where the next agent *looks* — a dated
[`docs/notes/`](docs/notes/) note or the leaf's doc — not buried in a script's control
flow. The asymmetry is the whole argument: over-promoting costs a short tool and a test,
while under-promoting costs the next agent the entire re-derivation. The harness session
scratchpad is lossy by construction — it is keyed by session-uuid and strands on resume
([`docs/notes/CONCEPT-HARNESS-SESSION-SCRATCHPAD-2026-07-02.md`](docs/notes/CONCEPT-HARNESS-SESSION-SCRATCHPAD-2026-07-02.md)),
so "I'll promote it later" is not a plan. (This is the *tooling* ladder — an artifact's
route out of scratch. It is not `internal/maturity`'s capability-lifecycle ladder, which
scores a declared leaf's rung; see [`fak maturity`](docs/cli-reference.md).)

**Close operator-facing turns with scannable bullets, verdict first; make the last line a bullet carrying the next checkable step.** This binds the summaries and handoffs an operator reads — headless workers, in-session loops, and interactive turns alike; tool-call arguments, commit subjects, and PR bodies keep their own conventions. One claim per line, evidence and paths inline. A short single-line closer ("nothing left; pushed abc123") already ends clean, so the "genuinely nothing left, say so plainly" escape above still holds.

## Version everything: cite `module@rev`, not just a bare SHA

Every module carries a **derived** version — there are no hand-maintained per-module version
files (×410 they would rot within hours on this shared trunk and spew merge noise). A module's
`rev` is the count of trunk commits that touched it, rendered `r<rev>+g<shortsha>` (e.g.
`internal/gateway r652+g1f75c56d`) — monotonic, conflict-free, and computed from history alone.
Read the table with `fak version modules` (`--json` for machine form, `--scores S.json` to join a
flat `{"module": score}` map); a nightrun / super-loop turn appends changed-module rows to the
`fak-module-versions/1` ledger with `fak version modules --stamp` (stamping twice at the same HEAD
is a no-op). **When you cite evidence in a claim or a handoff, prefer `module@rev` over a bare SHA**
— it says *which part moved and how far*, which a SHA alone does not. Full doctrine + e2e:
[`docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md`](docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md).

## Hard rules (enforced below the agent layer)

- **Ship green work by default.** Run the scope-correct gate above, then commit and push without waiting for a prompt. Stay on `main`; never force-push, create a feature branch, use `--autostash`, or escape a dirty/diverged tree into a worktree. Merge `origin/main` in place. If a peer owns `MERGE_HEAD`, unstage your paths and wait; do not finish or abort their merge.
- **Commit exactly one issue through explicit paths.** Prefer `fak commit --preview`, then `fak commit --path <p> ... -m "<subject>"`; use `fak sweep` for one coherent lane. The fallback is `git commit -s -m "<subject>" -- <paths>`, never `git add -A`. One issue lands in one commit and one leaf; do not split a green issue into patch commits or batch unrelated issues.
- **Make the first subject final.** Sign off with DCO, use a Conventional-Commits subject, and include a recognized `(fak <leaf>)` trailer. A peer may push your commit before an amend, so preview the subject and paths first. Demo binaries use their `cmd/<dir>` name as the leaf.
- **Preserve shared-tree buildability.** A tracked or untracked `.go` sibling enters every package build. Fence incomplete cross-file WIP with `//go:build wip_<feature>` until its symbols exist; validate only your explicit paths with `fak validate --mine`. Build verification never writes an in-tree binary.
- **Migrate CI/CD contracts atomically.** Before changing workflow inputs, JSON schemas consumed by workflows, check/job names, secrets/env, runner labels, or artifact/cache names, search the whole tree for consumers and update them together. Include changed contract, migrated consumers, impact/cutover, and rollback in the commit body; prove committed tip with `fak-dev ci-preflight`. Checklist: [`docs/ci/ci-spec-change-migration.md`](docs/ci/ci-spec-change-migration.md).
- **Use only the sanctioned detached worker worktree.** Drive isolated workers through `fak worktree worker prepare|land|reap|list`; all branch worktrees remain forbidden. Reap stale scratch worktrees with `tools/worktree_doctor.py`, which preserves live sessions.
- **Allocate and reap scratch explicitly.** Use `fak tree-doctor --scratch-dir <producer>` or `--scratch-path <producer>/<file>`, then `--reap-scratch <producer> --json`. Never use `git clean -Xdf` under `_scratch/`. Per-run prompts/transcripts belong in allocated scratch; `testdata/` is only for fixtures landed with a consuming test. See [`docs/generated-output-defaults.md`](docs/generated-output-defaults.md).
- **Keep comments durable.** Explain non-obvious invariants, safety, concurrency, compatibility, or performance tradeoffs; do not narrate syntax. Preserve required exported-API, package, directive, generated, and legal comments.
- **Keep claims and gains witnessed.** Every `CLAIMS.md` row needs its required tag. Report performance only from net-true end-to-end accounting, with quality and operating-envelope constraints; setup/recovery/verification overhead counts. Use the project claim and benchmark gates rather than hand-written “looks faster” prose.
- **Extend through leaves and Go tooling.** Add capabilities with `fak new-leaf`; avoid editing the core registry directly. New durable tooling is a Go leaf plus a `fak` verb, not another `tools/*.py`; the Python gate permits maintenance of existing scripts, not new ones.
- **Keep private control private.** GPU-server credentials, hostnames, SSH details, private paths, and raw internal logs stay in the private companion repo. Public evidence must be scrubbed and reproducible through [`docs/private-comms-channel.md`](docs/private-comms-channel.md).
- **Respect Windows operational fences.** Never run broad `find /`, `find ~`, `find /mnt`, or `find /proc` from Git Bash. Route git and hang-sensitive commands through the guarded fak verbs. Treat low-utilization whole-machine stalls as kernel-path churn: stop launch storms, inspect process counts/handles, and use the committed diagnostics rather than adding retries.
- **Make external writes explicit.** `OUT_OF_TREE_WRITE` records writes outside the repo. Use the operation’s declared external target and reversible preview/confirm path; never disguise an external mutation as repository scratch.

The detailed recovery for a refusal is intentionally paged: query the token below instead of carrying every failure cookbook in every agent context.

### If the kernel refuses you (recover, don't fight it)

A guard refusal names a token from the closed `[reasons.*]` vocabulary in
[`dos.toml`](dos.toml). Query that source instead of carrying the full recovery
cookbook in every agent context:

```bash
dos man wedge <TOKEN> --explain  # summary, category, fix, and references
fak recover <TOKEN>              # concrete recovery plan when one exists
fak recover --list               # list executable and manual-only plans
```

Recover by the named action; do not route around the guard. `fak recover` is a
dry-run unless `--execute` is passed, and manual-only plans refuse execution
rather than guessing. The token argument is case- and separator-insensitive.

Keep the common commit-lane rules available before a refusal: stay on `main`,
commit only explicit paths, never amend or force-push shared history, wait out a
peer's `MERGE_HEAD`, and reconcile divergence in place. The hard rules above are
the preventive contract; the query surfaces are the token-specific recovery path.

Check setup failures with `python tools/extend_preflight.py`; the full contributor
contract is [`CONTRIBUTING.md`](CONTRIBUTING.md). If a refusal is genuinely wrong,
file its witnessed, deduplicated appeal with `fak complain --summary "…" --reason
<TOKEN> --tool <Tool> --from-journal --args-digest <sha256:…> --live`. Recover
first; appeal only when the guard, not the call, is wrong. Taxonomy and routing:
[`docs/notes/CONCEPT-AGENT-FRICTION-COMPLAINT-CHANNELS-2026-06-29.md`](docs/notes/CONCEPT-AGENT-FRICTION-COMPLAINT-CHANNELS-2026-06-29.md).

## Releasing (cut, publish, roll back)

The version source-of-truth is the bare `VERSION` file; the shipped history is the
`vX.Y.Z` git tags; release notes live under [`docs/releases/`](docs/releases/).
`@latest` (what `go install …/cmd/fak@latest` resolves) is the newest tag, so if no
tag is cut as work lands, `@latest` rots behind HEAD — check the lag any time with
`make release-staleness` (`fak release-staleness --json`), and the whole release
posture with `make release-readiness` (the deterministic release-debt scorecard).

To cut one from the normal hot shared tree, run `fak release ship --execute --json`.
It creates a transient detached worktree at `origin/main`, shares the repo-level
release lock, runs the mechanical helpers in order, pushes the release commit to
`main`, tags after the CI witness, creates the GitHub release page, and removes the
worktree. The lower-level helpers remain available for diagnosis:

1. **Decide** — `python tools/release_decide.py --json`. `decision: "hold"` names a
   blocker (`CI_BASE_RED`, `VERSION_DRIFT`, `NOTHING_TO_SHIP`); don't cut through it.
2. **Lock** (multi-session tree) — `python tools/release_lock.py acquire --ttl 1800`
   so a second `/release` session can't race `VERSION`/the tag.
3. **Cut** — bump `VERSION` + draft the note + commit, by explicit path only (never
   `git add -A` — the lock's `guard` verb catches a sweep that grabbed a peer's file).
4. **Push then tag** — push the release commit to `main` *first* (the tag check reads
   local `main`), then `release_tag`; then `release_publish` creates the GitHub release
   page (the `release-artifacts.yml` workflow decorates it with the cross-platform
   binaries + checksums — it fails `release not found` if the page doesn't exist yet).

Same trunk rules as everything else: commit on `main`, by path, `-s` for DCO; on a
**hot tree** use `fak release ship --execute` rather than stashing peers' work.
Stable rollback anchors are a separate, slower channel — see
[`docs/stable-releases/`](docs/stable-releases/).

## Planning: two kinds of work (don't put an ongoing program on a % bar)

Before you track a piece of work as an epic, classify it. The project draws one line,
encoded in [`internal/worktype`](internal/worktype/worktype.go):

- An **ongoing optimization program** is never "done": **kernel-optimization** (chasing
  decode/prefill throughput + numeric parity toward and past SOTA) and
  **cache-optimization** (agent memory + reuse — multi-agent KV reuse, O(1) bounded
  context, provider-cache preservation, addressable KV deletion). Its honest measure is a
  **frontier + a trend**, never a completion %. A "60% complete" line on either is a
  category error — there is always a faster kernel and a better reuse ratio. Track it with
  `fak program report`; its operating spines are
  [`docs/perf-parity-rsi-loop.md`](docs/perf-parity-rsi-loop.md) and
  [`docs/cache-value-rollup.md`](docs/cache-value-rollup.md).

  **Before you optimize a kernel, check the prior art — build on known art rather than
  re-deriving it.** Almost every contraction fak performs (a quantized GEMM, a fused
  attention, a KV-cache reuse, a MoE dispatch) has a production reference (llama.cpp / Marlin / CUTLASS / FlashInfer / vLLM /
  SGLang / a named paper). Run `fak sota <operation|file>` to surface the reference, route,
  and oracle *before* writing from scratch; read [`docs/sota/README.md`](docs/sota/README.md)
  for the map and the process; stamp the kernel commit with a `Prior-art:` trailer naming what
  you consulted. The source of truth is [`internal/sotamatrix`](internal/sotamatrix/sotamatrix.go);
  the `PRIOR_ART` advisory gate nudges at commit time and `tools/sota_coverage_scorecard.py`
  keeps the matrix complete against the tree.
- A **discrete deliverable epic** has a definition of done and converges on 100% (the
  native harness, release-at-agentic-speed, support-maturity). Completion % is the right
  lens; track it in the `fak milestone report` roadmap.

`fak milestone report` shows both, in separate sections, so a never-done program is never
read as a stalled deliverable. To mark a new epic as a program, add its number to the
`epicClass` map in `internal/worktype` (one line) — not a magic number in a report.

A second, orthogonal axis separates work by **class** — fleet **infra** (CI, dispatch
loops, observability, slack, build) vs product **dev** leaves vs the public
**front-door / mainline** release path. It is derived from the file-tree lane an issue
routes to (`tools/issue_lane_router.py`) and surfaced as three issue-views —
`fak-dev index work dev-leaves` (product only, plumbing hidden), `... infra`, and
`... front-door` (the fenced release path). Use it to dispatch one class at a time; see
[`docs/work-class-axis.md`](docs/work-class-axis.md).

## Where to go next

| If you want to… | Read |
|---|---|
| Every CLI verb + what's shipped | [`docs/cli-reference.md`](docs/cli-reference.md) |
| Learn every concept in prerequisite order (a course, join at your level) | [`LEARNING-PATH.md`](LEARNING-PATH.md) |
| Install / run tiers (offline → gateway → in-kernel model) | [`fak/GETTING-STARTED.md`](GETTING-STARTED.md) |
| Put fak in front of *your* agent (Claude Code / Cursor / MCP) | [`docs/integrations/`](docs/integrations/) · [`fak/examples/mcp/`](examples/mcp/) |
| Run hardware-gated work (no local GPU) — the sanctioned compute nodes | [`docs/fleet-compute-nodes.md`](docs/fleet-compute-nodes.md) · `fak hwgate-lint` |
| The deployable capability floor (policy manifests) | [`fak/POLICY.md`](POLICY.md) · [`fak/examples/README.md`](examples/README.md) |
| Extend the kernel (plug in → prove correct → prove faster) | [`fak/EXTENDING.md`](EXTENDING.md) · [`fak/ARCHITECTURE.md`](ARCHITECTURE.md) |
| Optimize a kernel without re-inventing known art (check prior art first) | [`docs/sota/README.md`](docs/sota/README.md) · `fak sota <op>` |
| Score agent-steer prose for negative framing (suggests positive reframes) | `fak score negframe --suggest` |
| Every feature by subsystem, with honest status | [`docs/supported/features.md`](docs/supported/features.md) |
| What's real vs simulated vs stub | [`fak/CLAIMS.md`](CLAIMS.md) · [`fak/STATUS.md`](STATUS.md) |
| Every benchmark number (single source of truth) | [`fak/BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) |
| Roll back to a stable version (revert / downgrade / pin) | [`docs/ROLLBACK.md`](docs/ROLLBACK.md) |
| A curated map of all the docs | [`llms.txt`](llms.txt) |

License: [Apache-2.0](LICENSE).
