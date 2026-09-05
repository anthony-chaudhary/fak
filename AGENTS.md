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

## Divide and conquer: delegate real work; keep the coordinator context clean

Use guarded headless agents or an equivalent isolated worker for every substantive
unit of work. The primary agent is the coordinator: decompose the request, give each
worker a bounded goal and distinct file set, preserve only decisions and compact
evidence in the primary context, and independently witness worker results before
landing or reporting them. Delegate investigation, implementation, tests, long command
output, and independent review; do not pull their full transcripts into the coordinator.

- **Divide and conquer by default**: When handling substantive, multi-part, or complex tasks,
  decompose them into atomic, single-concern units or parallel subagent packets. Launch specialized
  subagents (`task`: worker, researcher, explore, deep-reason, cross-validator) concurrently for
  independent components to maximize throughput and prevent coordinator context saturation.
- **Tree-disjoint boundaries**: Assign each worker a distinct, non-overlapping file set to avoid
  concurrent collisions on the shared trunk.
- **Isolate and witness**: Keep heavy command logs and raw transcripts in worker boundaries; pull
  only compact decisions, diffs, and verification receipts into the coordinator.

The primary agent may directly perform only lightweight coordination: inspect enough
state to scope packets, launch and supervise workers, adjudicate conflicts, integrate
witnessed results, and run the final completion audit. A trivial one-command answer or
tiny edit may stay local when launching a worker would cost more than the work itself.
Use FAK's managed launch, lane, lease, detached-worktree, landing, and witness surfaces;
worker self-reports are not evidence, and delegation does not relax ownership of the
final result.

## Scope discipline for smaller models: subdivide or abstain

Smaller or resource-constrained models (such as local 7B/14B checkpoints, fast/flash models,
or bounded worker subagents) achieve reliability by keeping work tightly focused and strictly verified.
When operating as or delegating to smaller models, enforce scoping safeguards:

1. **Subdivide into atomic units (S0/S1 leaves)**:
   - Restrict each task or dispatch packet to a single observable deliverable and exactly one witness command.
   - Limit the active write surface to 1–3 closely related files within a single package or lane.
   - Decompose multi-step tasks into sequential, verified phases: write the reproduction test first, commit the minimal implementation, and verify the targeted package.
   - Complete and witness one step before advancing to the next; keep edits focused rather than attempting broad multi-subsystem sweeps in one turn.

2. **Abstain from high-difficulty aspects (scoped fail-to-abstain)**:
   - Identify task aspects that demand deep architectural context or high-risk reasoning: concurrency invariants and lock ordering, frozen ABI modifications (`internal/abi`), low-level SIMD/CUDA kernel mechanics, cross-subsystem protocol migrations, and security policy gates.
   - Scope abstention strictly to the bounded high-difficulty aspect; maintain momentum by executing all independent, safe, solvable sub-components (such as baseline reproduction tests, diagnostic witnesses, non-gated packages, or documentation) rather than abandoning the prompt.
   - Emit a structured `ABSTAIN` verdict with a typed refusal token and exact boundary description for the escalated aspect alongside the landed partial evidence.
   - Deliver the verified sub-component and cleanly escalate the isolated difficult aspect to a higher-capability model or human operator.

3. **Guard against fast/flash model sharp edges (Gemini 3.8 Flash & peers)**:
   - **Curb token inflation and verbosity**: 3.8 Flash is designed to "work harder" and can output 2×–4× more tokens than other models per task. Enforce extreme conciseness (<3 lines commentary in CLI), eliminate conversational preambles/postambles, and keep explanations minimal.
   - **Resist over-scaffolding ("happy-go-lucky" sprawl)**: Flash models are prone to generating unsolicited companion abstractions, multi-panel apps, or broad refactors for simple requests. Confine diffs strictly to the requested lines/files; do not introduce unasked scaffolding.
   - **Beware thinking effort tradeoffs**: At low thinking effort, 3.8 Flash exhibits quality regressions (spatial/geometric flaws, shallow verification); at high effort, it burns large token budgets. Never trust low-effort intuition on complex logic—always verify against deterministic external tools (`go test`, `go vet`, `fak validate`).
   - **Break interactive tool loops while persisting toward the goal**: In CLI/tool loops, Flash models can confabulate success or loop in repeat-failure cycles ("apologizes, then retries the exact same command"). Ground every claim in an observed tool receipt. When a tool call is denied or fails, halt repetition of the identical call or cycling argument variations; read the error or refusal receipt, query `fak recover <TOKEN>` for structured recovery, and pivot to an alternate sanctioned route or decomposed subtask. Persist through recoverable hurdles by adapting the approach rather than repeating failed calls or abandoning the objective.
   - **Prefer specialized file tools over shell scripting**: Flash models experience higher failure rates on complex CLI/terminal pipelines (TerminalBench regressions). Prefer structured tools (`Read`, `Edit`, `Glob`, `Grep`) over complex piped bash commands.
   - **Anticipate safety false-positives**: Standard 3.8 Flash guardrails can trigger false refusals on legitimate security inspection, redaction, or policy code; frame technical security contexts neutrally or emit structured `ABSTAIN` rather than hallucinating workarounds. Full analysis: [`docs/notes/2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md`](docs/notes/2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md).

4. **Calibrate test breadth (prevent reasoning model test over-engineering)**:
   - Restrict test authoring to a single atomic reproduction or regression unit test demonstrating the defect or behavior change; prohibit sprawling 20-test suites when 1 witness suffices.
   - Prevent token exhaustion and context pollution by targeting only the explicit requirement or failure mode instead of writing speculative matrix permutations or redundant assertion variations.
   - Keep test execution deterministic, bounded, and fast-failing within the target package (`go test -v ./internal/<pkg>` and `go vet ./internal/<pkg>`).

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
New native-performance work prefers Qwen3.8.
Qwen3.6 is allowed only when the task states an explicit task-specific exception, such as regression, compatibility, historical comparison, or a hardware/artifact constraint.
Preserve historical Qwen3.6 artifacts; do not rename or rewrite them as Qwen3.8 evidence.

## What this project is

**fak** is an *agent kernel*: one Go binary that sits between an AI agent and the tools
it calls, and handles every tool call before it runs — reuse the shared setup, route each
call to the right model, serve repeats locally, and shed old turns while the provider's
cache survives. It is first a **performance gate** (do the shared setup work once, not
every turn — cheaper, faster, longer-running sessions) and, on the very same checkpoint, a
**security gate** (a default-deny capability floor the model can't talk past).

## Default priorities (portfolio hierarchy)

When prioritizing design, implementation, architecture, triage, or backlog work, the repository defaults to this four-tier hierarchy:

1. **fak all in one (serving and harness + memory — the "one touch" thing)**:
   The primary focus: a turnkey, single-binary runtime (`fak up`) that bundles model serving, the agent harness, and durable memory into a seamless "one touch" deployment. Everything required to run governed, cache-accelerated agentic workflows out of the box with zero glue code.
2. **fak serving only**:
   High-performance model inference runtime (`fak serve`), disaggregated gateway, KV-cache/context MMU acceleration, continuous batching, and native model execution.
3. **fak harness only**:
   Standalone agent harness and governance substrate (`fak guard`), default-deny tool capability floor, policy adjudication, and interception over external frontier models and harnesses.
4. **other things**:
   Peripheral utilities, standalone integrations, off-spine tooling, and secondary stewardship tasks not directly advancing tiers 1–3.

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

The Go module is the repository root and needs Go 1.26+ (`GOTOOLCHAIN=auto`). Zero external
dependencies means no `go.sum`.

```bash
go build ./cmd/fak        # build ./fak (fak.exe on Windows)
make build                # debuggable binary
make test-fast            # build + vet + short tests
make mac-perf             # Mac shift-left: Metal tok/s and prefill bench
make test-race            # WSL race gate
make test                 # full suite, including model witnesses
make ci                   # build + vet + test + claims lint
```

Build profiles and their one flag-delta table: [`docs/dev-tooling.md`](docs/dev-tooling.md#build-profiles).
Install: `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`.

### Which shared-tree question are you asking?

This checkout is permanently peer-dirty; raw in-place `go build ./...` or `go vet ./...` can
mix your work with peers' WIP. Use the matching isolated verb:

| Question | Command |
|---|---|
| Is committed trunk buildable and gofmt-clean? | `fak-dev ci-preflight` |
| Does trunk plus only my paths pass build, vet, and affected tests? | `fak validate --mine <path>...` |
| Does my change compile while hiding peer WIP? | `fak-dev buildcheck [--vet] [--mine <file>]` |
| Does the literal working tree compile? | `fak-dev buildcheck --isolate=false --vet` |
| Will my push break another worker's graph? | `fak hooks pre-push` |

Compile verification must never write the in-tree binary: prefer `fak-dev buildcheck`; if the
binary itself is broken, use a unique temp output. On this Windows box, Microsoft Defender
Antivirus real-time ML has repeatedly quarantined transient native Go test executables; the
August 25 audit for issue #8919 supersedes the earlier Application Control diagnosis. Run
`./test.ps1` under WSL, use fleet nodes for real serves, and follow
[`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md).
Do not disable Defender or add broad exclusions.
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
- **Runtime / CLI / Dogfood / Integration** (an executable verb, CLI command, gateway protocol
  adapter, daemon hook, or multi-component flow): the proof is **meaningful execution of the real
  path**, not an in-memory mock. Hermes' rule: "mocks hide integration bugs." For executable verbs
  or protocol layers, run the real binary or harness in dogfood (`make dogfood-test`,
  `scripts/dogfood-claude.sh --probe`, `recent_feature_dogfood.py`) or a loopback integration test
  (`make test-integration`, `*integration*test.go`) against a temporary workspace before claiming
  completion. A green unit test that only asserts mocks or verifies syntax without executing the
  real path is unproven and must not be declared done. Shift left: prove execution early during
  development, never deferring validation to post-merge or scheduled nightly runs.
- **"Shipped / done" claims**: a witnessed commit (`dos verify`, the `(fak <leaf>)` trailer) — see
  the witness rules below. A subject line is forgeable; the diff and the registry are not.

The rule of thumb: reproduce the defect as a captured artifact *first*, then make that artifact
clean. If you cannot capture it, you cannot prove you fixed it — say `not yet` with the missing
witness instead of claiming a fix.

## Track work in GitHub before implementation

Use a GitHub issue as the durable tracker for every substantive unit of work whenever
reasonably possible. Before editing code or docs, changing configuration, or launching an
implementation worker, search open and closed issues for duplicates, then claim the matching
issue or create one that states the problem, intended outcome, and witness. Put the issue
number in worker packets and keep discoveries, scope changes, and follow-ons reconciled there.

Scoping, reproduction, and read-only triage may precede the issue when needed to write an
honest ticket. Skip advance ticketing only when it would be unreasonable: the request needs no
repository change, the change is truly trivial and tracking would cost more than the work, an
urgent safety or outage response must start immediately, or GitHub is unavailable. For urgent
or offline work, create or reconcile the issue as soon as the constraint clears and record why
work began first. Do not use plans, TODOs, commit messages, or chat as substitutes when GitHub
issue tracking is reasonably available.
## New work defaults: spine first, then fan out

For every new unit, align with the default priority hierarchy (1: All-in-one [serving + harness + memory], 2: Serving only, 3: Harness only, 4: Other things), classify centrality (`Core`, `Enabling`, `Stewardship`, `Peripheral`), run
all P1-P4 checks in [`docs/problems-we-solve.md`](docs/problems-we-solve.md), and state **For /
Problem / Today / Better because / Witness** against the real next-best alternative. Then:

1. Ship the smallest working end-to-end spine in the same session: applied implementation,
   captured witness, operating-envelope proof, then optimization. Include the safety needed to
   run it. If it cannot ship confidently, file that spine first as a `gen/now` issue with its
   missing witness; never defer it silently.
2. Once it ships, create the 3..50+ QA, dogfood, productization, observability, integration,
   docs, and release follow-ons with
   `fak-dev issue fanout --title T --leaf L --spine <sha|cmd|doc> --json` (or cohort-plan first).

Full doctrine: [`docs/spine-first-defaults.md`](docs/spine-first-defaults.md) and `/spine-fanout`.
At run end, dedupe and file every real leftover as an open issue; otherwise say nothing remains.

Promote reusable successful scratch through **scratchpad → committed Go tool → `fak` verb →
captured knowledge**. Promote when it solved a real recurring task, records a non-obvious fact,
or beats the committed equivalent; keep only true probes disposable. New tooling is a Go leaf or
nested Go sub-module, not another `tools/*.py` or shell/PowerShell script, and operational facts belong in the leaf doc or dated `docs/notes/`. Allocate
and reap scratch through `fak tree-doctor`; see [`docs/generated-output-defaults.md`](docs/generated-output-defaults.md).

Close operator-facing turns with verdict-first bullets, one claim and inline evidence per line;
make the final line the next checkable step. A one-line “nothing left; pushed X” is sufficient.
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

- **Ship green work and sync safely by default.** Do more work by default: when implementing or fixing tasks yourself, do not stop at partial edits or ask conversational permission to test, sync, or push. Run the full lifecycle through to completion:
  1. **Pre-flight safe sync**: Check and synchronize safely with `origin/main` (`fak sync check`, `fak sync reconcile --apply`, or `fak sync apply`) so local work is grounded on trunk.
  2. **Atomic implementation & self-verification**: Decompose work into single-concern leaves, write/update tests, and verify on-device (`fak validate --mine <paths>`, `go test ./internal/<pkg>/...`).
  3. **Stage-and-commit by explicit path**: Lint the subject (`fak commit --preview`) and commit only your touched paths (`fak commit --path <p> -m "<subject> (fak <leaf>)"`).
  4. **Safe push unprompted**: Push verified commits via `fak sync push` or `fak commit --push`. `fak sync push` automatically retries transient non-fast-forward races (when a peer lands between fetch and push while HEAD already contains origin) and stops safely on genuine diverged states. Never wait for an operator prompt to push once the tree is green.
  Stay on `main`; never force-push, create a feature branch, use `--autostash`, or escape a dirty/diverged tree into a worktree. Merge `origin/main` in place. If a peer owns `MERGE_HEAD`, unstage your paths and wait; do not finish or abort their merge.
- **Match scope to capability.** Constrain smaller models and workers to atomic S0/S1 leaf units with single-concern boundaries and one witness. When encountering high-difficulty aspects (concurrency, frozen ABI, complex kernel algorithms), practice scoped fail-to-abstain: land partial verified evidence and escalate only the isolated high-difficulty boundary with a structured ABSTAIN record rather than guessing or emitting speculative changes. Persist through recoverable hurdles using alternate sanctioned routes or waiting out transient locks rather than abandoning the task.
- **Commit exactly one issue through explicit paths.** Prefer `fak commit --preview`, then `fak commit --path <p> ... -m "<subject>"`; use `fak sweep` for one coherent lane. The fallback is `git commit -s -m "<subject>" -- <paths>`, never `git add -A`. One issue lands in one commit and one leaf; do not split a green issue into patch commits or batch unrelated issues.
- **Make the first subject final.** Sign off with DCO, use a Conventional-Commits subject, and include a recognized `(fak <leaf>)` trailer. A peer may push your commit before an amend, so preview the subject and paths first. Demo binaries use their `cmd/<dir>` name as the leaf.
- **Preserve shared-tree buildability.** A tracked or untracked `.go` sibling enters every package build. Fence incomplete cross-file WIP with `//go:build wip_<feature>` until its symbols exist; validate only your explicit paths with `fak validate --mine`. Build verification never writes an in-tree binary.
- **Migrate CI/CD contracts atomically.** Before changing workflow inputs, JSON schemas consumed by workflows, check/job names, secrets/env, runner labels, or artifact/cache names, search the whole tree for consumers and update them together. Include changed contract, migrated consumers, impact/cutover, and rollback in the commit body; prove committed tip with `fak-dev ci-preflight`. Checklist: [`docs/ci/ci-spec-change-migration.md`](docs/ci/ci-spec-change-migration.md).
- **Use only the sanctioned detached worker worktree.** Drive isolated workers through `fak worktree worker prepare|land|reap|list`; all branch worktrees remain forbidden. Reap stale scratch worktrees with `tools/worktree_doctor.py`, which preserves live sessions. See [`docs/managed-worker-worktrees.md`](docs/managed-worker-worktrees.md) for the operator guide, portable defaults, and remote recovery runbook.
- **Allocate and reap scratch explicitly.** Use `fak tree-doctor --scratch-dir <producer>` or `--scratch-path <producer>/<file>`, then `--reap-scratch <producer> --json`. Never use `git clean -Xdf` under `_scratch/`. Per-run prompts/transcripts belong in allocated scratch; `testdata/` is only for fixtures landed with a consuming test. See [`docs/generated-output-defaults.md`](docs/generated-output-defaults.md).
- **Keep comments durable.** Explain non-obvious invariants, safety, concurrency, compatibility, or performance tradeoffs; do not narrate syntax. Preserve required exported-API, package, directive, generated, and legal comments.
- **Keep claims and gains witnessed.** Every `CLAIMS.md` row needs its required tag. Report performance only from net-true end-to-end accounting, with quality and operating-envelope constraints; setup/recovery/verification overhead counts. Use the project claim and benchmark gates rather than hand-written “looks faster” prose.
- **Extend through leaves and Go tooling.** Add capabilities with `fak new-leaf`; avoid editing the core registry directly. New durable tooling is a Go leaf or nested Go sub-module plus a `fak` verb, not another `tools/*.py`, `.ps1`, or `.sh` script; the Python and script gates permit maintenance of grandfathered scripts, not new ones.
- **Prefer Go sub-modules and applications over scripts.** Prefer making Go sub-modules (quarantined nested modules under `tools/<name>/<dep-heavy>/go.mod` with a stdlib façade) and Go applications (`cmd/` binaries or Go leaves under `internal/` registered as CLI verbs) rather than PowerShell (`.ps1`) or shell scripts (`.sh`, `.bat`, `.cmd`). Do not create random scripts; new automation, dogfood runners, harnesses, test utilities, and background tasks must be implemented in Go. The git pre-commit filter refuses un-grandfathered scripts (`FILE_ADMISSION`).
- **Keep private control private.** GPU-server credentials, hostnames, SSH details, private paths, and raw internal logs stay in the private companion repo. Public evidence must be scrubbed and reproducible through [`docs/private-comms-channel.md`](docs/private-comms-channel.md).
- **Respect Windows operational fences.** Never run broad `find /`, `find ~`, `find /mnt`, or `find /proc` from Git Bash. Route git and hang-sensitive commands through the guarded fak verbs. Treat low-utilization whole-machine stalls as kernel-path churn: stop launch storms, inspect process counts/handles, and use the committed diagnostics rather than adding retries.
- **Make external writes explicit.** `OUT_OF_TREE_WRITE` records writes outside the repo. Use the operation’s declared external target and reversible preview/confirm path; never disguise an external mutation as repository scratch.
- **Do not stop or restart a managed inference server during active sessions.** Before stopping, restarting, or reconfiguring any locally or remotely managed model inference endpoint or server, verify that all dependent sessions have completed or ownership has been explicitly handed off. Stopping an active upstream server strands dependent guarded sessions behind repeated 502 Bad Gateway errors.

The detailed recovery for a refusal is intentionally paged: query the token below instead of carrying every failure cookbook in every agent context.

### If the kernel refuses you (recover, don't fight it)

A guard refusal is per-call runtime feedback; it is not a session stop or permission to abandon the task. A worker that treats a refusal as a prompt termination converts a recoverable hurdle into an unforced failure.

A guard refusal names a token from the closed `[reasons.*]` vocabulary in
[`dos.toml`](dos.toml). Run `dos man wedge <TOKEN> --explain` instead of carrying
the full recovery cookbook in every agent context:

```bash
dos man wedge <TOKEN> --explain  # summary, category, fix, and references
fak recover <TOKEN>              # concrete recovery plan when one exists
fak recover --list               # list executable and manual-only plans
```

Recover by the named action; do not route around the guard:
1. **Execute recovery**: Run `fak recover <TOKEN>` and apply the output. Manual plans name the required remediation.
2. **Handle transient locks**: For concurrency blockers (`MERGE_IN_PROGRESS`, `COLLISION_RISK`, `LOCK_BUSY`), unstage local paths (`git restore --staged`), wait for quiescence or inspect `dos top`, checkpoint finished progress with `fak wip checkpoint` or `git tag hold/<lane>`, or pivot to an independent disjoint subtask.
3. **Use sanctioned alternative routes**: When an operation is refused (such as `FILE_ADMISSION` on a script), use the sanctioned Go tool, leaf verb, or structured file tool instead of quitting.
4. **Scope boundary blocks**: When an explicit policy or core-lock blocks a specific path (`POLICY_BLOCK`, `SELF_MODIFY`), land your safe, verified sub-components, record the structured refusal, and escalate the isolated boundary with checkable next steps.

`fak recover` is a dry-run unless `--execute` is passed, and manual-only plans refuse execution
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

## Releasing and planning

Use `/release` for the full release path. It enforces a clean synchronized `main`, full CI,
version bump, release notes, signed commit/tag, push, GitHub release, and rollback guidance.
Never hand-edit version constants, tag partially, force-push, or publish from a dirty/diverged
tree. Release history and channel contract: [`docs/releases/`](docs/releases/) and [`docs/releases-channel.md`](docs/releases-channel.md).

Choose the planning primitive by work shape:

- **Phased deliverable:** finite ordered phases with observable completion. Keep
  **Current state** accurate across the independent product, evidence, and queue axes in
  [`docs/progress-state-defaults.md`](docs/progress-state-defaults.md), append dated facts to
  **Execution log**, run `/phased-plan` each phase, and archive completed plans under
  `docs/plans/completed/`. `fak plan-audit` catches drift.
- **Ongoing program:** recurring unbounded work. Track health, cadence, and next movement rather
  than percent complete, and archive under `docs/programs/completed/` only when retired.

Plans are canonical state, not narration. A missing or failed performance receipt changes the
evidence axis; it does not erase delivered scope or force the whole item into `HOLD`. Preserve
historical `KEEP`/`REJECT`/`HOLD` decisions in the execution log, keep delivery credit separate
from performance credit, and never weaken the applicable claim gate. Update plans in the same
commit as the work; document scope changes explicitly. For parallel work, assign distinct file
sets and integrate only after all workers finish. Audit before starting a new plan when
active-plan load is high.
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
