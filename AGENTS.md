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

## Hard rules (these WILL bite an agent — they are enforced below the agent layer)

**Default: ship.** Once the tree is green, **commit AND push** unprompted.
Green = `make ci` (build + vet + test + claims-lint; on a native-Windows host run the test
suite under WSL with `./test.ps1`, since native `go test` is blocked). The commit-message,
file-admission, public-leak, and trunk guards then run automatically as git hooks at
commit/push — so "the guard passed" means CI is green *and* the commit/push was accepted.
(Those gates run in one process via `fak hooks pre-commit` / `commit-msg` when a `fak`
binary is on PATH — ~10.7s → ~0.3s vs spawning a Python interpreter per gate; the shell
hooks fall back to the `tools/check_*.py` checkers when no binary resolves, so a fresh
clone is still gated.)
The HOW below is unchanged and gates the WHEN: stay on the trunk, `git commit -s -m "…" -- <paths>`
(`-m` before `--`; never `git add -A`), merge **in place** if the trunk diverged, wait out a peer's
`MERGE_HEAD`, and **never force-push**. If a guard refuses (`OFF_TRUNK`), a peer merge is
mid-flight, or a blocker stands — reconcile in place or STOP; the default does not fire
until it clears.

Dirty shared trees are normal; land finished, green work promptly. Before reporting
done, use the repo's index-safe commit tools: `fak sweep [--json]` to group the dirty tree
by lane, then `fak sweep --apply --lane <lane> -m "<subject>" [--push]` for a whole lane
group or `fak commit --preview ...` followed by `fak commit --path <p> ... -m "<subject>"`
for a narrower change. Use raw `git commit -s -m "<subject>" -- <paths>` (`-m` before `--`, or
the message editor opens and hangs headless as `INTERACTIVE_HANG`) only as a fallback when the
binary/tooling is unavailable, and say so in the handoff.

Filing a GitHub issue from an agent session works the same way: prefer
`fak-dev issue create --title T (--body B | --body-file F)` over raw `gh issue create` —
the default body must decide `## Core through-line` and `## Gold-plating boundary` before
creation (`--raw-body` is only for a deliberate non-contract administrative issue) —
it shells to `gh` directly from the trusted binary, the same way `fak commit`/`fak
sweep` do for git, so it does not trip the reversibility/ESCALATE preview-confirm
gate on every call. Use raw `gh issue create` only as a fallback when the binary is
unavailable, and say so.

**Changing a CI/CD spec is never a local edit.** A workflow input contract, a tool's
`--json` schema that a workflow `jq`-reads, a required-check/job name, a secret/env, a
runner label, or an artifact/cache name each has a consumer in *another* file that breaks
*later* (a scheduled job, a peer's PR, a silently-wrong Slack card) — not at edit time.
Before you touch one: grep the whole live tree for every consumer, migrate them in
lockstep, and put a three-line impact statement (what changed · consumers migrated ·
impact/cutover/rollback) in the commit body. The full checklist and a worked example are
in **[docs/ci/ci-spec-change-migration.md](docs/ci/ci-spec-change-migration.md)**. Prove
the *committed* tip (not the peer-dirty tree) with `fak-dev ci-preflight`.

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree.**
  The trunk guard *refuses* off-trunk commits (the `OFF_TRUNK` law). A dirty/diverged
  tree means reconcile **in place** or STOP — never escape into a side branch.
  - *Stray worktrees still accrue* — the harness and subagents spin scratch worktrees the
    rule can't prevent. `tools/worktree_doctor.py` is the janitor: it auto-detects the
    trunk, safely prunes loss-free strays, and `--sweep-disposable` archives-then-reaps
    dead scratch worktrees (temp / scratchpad / pr-work) while sparing live sessions via a
    freshness guard. A scheduled task runs it (`tools/register_worktree_doctor.ps1`).
  - *Scratch files, not just worktrees* — agent/tool scratch belongs in the OS scratchpad
    (or a single gitignored `_scratch/`), never loose in the repo root. Do not invent a
    root-level output name and rely on `.gitignore`: allocate it first with `fak tree-doctor
    --scratch-dir <producer>` (a run directory) or `fak tree-doctor --scratch-path
    <producer>/<file>` (one generated file), then redirect there. Close one producer with
    `fak tree-doctor --reap-scratch <producer> --json`; it resolves one exact top-level
    `_scratch/<producer>` directory, refuses roots/paths/globs/reparse escapes, and receipts the
    resolved target, verdict, and removed-entry count. Never substitute
    `git clean -Xdf -- _scratch/<producer>`: Git may traverse unrelated ignored siblings because
    `_scratch/` is the ignored ancestor (#8254). `--sweep-scratch --dry-run` /
    `--sweep-scratch` remain the explicit whole-namespace preview/reap pair (#3211), not a
    producer cleanup. See
    [`docs/generated-output-defaults.md`](docs/generated-output-defaults.md).
  - *Control prompts and fixtures are durable WIP too:* `.claude/` admits reusable project
    infrastructure, not per-run residue. Put issue-numbered launch/recovery fuel and transcripts
    under an allocated `_scratch/<producer>/` or private path, then reap that producer with the
    exact `--reap-scratch <producer>` verb when the run closes.
    Put a file under `testdata/` only when a test consumes it and land both together; generated
    fixture candidates and reports stay in scratch. `fak tree-doctor` inventories untracked
    `.claude/` files as `park-or-delete` and untracked `testdata/` files as `land-or-delete`.
    See [`docs/generated-output-defaults.md`](docs/generated-output-defaults.md#control-prompts-and-test-fixtures).
  - *The one sanctioned worktree — detached, lands on `main`:* per-worker build
    isolation (#1334 / epic #3165) uses a **detached** worktree pinned at trunk HEAD
    whose diff lands on `main` through the serialized `land_worktree_diff` under the
    worker's held lane lease. Because it is detached (never a branch) and reaches the
    trunk only via that lane-serialized land, it is **not** off-trunk and never trips
    `OFF_TRUNK` — it is the *only* permitted worktree use. Drive it through the durable
    verbs `fak worktree worker prepare|land|reap|list` (or the `tools/worker_worktree.py`
    reference CLI), never a hand-rolled `git worktree add` on a branch. Everything else —
    a feature branch, or any worktree that commits off-trunk — stays forbidden.
  - *Diverged trunk (`git status` says "have diverged"):* `git fetch origin main`, then
    `git merge origin/main` **in place** and resolve. This is a shared trunk — peers
    routinely build the SAME feature under a different SHA, so most conflicts resolve to
    the trunk **superset** and the merged tree often equals HEAD (verify:
    `git diff --cached` is empty). Finish with `git commit -s --no-edit` — the merge commits
    the index as-is with its prepared message (`--no-edit` skips the editor, which would
    otherwise hang headless as `INTERACTIVE_HANG`); never `-a` / `git add -A`, which would sweep a peer's files into your
    merge. Prefer **merge over rebase**: rebase replays every local commit and re-hits the
    same conflict N times; merge resolves it once. **Never `--autostash`** (on `rebase` or
    `pull --rebase`): an aborted/conflicted rebase pops the stash back as a working-tree
    blob, dumping a peer's in-flight WIP into your tree and leaving a dangling `autostash`
    stash. Reach a clean tree first, *then* `git fetch` + `git rebase origin/main` with no
    autostash — the `gitgate` rung refuses the flag for exactly this reason. After a clean
    `git push` the pushed tip may sit *ahead* of your commit — a peer landed on the shared
    ref between commit and push; that's expected, not a force.
  - *A merge is mid-flight* (`git rev-parse -q --verify MERGE_HEAD` prints a SHA): a
    path-scoped `git commit -- <paths>` then fails with *"cannot do a partial commit during
    a merge."* If it is **your** merge, finish it promptly — peers are blocked until
    `MERGE_HEAD` clears. If it is a **peer's**, do NOT abort or complete it:
    `git restore --staged` your files, leave edits in the working tree, and wait for
    `MERGE_HEAD` to clear, then commit by explicit path.
- **Commit by explicit path** — prefer `fak commit --path <p> ...` (or `fak sweep --apply`
  for one lane group); fallback is `git commit -s -m "<subject>" -- <paths>` (`-m` before
  `--`), never `git add -A`. This is a
  shared multi-session tree; never stage a peer's uncommitted files. `fak commit --path
  <p> -m "<msg>"` mechanizes this whole rule: it stages only the named paths under an
  advisory lock, writes the message to a file (so an em-dash/multiline subject can't
  misparse as a pathspec), runs the real hooks, then **asserts the committed file set
  equals the requested paths** — refusing `PATHSPEC_RACE` (and leaving the commit intact,
  never force-pushing) if a peer swept extra files in. It also refuses `OFF_TRUNK` /
  `MERGE_IN_PROGRESS` up front, so the runbook above is a verb, not a discipline you have
  to remember. The [`/commit-clean`](.claude/skills/commit-clean/SKILL.md) skill mechanizes
  the rule end to end — lint the subject with `--preview`, stage-and-commit exactly your
  paths under the lock, and verify only your paths landed.
- **One issue, one commit; one commit, one leaf.** The atom is a whole issue's coherent
  change, landed once — not one file, not one step. Finish and *green the acceptance
  criteria before the first commit*: splitting an issue across follow-ups that re-touch the
  same files (land → patch → patch) is churn, and each re-run pays the full preflight +
  hook-audit + advisory-lock + CI cost while spreading the diff-witnessed close over N SHAs
  (#3848 landed one audit change in three commits, `crossaudit.go` re-touched in all three).
  Don't swing the other way either — batching *different* issues into one commit blurs the
  closure gate, which binds one `#N` to one diff-witnessed SHA (over 300 commits, one does
  this; the rarity is the point). When a lane you own carries many dirty paths from finished
  work, `fak sweep` groups them by leaf so a coherent slice commits together (a lane over ~10
  paths splits into directory-coherent sub-units — small, but each still a whole change).
- **Sign off every commit** — `git commit -s` (DCO). Use a Conventional-Commits subject
  with a `(fak <leaf>)` trailer; a docs-only change uses a `docs(scope):` subject.
  A `cmd/` **demo or binary** has no `internal/<name>/` package, so stamp it with its
  directory name — `(fak <dir>)` for `cmd/<dir>/` (e.g. `(fak turntaxdemo)`). The leaf
  binds to the `cmd` lane (which owns `cmd/**` as one tree) and keeps per-demo attribution
  in the subject; `tools/commit_stamp_doctor.py` recognizes any real `cmd/<dir>` leaf, so a
  residual off-lane warning means a genuine typo, not a `cmd/` demo (#518).
  *Check the subject BEFORE you commit:* `fak commit --preview -m "<subject>" --path <p> …`
  lints the message + paths without touching git — is it witness-gradeable, does it carry a
  bindable `(fak <leaf>)` stamp, and does the leaf match the lane those paths live in? It
  catches a noun-led subject, a missing/typo'd trailer, or a stamp/lane mismatch up front,
  the only place you can fix them: on the shared trunk a peer can push your local commit
  before you amend, so the FIRST subject has to be right (exit 0 clean / 1 issues / 2 usage).
- **Leave the shared tree buildable — gate not-yet-compiling WIP behind a build tag.**
  `go build ./cmd/fak` must stay green on the shared trunk regardless of your uncommitted
  work. A tracked **or untracked** `.go` file that references a symbol you have not committed
  yet reds the whole package for every *other* session (they cannot rebuild `fak` to pick up
  a new flag — e.g. `accounts --adopt`) and for CI, because `go build` compiles every
  non-test `.go` file in the package, committed or not. If a cross-file feature needs
  multi-file WIP that does not yet compile against committed symbols, fence each
  not-yet-buildable file behind a build tag — first line `//go:build wip_<feature>`, a blank
  line, then `package …` — so the default build (and every peer) stays green while the work
  lives on disk; drop the tag once the defining symbol lands, and build the WIP meanwhile
  with `go build -tags wip_<feature> ./cmd/fak`. The `internal/buildwitness` test enforces
  this: it runs `go build ./cmd/fak` under default tags and fails with the exact undefined
  symbol when it is red — the durable guard for a recurring class (#3217 #3127 #2251 #1325),
  not another one-off instance patch.
- **Keep code comments succinct and durable.** Do not narrate syntax, restate a function name,
  or leave step-by-step "now we" commentary that the code already makes clear. Comment the
  non-obvious **why**: invariants, safety/security boundaries, concurrency, compatibility,
  performance tradeoffs, and surprising operational constraints. Preserve required exported-API,
  package, directive, generated, and legal comments. Prefer clearer names or a small helper over a
  paragraph; move durable tutorials and runbooks to docs. The changed-lines-only
  `COMMENT_QUALITY` pre-commit gate is advisory by default (`FLEET_COMMENT_QUALITY_GUARD=block`
  opts into enforcement) because prose quality is contextual and must not become a brittle blocker.
- **Every claim carries a tag.** Each `- [` line in [`fak/CLAIMS.md`](CLAIMS.md) must
  carry exactly one of `[SHIPPED]` / `[SIMULATED]` / `[STUB]` (lint-enforced by
  `make claims-lint`). Don't overclaim; the repo keeps an honesty ledger.
- **A gain is net-true or it isn't reported.** Before you claim an efficiency/perf win —
  yours, or one you read in a paper — run it through the
  [net-true-value standard](docs/standards/net-true-value.md): measured against the *real*
  alternative (not a strawman), net of the cost it adds, scope stated, provenance-labeled
  (witnessed/observed/modeled), and reproducible — no witness ⇒ `not yet`. Quote the tuned
  baseline as the headline, never the naive one (the `A=naive / B=tuned / C=fak` letters in
  [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) are the mechanical form). Grade a claim
  mechanically with `fak claim-check` (the verb the standard names): it takes a claim +
  baseline + witness and returns `net-true` / `strawman` / `not-yet` against the six questions
  (exit 0 / 3 / 3); `fak claim-check --self-test` grades the built-in honest+strawman corpus.
- **Add a feature as a leaf, not a core edit.** `fak new-leaf <name> --tier
  <tier> [--register]` stamps a conforming skeleton; the frozen ABI (`fak/internal/abi`)
  is additive-only and human-owned. `internal/architest` fails the build on a bad import.
- **New tooling is Go, not Python.** The repo is a Go project; the ~560 `tools/*.py` scripts
  are a *grandfathered* baseline, frozen - not a pattern to copy. A new tool ships as a `fak`
  subcommand (pure logic in `internal/<name>/`, a thin shell in `cmd/fak/<name>.go` - see
  `cmd/fak/stallscan.go`) or a `cmd/<name>/` binary, never a new `tools/*.py`. The
  `internal/pythongate` ratchet (`go test ./internal/pythongate -run TestNoNewPythonTools`)
  reds the trunk on any `tools/*.py` outside the baseline, and porting a grandfathered script
  to Go shrinks that baseline - the ratchet only ever tightens. When you *touch* a `tools/*.py`
  for non-trivial work, default to porting it to Go in the same pass (`REASON_NEW_PYTHON_TOOL`).
  **Dependency-heavy Go tools use a nested-module quarantine, not Python and not a root dependency.**
  Put the public façade at `tools/<name>/` and keep it stdlib-only; put dependency-heavy
  implementation code beneath it (for example `tools/<name>/terminal/go.mod`). The façade
  preserves the root invocation (`go run ./tools/<name> ...`) while invoking the nested
  module explicitly. Never add a root requirement to make a tool compile. The
  `internal/dependencyquarantine` gate pins the reviewed root require/checksum sets, walks
  the repository for nested `go.mod` files, rejects non-stdlib façade imports, and runs
  every discovered nested module under CI. A dependency-budget change requires an explicit
  allowlist update and review; a new nested module requires no central enumeration.
- **GPU-server private control is private; public evidence is scrubbed.** Benchmark results and
  runbooks can live here once scrubbed to generic GPU-server language, but live private
  control code belongs in `fak-private`: private bridge/control packages, private cleanup
  helpers, and sunset private bridge tooling. See
  [`docs/gpu-server-private-boundary.md`](docs/gpu-server-private-boundary.md). **To actually reach the
  channel** (the private control bridge to the lab GPU servers), start at the public stub
  [`docs/private-comms-channel.md`](docs/private-comms-channel.md) — it points to the live
  plumbing in `fak-private` (checked out at `../fak-private`).
- **Never `find /` (also `find ~`, `find /mnt`, `find /proc`) in Git Bash on Windows.**
  `/` descends into `/proc/registry*` (the whole Windows Registry, x3 views) and `/mnt/c`
  (all of `C:`, which holds self-referential junction loops); MSYS `find` can't detect the
  cycles, so it recurses for hours and leaks millions of handles (it took down this box on
  2026-06-21 — two orphaned finds held 98.8% of system handles). Search with `rg`
  (`rg --files | rg <pat>`) or anchor **and** bound: `find /c/work/fak -xdev -maxdepth 8 …`,
  `timeout`-wrapped. Backstop: `tools/runaway_process_reaper.ps1` reaps stragglers; audit
  anytime with `tools/runaway_process_scan.ps1`.
- **Random whole-machine stalls at "low usage" are kernel-path CHURN, not disk/RAM.** When
  the box locks up for a beat while CPU/RAM/disk read fine, the cause is soft-fault + process-
  spawn storms saturating the scheduler/page-fault locks — invisible to every usage meter
  (diagnosed 2026-07-07, issue #3153). Fingerprint it with **`fak stallscan`** (one-shot) or
  **`fak stallscan --watch`** (rolling JSONL self-monitor): it reads the churn axes (soft vs
  hard faults, ctx-switch/syscall rate, spawn bursts) and names the cause, ruling out disk/RAM.
  Run it in the background by default with `tools/fak_stall_monitor.ps1 -Install` (logon task,
  `-AutoMitigate` to auto-tame the floor). The non-fak daemon floor (AMD AUEPMaster telemetry,
  Defender, CCleaner) is tamed reversibly by `tools/host_stall_mitigations.ps1` (preview by
  default, `-Apply` to act). Fak's own churn sources track under #3153 (#3154-#3157).
- **On this Windows host, run git (and anything you can't afford to hang) through the
  PowerShell tool, not the Bash tool.** MSYS Bash stalls intermittently: `git diff`,
  `git log`, `git status`, even `git remote -v` sit until the 120s tool timeout and die
  with exit 143 — `--no-pager` / `-c core.pager=cat` do not save them (a 2026-07-01
  trajectory audit counted 11 such kills across five sessions in two days). The same
  commands via PowerShell return promptly and fail *fast* (exit 128) when a peer holds
  `index.lock`, which is recoverable; a silent two-minute hang is not. Relatedly, never
  foreground-`sleep` to poll for slow work — the harness blocks `sleep N; <cmd>` chains;
  run the wait as a background task or monitor and let it notify you.
- **Writes that resolve *outside* the repo are *flagged* (`OUT_OF_TREE_WRITE`) — recorded,
  not refused, by default.** The `repo-guard` PreToolUse hook (Go binary `cmd/repoguard`, on by
  default on a fleet host) resolves a Bash/Write/Edit op whose target escapes the workspace — a
  `../sibling` path or an absolute `/c/.../work/other-repo`. `work/` holds many sibling repos, so
  a one-level escape lands in *another* project. Prefer a temp dir or an in-repo path, not `..`.
  But because cross-repo work is routine on a fleet host, the **default severity for
  `OUT_OF_TREE_WRITE` is `record`**: the crossing is written to the decision journal (countable
  via `repoguard --summary`) and the call **proceeds silently** — nothing lands in your context.
  A security-minded operator dials it back up with `FAK_REPO_GUARD_SEVERITY=OUT_OF_TREE_WRITE=deny`
  (then it hard-blocks); the master switch `FAK_REPO_GUARD=warn|off` still overrides. Allowed
  without even a record: the null/std-stream sinks (`> /dev/null`, `> /dev/stderr`) and the paired
  `fak-private` companion repo. This is the **write-time** half of the public/private split; the
  **commit-time** half — `FILE_ADMISSION` (`check_committed_files.py`) and `PUBLIC_LEAK`
  (`scrub_public_copy.py`) — keeps private *content* out of the public history, and those stay
  hard gates. Full doc: [`docs/repo-guard.md`](docs/repo-guard.md).

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
