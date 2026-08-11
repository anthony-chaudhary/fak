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
| Is the **committed trunk** buildable + gofmt-clean? *(what CI gates — what "clean **git** build" means)* | `fak ci-preflight` |
| Does the committed tip plus **only my explicit uncommitted paths** pass full build/vet and affected tests? | `fak validate --mine <path> [--mine <path>...]` |
| Does **my change** compile, ignoring peers' broken untracked WIP? | `fak buildcheck [--vet]` |
| Does the **literal working tree** compile — *my own* untracked WIP included? | `fak buildcheck --isolate=false --vet` |
| Will my **push** red another worker's build graph? | `fak hooks pre-push` |

`fak validate` archives the committed tip, overlays only repeatable explicit `--mine` paths, then runs
full `go build ./...`, `go vet ./...`, and dependency-aware affected tests; it never infers ownership
from the shared dirty tree.
`fak ci-preflight` archives the tip to a throwaway checkout (immune to the dirty tree;
`--skip-build` = gofmt-only fast path). `fak buildcheck` discards output to the null device
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
> what `fak buildcheck`'s per-process temp overlay avoids. The in-tree `go build -o fak` above
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
checklist in the `/spine-fanout` skill. Before either, pass the
[Feynman-simple value frame](docs/shift-left-task-organization.md): name **For / Problem /
Today / Better because** in plain language. Compare against the real next-best alternative,
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
   ships, run `fak issue fanout --title T --leaf L --spine <sha|cmd|doc> --json` to expand
   the QA / dogfood / productization / observability / integration / docs / release taxonomy
   into contract-ready candidates (every one dispatchable under `fak issue contract`), then
   file them with milestone + labels at creation, or wave-plan first via
   `fak issue cohort --from-plan`. The verb refuses to plan without a spine witness — that
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
`fak issue create --title T (--body B | --body-file F)` over raw `gh issue create` —
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
the *committed* tip (not the peer-dirty tree) with `fak ci-preflight`.

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree.**
  The trunk guard *refuses* off-trunk commits (the `OFF_TRUNK` law). A dirty/diverged
  tree means reconcile **in place** or STOP — never escape into a side branch.
  - *Stray worktrees still accrue* — the harness and subagents spin scratch worktrees the
    rule can't prevent. `tools/worktree_doctor.py` is the janitor: it auto-detects the
    trunk, safely prunes loss-free strays, and `--sweep-disposable` archives-then-reaps
    dead scratch worktrees (temp / scratchpad / pr-work) while sparing live sessions via a
    freshness guard. A scheduled task runs it (`tools/register_worktree_doctor.ps1`).
  - *Scratch files, not just worktrees* — agent/tool scratch belongs in the OS scratchpad
    (or a single gitignored `_scratch/`), never loose in the repo root. `fak treedoctor
    --sweep-scratch` reaps gitignored scratch via `git clean -Xdf` (ignored-only: it can
    never touch a tracked file or a real untracked WIP file); `--sweep-scratch --dry-run`
    previews (`git clean -Xdn`) before reaping (#3211).
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

A guard refusal names a token from a **closed vocabulary** — declared as `[reasons.*]`
blocks in [`dos.toml`](dos.toml), each with a `summary` + a `fix` you can look up live
with `dos check-reason <TOKEN>`. Eleven of those tokens also carry a **recovery plan built
into the binary**: `fak recover <TOKEN>` prints the concrete commands for that token —
dry-run by default, `--execute` runs only the steps the plan marks safe, and a manual-only
plan refuses `--execute` with exit 3 rather than guessing. `fak recover --list` shows which
tokens it knows and whether each is `executable` or `manual`; the token argument is
case- and separator-insensitive, so `fak recover off-trunk` resolves to `OFF_TRUNK`. When
you hit one, recover by the action below; don't route around the guard (that just trips the
next one).

| Token | What tripped it | Recover by |
|---|---|---|
| `OFF_TRUNK` | you branched / spun a worktree instead of committing to `main` | commit directly to `main` with `git commit -s -m "<subject>" -- <paths>` (`-m` before `--`); a dirty/diverged tree means merge **in place** or STOP — never escape into a side branch |
| `LOOP_DONE_UNWITNESSED` | a loop turn claimed done, but its configured external witness did not corroborate the done effect | re-arm the loop with this token as feedback, satisfy the witness criterion (`dos commit-audit`, `dos verify`, `dos test-witness`, or a registered witness), then re-check |
| `STALE_BASE_DELETION` | a pathspec commit would silently drop a peer-added block because the working-tree copy is stale relative to `origin/<trunk>` | fetch and merge/rebase `origin/<trunk>` in place so the working tree includes the peer block, then re-commit by explicit path |
| `FRESH_DELETION` | a staged commit deletes a path added within the recent trunk window, but the commit message does not mention that path | restore the path if the deletion is collateral; if intentional, name the deleted path in the commit message or override once with `FLEET_ALLOW_FRESH_DELETE=1` |
| `MESSAGE_RACE` | a safecommit operation landed a commit whose subject/body differs from the requested message | do not push it as verified; inspect the intact commit, recover through a witnessed follow-up/operator path, and avoid raw `git commit` on the shared tree |
| `NEVER_AMEND_SHARED` | a git command would rewrite shared trunk history (`commit --amend`, rebase, `pull --rebase`, or force-push) | do not amend, rebase, or force-push in the shared tree. Make a new path-scoped commit, or fetch and merge the configured trunk in place; push only with a plain fast-forward push. If an existing commit has only a message typo, leave that message intact—shared history has no compliant rewrite path—and validate future subjects first with `fak commit --preview`. |
| `ARCH_LAYER_VIOLATION` | an upward/cross-tier import, or a new leaf with no declared tier | invert the dependency through a registration seam, or push the shared type down a layer; declare a new leaf's tier (`fak new-leaf`). Floor: `internal/architest` |
| `OUT_OF_DIRECTION` | request-path logic in an untyped language, or a non-Go package blank-imported into the kernel | keep the request path Go-only; a non-Go seam stays off-path behind a typed, re-validated boundary. Floor: architest `TestHotPathHasNoExec` |
| `UNAUTHENTICATED_OFF_HOST_BIND` | `fak serve` was asked to bind an address reachable from off this host (`0.0.0.0`, `::`, a bare `:port`, a routable IP) while neither `--require-key-env` nor `--key-principal` configures an inbound token door | arm a token door (`--require-key-env FAK_API_KEY`, or `--key-principal <tenant>=<ENV_VAR>`) with the env var set, or keep the listener local with `--addr 127.0.0.1:8080`. For a deliberate isolated segment where a host firewall does the work, `--unsafe-allow-unauthenticated-bind` binds anyway and warns loudly every boot. Loopback, `--stdio`, and an authenticated off-host bind are never refused. Floor: `cmd/fak/serve_bind_safety.go` |
| `FILE_ADMISSION` | a staged path is private-only content, a **noisy one-off operational artifact** (GPU reserve/availability status, dispatch telemetry, scratch dump — by the `fak:operator-private` marker or the loose-ops-doc name backstop), regenerable junk, or an oversized blob | move private-only code + operator-only status to `fak-private`; mark a one-off ops doc `fak:operator-private` (or gitignore it); a genuine curated note goes under `docs/notes/` in scrubbed language; drop or gitignore junk; put real data under `experiments/` or `testdata/`. See [`docs/gpu-server-private-boundary.md`](docs/gpu-server-private-boundary.md) |
| `PUBLIC_LEAK` | staged content matches a redact-needle | remove or redact the needle before committing; `FLEET_ALLOW_LEAK=1` overrides once, only for an intentional adversarial fixture |
| `OUT_OF_TREE_WRITE` | a write op escaped the repo into a sibling tree, **and** the operator dialed this reason up to `deny` (default is silent-`record`, which does not refuse) | operate inside the workspace; send scratch to a temp dir, never `..`. The default already allows it; a `deny` here is a deliberate operator policy — respect it. See [`docs/repo-guard.md`](docs/repo-guard.md) |
| `INTERACTIVE_HANG` | the command waits for a human prompt/editor/login while the session has no TTY | re-run the non-interactive equivalent: pass `-m` for commits, use path-explicit staging instead of interactive add, provide tokens non-interactively, and use harness Edit/Write tools instead of terminal editors. |
| `FOREGROUND_SLEEP` | a single foreground sleep timer at or over 120s holds the tool turn open doing nothing | wait in the background instead: use a background monitor/poll loop, the Monitor tool, or ScheduleWakeup rather than a foreground `sleep`/`Start-Sleep`. |
| `LIVE_MONITOR_OUTPUT_READ` | a Read targets a live Monitor's `tasks/<id>.output` before the harness has materialized it — Monitor writes that file only when the stream ends, unlike a background Bash task, so polling it early yields a bare `ENOENT`, not a "not ready yet" hint | do not Read-poll the path while the Monitor is live; consume the pushed Monitor events and wait for its completion notification. Floor: `internal/repoguard/monitor_read.go` (PreToolUse hook via `cmd/repoguard`). |
| `WORKSPACE_PATH_UNMAPPED` | a Bash `cd` targets the workspace root with its Windows drive letter dropped (`cd /work/fak` for a `C:/work/fak` checkout) — the drive-less path resolves nowhere on the git-bash host, so the `cd` fails `No such file or directory` and aborts any `&&` chain after it | your shell already starts **in** the workspace — drop the redundant `cd`; if you must, use the host path the finding pre-fills (`cd 'C:/work/fak'`), never the drive-less form. Floor: `internal/repoguard/cdmap.go` (PreToolUse hook via `cmd/repoguard`). |
| `STALE_RECALL` | a loop is about to act on recalled status/plan memory whose witness is stale relative to git or the loop ledger | refresh from the source witness (`dos status`, `dos verify`, `dos commit-audit`, or current git ref), discard the stale recall, then retry |
| `COLLISION_RISK` | a dispatch or worker launch would overlap a live lease/region and risks two agents mutating the same file tree | wait for the lease, repartition, or pick a disjoint lane/region after checking `dos arbitrate` / `dos top`. Note (#5056, retracted liveness diagnosis): arbitrate's GO is an admission-view answer, not a promise `dos lease-lane acquire` will grant the lane — the acquire verb's lock-held verdict is authoritative, and a GO-then-refuse is by design, not a bug (see docs/agentic-issue-dispatch.md § "A GO is an admission-view answer") |
| `GATE_LATENCY_REGRESSION` | a guard or DOS gate grew slow enough to threaten the loop's dev-ex latency budget | measure the gate with the relevant hook/status benchmark, fix or budget the added latency, then rerun before claiming the loop is healthy |
| `CORE_SELF_MODIFY` | a pathspec commit touches hard-self core-lock machinery — the admission, witness, or ship-grade surfaces that would judge their own bad edit | do not clear this by self-report; use an operator/release maintenance path, or rerun `fak commit` with `--core-lock-maintenance-witness <claim>` only after an independent witness resolver confirms the maintenance claim |
| `FRONTIERSWE_SCORE_PARITY_FAILED` | a FrontierSWE fak arm is faster only by losing quality, so a time-to-solution win claim is refused | rerun or fix the fak-routed arm until `ScoreParity(raw_trials, fak_trials)` passes: fak Avg/Best gated score, full-correct count, and speedup distribution must not regress raw. Floor: `go test ./internal/frontierswe -run TestScoreParity` |
| `OVERHEAD_BUDGET_EXCEEDED` | a lifecycle rung's measured overhead exceeded its declared elapsed-time or token budget envelope | confirm the breach against the rung's declared envelope (`internal/turntaxmeter.DefaultBudget`); fix or re-budget a real regression, and judge red/green from the net-of-savings line when the rung also saves more than it costs. Floor: `go test ./internal/turntaxmeter` |
| `RUN_STATUS_CLAIMED_FIELD` | `dos status` returned a run digest containing a `claimed` field, violating the witnessed-status contract | fix the status producer so peers read only liveness, ledger-verified progress, lease region, and resume evidence; do not consume claimed status |
| `INDETERMINATE` | a verification ladder could not conclusively decide the claim cheaply and no costlier rung was available, so the residual verdict fails closed | consult a stronger rung (`require-witness`, CI, git evidence, isolated-worktree keep-bit, or human escalation), or accept the fail-closed deny; for repeat cases, declare a sufficient rung for the claim's risk class in `dos.toml` `[ladder]`. Floor: `go test ./internal/kernel` |
| `L3_CROSS_TENANT_SCOPE_DENIED` | a cross-tenant reader asked for an L3 cache page whose `ShareScope` does not reach across the trust boundary — a page private to one agent (`ScopeAgent`) or bound to the owner's tenant (`ScopeTenant`), fetched by a *different* tenant (a capability check, not a namespace-prefix match) | the page is not shareable across tenants — only an owner-marked fleet/public (`ScopeFleet`) page crosses. Mark the prefix `ScopeFleet` at the producer if it is genuinely public, or serve the reader its own tenant-scoped page. Floor: `go test ./internal/gateway -run TestL3Share_CrossTenant` |
| `L3_PAGE_DIGEST_MISMATCH` | the bytes an L3 `get` returned do not hash to the digest the page claims (`Ref.Digest`) — a corrupt or mis-tagged page the semantics-free store cannot detect, refused for *every* reader (same-tenant included) | do not admit the page — the content address does not match its bytes. Re-fetch / re-key at the store, or re-mint the page so its `Ref.Digest` is the true `hex(sha256)` of its content. Floor: `go test ./internal/gateway -run TestL3Share_DigestMismatch` |
| `ASSUMPTION_STALE` | a guarded assumption's backing evidence aged past its declared freshness bound relative to the witness's source of truth | re-witness from its declared source (`fak assume check <id>` gathers fresh evidence), discard the stale evidence, then retry once the fresh witness confirms HOLDS. Floor: `internal/assumecheck` |
| `ASSUMPTION_UNVERIFIABLE` | a guarded assumption could not be witnessed either way (the witness could not run, wrong witness kind, or a self-report only) — fail-closed: unverifiable is NOT holds | wire or repair the assumption's declared witness so it can adjudicate, gather fresh evidence, then re-run `fak assume check <id>` until the outcome is HOLDS. Floor: `internal/assumecheck` |
| `ASSUMPTION_VIOLATED` | a guarded assumption's declared witness REFUTED it — the assumed condition is provably false, so proceeding on it is refused | fix the violated condition (or retire the assumption), re-witness with `fak assume check <id>` until the outcome is HOLDS, then retry the refused action. Floor: `internal/assumecheck` |
| `BARE_COMMIT_SWEEP` | a raw `git commit` / `git add -A && git commit` reached the pre-commit hook with no vetted handshake, folding the entire staged index — including hunks you never declared — into one shared-trunk sweep | commit by explicit pathspec so only your files land: `fak commit --path <yours> -m …` (sets the vetted marker) or `git commit -- <yours>`. One-shot escape: `ALLOW_BARE_COMMIT=1`; soften fleet-wide: `FLEET_BARE_COMMIT_GUARD=warn` |
| `BLOCKED_BY_KNOWN_BAD` | an issue's declared paths intersect a live fleet known-bad signature, so dispatching it would send a worker into a known shared failure | wait for the signature to clear (auto-release, W6) or land the elected fixer (W5); a disjoint issue keeps dispatching. Floor: `fak dispatch route` over `internal/knownbad` |
| `BLOCKED_BY_OPEN_PREREQ` | a dispatchable leaf declared a still-open prerequisite (`depends-on:/blocked-by: #N`), so dispatch would run ahead of its dependency | let the named prerequisite close (the hold self-clears the next tick) or break the dependency edge in the issue body; a leaf whose prerequisites are all closed dispatches normally. Floor: `fak dispatch graph` over `internal/dispatchorder` |
| `BROADCAST_MALFORMED` | a fleet/lane-scoped broadcast had no applicable shape (empty selector, nil session table, unknown op, or a non-lifecycle op with no payload) and was refused whole at the edge | state at least one selector axis (`--lane` / `--wave` / `--label`) and fan only a lifecycle op (pause\|resume\|cancel\|terminate\|throttle). Floor: `internal/sessionctl` Broadcast validation |
| `CACHE_PREFIX_RESIDENT` | TOON auto-fire declined — the span is already inside a cached prefix, so re-encoding busts a 0.1x cache read into a full-price recompute | keep JSON — the span is cache-resident; auto-fire TOON only on spans OUTSIDE the resident prefix. Floor: `go test ./internal/toon -run TestDecide` |
| `CHECKER_TAMPERED` | a run's declared checker/test file changed bytes between declare-time and grade-time, so the grade is judged against a checker that moved and is not evidence | restore the checker to its declared bytes and re-grade, or — if the change was intentional — re-declare the run so its baseline re-pins against the new checker before grading. Floor: `go test ./internal/safecommit` |
| `CONTROL_REV_STALE` | an optimistic-concurrency (`--if-rev`) control write lost its CAS race — the drive record moved to a newer Rev between your read and your write | re-read the session's current state (Rev included) and retry the control op against the live revision. Floor: `internal/session` CompareAndSet |
| `CONTROL_SESSION_TERMINAL` | an out-of-band control op (cancel/pause/resume/throttle/budget/pace/priority) targeted a TERMINAL (Stopped) session, which rejects every drive-state write | start a fresh continuation instead of reviving the terminal record (`session.Table.Recontinue` / a new session), or target the live trace you meant. Floor: `internal/session` ControlRefusalFor |
| `CRASH_RESTART_EXHAUSTED` | the guarded harness crash-looped without shipping progress and exhausted its safe in-place restart runway | inspect the CHILD_CRASH rows with `fak guard restart-audit`, repair the systematic harness fault, then start a fresh guarded session instead; do not raise the limit to mask the loop |
| `DOOM_LOOP` | a live worker keeps burning turns/tokens while verified forward progress stays flat for K consecutive windows — a confirmed doom loop, not a mere stall | deliver the graduated correction the classifier recommends — a soft re-anchor NUDGE to the steer channel first (reversible), then an operator ESCALATE if it persists; never auto-teardown. Floor: `fak doomloop scan` / `internal/doomloop` |
| `FOCUS_WIP_SATURATED` | a dispatch spawn would OPEN a new concurrent objective while fleet breadth is at/over the focusscore WIP cap | converge before broadening — pause or meet an active objective to get back under the cap, or continue an already-open objective (never held) instead of opening a new one. Tune with `--focus-hold`. Floor: `go test ./internal/dispatchtick` |
| `GATE_PRESSURE` | the spawn preflight refused because degraded gate health (a guard-hook p99 over budget, or a standing overhead-budget breach) is the sole binding admission term | let the fleet drain and re-measure — reduce the hook path taxing every tool call or fix the over-budget rung until the p99 recovers under budget, then it admits again. Floor: `go test ./internal/dispatchtick` |
| `HOST_CHURN_BACKOFF` | the spawn preflight refused because a whole-host process-spawn storm (several dispatchers co-launching waves in one window) froze the fleet at its live count | let the burst drain and stagger the dispatchers so they do not fire waves at the same instant; it re-admits once host spawns fall below the threshold. Tune with `FAK_CHURN_BURST_THRESHOLD`. Floor: `go test ./internal/dispatchtick` |
| `ISSUEFANOUT_CONTRACT_REFUSED` | the issue-fanout planner rejected an input that violates its contract (missing title/leaf/spine_ref, `--max` below the fan-out floor, unknown `--area`, or a candidate failing the private-boundary screen) | fix the caller's input — supply `spine_ref`, keep `--max` at/above MinFanout, pass a known `--area` (`AreaNames()`), and keep every leaf inside the marker-key alphabet with no private-boundary text. Floor: `internal/issuefanout` |
| `KNOWN_BAD_ALREADY_CLAIMED` | a `fak knownbad claim` lost the fixer election — another agent already holds the exclusive lease as this signature's sole fixer | park behind the named fixer and wait for the witness-gated auto-release (W6); do not open a competing fix. Floor: `fak knownbad claim <signature>` over `internal/leaseref` |
| `KNOWN_BAD_EXPIRED_OR_REVOKED` | a `fak knownbad claim/resolve/revoke` targeted a signature no longer live (its TTL lapsed, an operator revoked it, or it was already resolved) | stop chasing the aged-out signature — the hold has lifted, so the intersecting work is dispatchable again; if the failure is genuinely still live it re-fires with a fresh signature, so act on THAT instead. Floor: `fak knownbad match` |
| `KNOWN_BAD_NOT_WITNESSED` | a `fak knownbad resolve` was refused because the fix is not witnessed — no green over the broken tree and no `dos verify` binding the fixer's commit | land the real fix, then re-run `fak knownbad resolve <signature> --witness tests\|verify` so the witness passes; only a witnessed green flips open → resolved. Floor: `internal/knownbad` |
| `L3_UNWITNESSED_FLEET_STAMP` | a ScopeFleet stamp was requested for a prefix the fleet-share registry never declared fleet-shareable — an unwitnessed boundary-crossing stamp | declare the prefix fleet-shareable at the producer (`FleetShareRegistry.DeclareFleetShareable`) if it is genuinely public, or stamp the page ScopeAgent/ScopeTenant instead. Floor: `go test ./internal/gateway -run TestL3Share_UnwitnessedFleetStampRefused` |
| `LESSON_OVERCLAIMS` | a distilled look-ahead Lesson asserted more authority than its rollout evidence earned (a FACT from below-W3 evidence, or any lesson from a W1/W0 rollout) | emit the lesson at the authority its evidence proves — a W2 rollout may flag a RISK, not a FACT; earn W3 (a witnessed commit / green suite / benchmark row) before asserting a FACT. Floor: `internal/lookahead` |
| `MODEL_TOON_UNFIT` | TOON auto-fire declined — the target model's measured TOON fitness is below the phi bar and a primer will not close the gap within budget | keep JSON for this model; route TOON only to models whose fitness clears the bar (or where an in-budget primer closes it). Floor: `go test ./internal/toon -run TestDecide` |
| `NET_TOKENS_NONPOSITIVE` | TOON auto-fire declined — the caller's real tokenizer measured TOON tokens ≥ JSON tokens − margin, so there is no net win | keep JSON — the measured token delta is non-positive; do not override the never-fire-at-a-loss invariant with an estimate. Floor: `go test ./internal/toon -run TestDecideNeverFiresAtALoss` |
| `OBJECTIVE_SCORER_MISSING` | a dispatch objective has no attached witnessed scorer | add a non-empty Witness section to the issue contract naming the independent progress evidence — a scorer-less objective is refused — then dispatch again |
| `OUTPUT_DIRECTION` | TOON auto-fire declined — the payload is a response schema the model must PRODUCE, and model TOON *output* support is weak | keep JSON — TOON auto-fire is input-only; only encode inbound (model-consumed) spans. Floor: `go test ./internal/toon -run TestDecide` |
| `PAYLOAD_TOO_SMALL` | TOON auto-fire declined — rows < R_min or bytes < B_min, so the fixed header overhead is not amortized | keep JSON — the payload is below the amortization floor; auto-fire only once rows/bytes clear R_min/B_min. Floor: `go test ./internal/toon -run TestDecide` |
| `RATE_LIMIT_BACKOFF` | the spawn preflight refused because a burst of genuine concurrency rate-limit walls (transient 429/529 overload) on this backend is the sole binding admission term | let the throttled backend drain and route new work to a DIFFERENT provider/backend while the 429 burst ages out; it re-admits once recent `rate_limit` exits fall below the threshold. Tune with `FAK_RATELIMIT_MIN_429`. Floor: `go test ./internal/dispatchtick` |
| `REDIRECT_MALFORMED` | an out-of-band redirect / set-objective op had no applicable shape (an empty goal or an empty session trace) and was refused at the enqueue edge | send the redirect with a non-empty goal and the target session's trace id; freeform prose belongs on the steer escape hatch, not the structured redirect. Floor: `internal/sessionctl` Redirect.Validate |
| `REDIRECT_NO_REDIRECTABLE_STATE` | an out-of-band redirect targeted a session whose objective is TERMINAL (met/abandoned) — there is no live objective to redirect | start a new session/objective for the new goal instead of redirecting the finished one; redirect only an active or paused objective. Floor: `internal/sessionctl` EnqueueRedirect |
| `REQUIRE_WITNESS` | a reversibility preview-confirm gate paused an irreversible or outward-facing call, pending an echoed confirm token or a compiled sidestep — a pause, not a terminal denial | prefer the sanctioned compiled sidestep the refusal names (e.g. `fak sync push` for a gated `git push`); otherwise re-propose the same call byte-identical with only the `_fak_confirm` key added (the command text binds the token, not the free-text description) |
| `RESUME_COST_EXCEEDED` | a session's observed post-resume provider spend crossed its declared recovery-cost envelope, so a new automatic resume is deferred at a reversible hold | confirm the recovery is worth continuing, then force this launch past the cost hold (the operator force bit) or raise the session's `RecoveryCostCap`; inspect observed spend on `internal/resumemetrics`. Floor: `internal/resume` |
| `ROUNDTRIP_LOSSY` | TOON auto-fire declined — `Decode(Encode(payload)) != payload`, so the TOON encoding is not provably reversible | keep JSON — fix the codec or leave the payload as JSON; never emit an unverifiable encoding. Floor: `go test ./internal/toon -run TestDecide` |
| `STEER_NO_OWNED_LOOP` | a steer was posted to a session served on the default PROXY path, which forwards a single upstream turn and owns no agent loop to drain the steer bus | deliver the steer to a session driven by an owned native loop (`fak serve --native`, which drains the a2achan bus at each turn boundary), or hand the input to the harness that owns this session's turn loop instead |
| `TABULAR_ELIGIBILITY_LOW` | TOON auto-fire declined — the payload is nested/non-uniform (tabular eligibility below the threshold), which collapses TOON accuracy | keep JSON for this span; auto-fire TOON only on flat, row-uniform payloads whose eligibility clears the threshold. Floor: `go test ./internal/toon -run TestDecide` |
| `UNTIERED_LEAF` | a commit adds a new `internal/<leaf>/` non-test Go file with no row in the architest tier table, which reds the module and makes every peer's `fak sync push` refuse | add `"<leaf>": <tier>,` before the `// new-leaf:tier` marker in `internal/architest/architest_test.go` (or run `fak new-leaf <leaf> --tier <tier>`) and stage the tier table in the SAME commit. One-shot escape: `ALLOW_UNTIERED_LEAF=1` |
| `VOLATILE_SPAN` | TOON auto-fire declined — the span head is volatile (per-turn-rewritten), so encoding it thrashes the stable cache prefix | keep JSON — a per-turn-rewritten head must not enter the stable prefix as TOON; auto-fire only on stable spans. Floor: `go test ./internal/toon -run TestDecide` |
| `WEBHOOK_URL_NOT_ALLOWLISTED` | an A2A push-notification webhook target is not admissible (a non-allowlisted host, a non-http(s) scheme, a malformed URL, or an SSRF-range IP literal) so delivery is refused before any dial | register the receiver's host on the delivery allowlist (`Server.SetA2APushWebhookAllowlist`), use an http(s) URL with a public host, or explicitly enable loopback delivery for a deliberately-internal receiver. Floor: `internal/gateway` a2aWebhookAdmit |

Check your setup first: `python tools/extend_preflight.py`. Full contributor contract:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

**If you judge the refusal itself wrong — appeal it, don't just journal it.** A
false-positive `DENY` is byte-identical in the decision journal to a correct one, so
only the agent that made the call knows it was wrong; a private memory note fixes
nothing for the next agent. File a deduping, witnessed appeal:
`fak complain --summary "…" --reason <TOKEN> --tool <Tool> --from-journal
--args-digest <sha256:…> --live` (repeat appeals about the same class fold onto one
escalating issue). It files a gh ticket only with `--live`, **or** set
`FAK_COMPLAIN_LIVE=1` fleet-wide so every appeal auto-files — a dry-run says on
stderr that nothing was filed. Recover first; appeal only when you are confident the
guard, not your call, is wrong. Taxonomy + routing:
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
`fak index work dev-leaves` (product only, plumbing hidden), `... infra`, and
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
