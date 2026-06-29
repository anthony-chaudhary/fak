# AGENTS.md — orientation for coding agents

> You are an autonomous agent working in this repo. This file is the machine-read entry
> point (the [agents.md](https://agents.md) convention). It is intentionally
> command-dense and free of philosophy. For the *why*, read [`README.md`](README.md);
> for a curated doc map, read [`llms.txt`](llms.txt). Humans: see [`START-HERE.md`](START-HERE.md).

## What this project is

**fak** is an *agent kernel*: one Go binary that sits between an AI agent and the tools
it calls, and adjudicates every tool call before it runs — deny by structure, repair
malformed calls, quarantine poisoned results. It is both a **security gate** (a
default-deny capability floor the model can't talk past) and a **performance gate** (do
the shared setup work once, not every turn).

## Repo layout (where things live)

| Path | What it is |
|---|---|
| `go.mod` · `cmd/` · `internal/` | **The Go module is the repository root** (the kernel + the `fak` CLI). |
| `cmd/fak/` | The `fak` binary (every verb: `preflight`, `serve`, `agent`, `policy`, `bench`, …). |
| `internal/` | Kernel subsystems: `adjudicator`, `policy`, `vdso`, `engine`, `gateway`, `ctxmmu`, `model`, … |
| `examples/` | Policy manifests **and** runnable demos (`adjudication-demo/`, `agentdojo-redteam/`, `mcp/`). |
| `docs/` | Explainers, integration guides (`docs/integrations/`), benchmark methodology, proofs. |
| `docs/private-comms-channel.md` | **The private comms channel** (Slack control-bridge to the lab GPU servers) — a public stub pointing to its home in the `fak-private` companion repo. Start here to reach the hardware. |

## Build / test / run

> **The Go module is the repository root** — run `go` commands from the clone root.
> Needs Go 1.26+ (`GOTOOLCHAIN=auto` self-fetches). Zero external deps, so no `go.sum`.

```bash
go build ./cmd/fak        # -> ./fak  (fak.exe on Windows).
make test-fast            # ~2s smoke gate: build + vet + `go test -short ./...`
make test                 # full suite incl. the weight-backed model witnesses
make ci                   # the full gate: build + vet + test + claims-lint  (Windows: scripts/ci.ps1)
```

Or install the released binary directly — the module is at the repo root, so this resolves:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Windows:** `go build` / `go vet` / `go run` work natively, but native `go test` is
> blocked by an OS Application-Control policy on the freshly-compiled test binaries — run
> the suite under WSL: `./test.ps1` from the repo root. This is an OS quirk, not a code failure.
> For now, **avoid running tests or long-lived local `fak serve` directly on the dev box** —
> use WSL/CI for tests and the GPU/cloud nodes for a real serve. See
> [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md).

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

## Hard rules (these WILL bite an agent — they are enforced below the agent layer)

**Default: ship.** Once the tree is green, **commit AND push** — don't wait to be asked.
Green = `make ci` (build + vet + test + claims-lint; on a native-Windows host run the test
suite under WSL with `./test.ps1`, since native `go test` is blocked). The commit-message,
file-admission, public-leak, and trunk guards then run automatically as git hooks at
commit/push — so "the guard passed" means CI is green *and* the commit/push was accepted.
(Those gates run in one process via `fak hooks pre-commit` / `commit-msg` when a `fak`
binary is on PATH — ~10.7s → ~0.3s vs spawning a Python interpreter per gate; the shell
hooks fall back to the `tools/check_*.py` checkers when no binary resolves, so a fresh
clone is still gated.)
The HOW below is unchanged and gates the WHEN: stay on the trunk, `git commit -s -- <paths>`
(never `git add -A`), merge **in place** if the trunk diverged, wait out a peer's
`MERGE_HEAD`, and **never force-push**. If a guard refuses (`OFF_TRUNK`), a peer merge is
mid-flight, or a blocker stands — reconcile in place or STOP; the default does not fire
until it clears.

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree.**
  The trunk guard *refuses* off-trunk commits (the `OFF_TRUNK` law). A dirty/diverged
  tree means reconcile **in place** or STOP — never escape into a side branch.
  - *Stray worktrees still accrue* — the harness and subagents spin scratch worktrees the
    rule can't prevent. `tools/worktree_doctor.py` is the janitor: it auto-detects the
    trunk, safely prunes loss-free strays, and `--sweep-disposable` archives-then-reaps
    dead scratch worktrees (temp / scratchpad / pr-work) while sparing live sessions via a
    freshness guard. A scheduled task runs it (`tools/register_worktree_doctor.ps1`).
  - *Diverged trunk (`git status` says "have diverged"):* `git fetch origin main`, then
    `git merge origin/main` **in place** and resolve. This is a shared trunk — peers
    routinely build the SAME feature under a different SHA, so most conflicts resolve to
    the trunk **superset** and the merged tree often equals HEAD (verify:
    `git diff --cached` is empty). Finish with a plain `git commit -s` — the merge commits
    the index as-is; never `-a` / `git add -A`, which would sweep a peer's files into your
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
- **Commit by explicit path** — `git commit -- <paths>`, never `git add -A`. This is a
  shared multi-session tree; never stage a peer's uncommitted files. `fak commit --path
  <p> -m "<msg>"` mechanizes this whole rule: it stages only the named paths under an
  advisory lock, writes the message to a file (so an em-dash/multiline subject can't
  misparse as a pathspec), runs the real hooks, then **asserts the committed file set
  equals the requested paths** — refusing `PATHSPEC_RACE` (and leaving the commit intact,
  never force-pushing) if a peer swept extra files in. It also refuses `OFF_TRUNK` /
  `MERGE_IN_PROGRESS` up front, so the runbook above is a verb, not a discipline you have
  to remember.
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
- **Add a feature as a leaf, not a core edit.** `python tools/new_leaf.py <name> --tier
  <tier> [--register]` stamps a conforming skeleton; the frozen ABI (`fak/internal/abi`)
  is additive-only and human-owned. `internal/architest` fails the build on a bad import.
- **New tooling is Go, not Python.** The repo is a Go project; the ~460 `tools/*.py` scripts
  are a *grandfathered* baseline, frozen - not a pattern to copy. A new tool ships as a `fak`
  subcommand (pure logic in `internal/<name>/`, a thin shell in `cmd/fak/<name>.go` - see
  `cmd/fak/velocity.go`) or a `cmd/<name>/` binary, never a new `tools/*.py`. The
  `internal/pythongate` ratchet (`go test ./internal/pythongate -run TestNoNewPythonTools`)
  reds the trunk on any `tools/*.py` outside the baseline, and porting a grandfathered script
  to Go shrinks that baseline - the ratchet only ever tightens. When you *touch* a `tools/*.py`
  for non-trivial work, default to porting it to Go in the same pass (`REASON_NEW_PYTHON_TOOL`).
- **GPU-server/Slack control is private; public evidence is scrubbed.** Benchmark results and
  runbooks can live here once scrubbed to generic GPU-server language, but live Slack
  control code belongs in `fak-private`: `cmd|internal/*dgx*/`, Slack bridge/control
  packages, `cmd/slackgc/`, and the sunset `tools/bench_slack.py` path. See
  [`docs/dgx-slack-boundary.md`](docs/dgx-slack-boundary.md). **To actually reach the
  channel** (the Slack control-bridge to the lab GPU servers), start at the public stub
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
- **Writes that resolve *outside* the repo are refused (`OUT_OF_TREE_WRITE`).** The
  `repo-guard` PreToolUse hook (`tools/repo_guard.py`, on by default on a fleet host) denies
  a Bash/Write/Edit op whose target escapes the workspace — a `../sibling` path or an absolute
  `/c/.../work/other-repo`. `work/` holds many sibling repos, so a one-level escape lands in
  *another* project. Write scratch to a temp dir or an in-repo path, not `..`. Allowed
  out-of-tree: the null/std-stream sinks (`> /dev/null`, `> /dev/stderr`) and the paired
  `fak-private` companion repo. This is the **write-time** half of the public/private split;
  the **commit-time** half — `FILE_ADMISSION` (`check_committed_files.py`) and `PUBLIC_LEAK`
  (`scrub_public_copy.py`) — keeps private *content* out of the public history. Soften with
  `FAK_REPO_GUARD=warn` (advisory) or `off`. Full doc: [`docs/repo-guard.md`](docs/repo-guard.md).

### If the kernel refuses you (recover, don't fight it)

A guard refusal names a token from a **closed vocabulary** — declared as `[reasons.*]`
blocks in [`dos.toml`](dos.toml), each with a `summary` + a `fix` you can look up live
with `dos check-reason <TOKEN>`. When you hit one, recover by the action below; don't
route around the guard (that just trips the next one).

| Token | What tripped it | Recover by |
|---|---|---|
| `OFF_TRUNK` | you branched / spun a worktree instead of committing to `main` | commit directly to `main` with `git commit -s -- <paths>`; a dirty/diverged tree means merge **in place** or STOP — never escape into a side branch |
| `LOOP_DONE_UNWITNESSED` | a loop turn claimed done, but its configured external witness did not corroborate the done effect | re-arm the loop with this token as feedback, satisfy the witness criterion (`dos commit-audit`, `dos verify`, `dos test-witness`, or a registered witness), then re-check |
| `STALE_BASE_DELETION` | a pathspec commit would silently drop a peer-added block because the working-tree copy is stale relative to `origin/<trunk>` | fetch and merge/rebase `origin/<trunk>` in place so the working tree includes the peer block, then re-commit by explicit path |
| `ARCH_LAYER_VIOLATION` | an upward/cross-tier import, or a new leaf with no declared tier | invert the dependency through a registration seam, or push the shared type down a layer; declare a new leaf's tier (`python tools/new_leaf.py`). Floor: `internal/architest` |
| `OUT_OF_DIRECTION` | request-path logic in an untyped language, or a non-Go package blank-imported into the kernel | keep the request path Go-only; a non-Go seam stays off-path behind a typed, re-validated boundary. Floor: architest `TestHotPathHasNoExec` |
| `FILE_ADMISSION` | a staged path is private-only content, regenerable junk, or an oversized blob | move private-only code to `fak-private`; drop or gitignore junk; put real data under `experiments/` or `testdata/` |
| `PUBLIC_LEAK` | staged content matches a redact-needle | remove or redact the needle before committing; `FLEET_ALLOW_LEAK=1` overrides once, only for an intentional adversarial fixture |
| `OUT_OF_TREE_WRITE` | a write op escaped the repo into a sibling tree | operate inside the workspace; send scratch to a temp dir, never `..`. Soften with `FAK_REPO_GUARD=warn`. See [`docs/repo-guard.md`](docs/repo-guard.md) |
| `STALE_RECALL` | a loop is about to act on recalled status/plan memory whose witness is stale relative to git or the loop ledger | refresh from the source witness (`dos status`, `dos verify`, `dos commit-audit`, or current git ref), discard the stale recall, then retry |
| `COLLISION_RISK` | a dispatch or worker launch would overlap a live lease/region and risks two agents mutating the same file tree | wait for the lease, repartition, or pick a disjoint lane/region after checking `dos arbitrate` / `dos top` |
| `GATE_LATENCY_REGRESSION` | a guard or DOS gate grew slow enough to threaten the loop's dev-ex latency budget | measure the gate with the relevant hook/status benchmark, fix or budget the added latency, then rerun before claiming the loop is healthy |
| `RUN_STATUS_CLAIMED_FIELD` | `dos status` returned a run digest containing a `claimed` field, violating the witnessed-status contract | fix the status producer so peers read only liveness, ledger-verified progress, lease region, and resume evidence; do not consume claimed status |
| `L3_CROSS_TENANT_SCOPE_DENIED` | a cross-tenant reader asked for an L3 cache page whose `ShareScope` does not reach across the trust boundary — a page private to one agent (`ScopeAgent`) or bound to the owner's tenant (`ScopeTenant`), fetched by a *different* tenant (a capability check, not a namespace-prefix match) | the page is not shareable across tenants — only an owner-marked fleet/public (`ScopeFleet`) page crosses. Mark the prefix `ScopeFleet` at the producer if it is genuinely public, or serve the reader its own tenant-scoped page. Floor: `go test ./internal/gateway -run TestL3Share_CrossTenant` |
| `L3_PAGE_DIGEST_MISMATCH` | the bytes an L3 `get` returned do not hash to the digest the page claims (`Ref.Digest`) — a corrupt or mis-tagged page the semantics-free store cannot detect, refused for *every* reader (same-tenant included) | do not admit the page — the content address does not match its bytes. Re-fetch / re-key at the store, or re-mint the page so its `Ref.Digest` is the true `hex(sha256)` of its content. Floor: `go test ./internal/gateway -run TestL3Share_DigestMismatch` |

Check your setup first: `python tools/extend_preflight.py`. Full contributor contract:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Where to go next

| If you want to… | Read |
|---|---|
| Every CLI verb + what's shipped | [`docs/cli-reference.md`](docs/cli-reference.md) |
| Learn every concept in prerequisite order (a course, join at your level) | [`LEARNING-PATH.md`](LEARNING-PATH.md) |
| Install / run tiers (offline → gateway → in-kernel model) | [`fak/GETTING-STARTED.md`](GETTING-STARTED.md) |
| Put fak in front of *your* agent (Claude Code / Cursor / MCP) | [`docs/integrations/`](docs/integrations/) · [`fak/examples/mcp/`](examples/mcp/) |
| The deployable capability floor (policy manifests) | [`fak/POLICY.md`](POLICY.md) · [`fak/examples/README.md`](examples/README.md) |
| Extend the kernel (plug in → prove correct → prove faster) | [`fak/EXTENDING.md`](EXTENDING.md) · [`fak/ARCHITECTURE.md`](ARCHITECTURE.md) |
| What's real vs simulated vs stub | [`fak/CLAIMS.md`](CLAIMS.md) · [`fak/STATUS.md`](STATUS.md) |
| Every benchmark number (single source of truth) | [`fak/BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) |
| Roll back to a stable version (revert / downgrade / pin) | [`docs/ROLLBACK.md`](docs/ROLLBACK.md) |
| A curated map of all the docs | [`llms.txt`](llms.txt) |

License: [Apache-2.0](LICENSE).
