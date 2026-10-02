---
name: commit-clean
description: Commit finished work cleanly on the shared trunk — lint the subject with `fak commit --preview`, then stage-and-commit EXACTLY your paths in one locked step via `fak commit --path … -m "…"`, verify the landed path-set and message are yours, and push when asked. Mechanizes the repo's "commit clean by default" mantra (trunk-only, explicit pathspec, DCO sign-off, Conventional-Commits subject with a bindable `(fak <leaf>)` stamp). Use when the user says "commit this", "ship my work", "commit cleanly", "land my change".
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "[subject] [--path <p>...] [--push]"
metadata:
  opencode: claude-only   # #422: the read-only allowed-tools boundary (commit via `fak commit`, never hand-edit) is load-bearing and Claude-only — opencode drops it
---

# /commit-clean — Commit clean by default on the shared trunk

One repeatable pass that lands YOUR finished paths on `main` with a lintable, bindable subject — and refuses cleanly when a peer races you.

**Git authorization.** Invocation of this skill is the user's explicit authorization to run `fak commit` and `fak sweep` (which shell to git underneath) and to pass `--push` when the user asked to push or the tree is green and the ship-by-default rule applies. The "never commit/push unless asked" default does NOT apply here — committing IS the skill's job. Destructive operations the steps don't list (force-push, `--amend`, `git reset --hard`, rebase) still require explicit confirmation — and the tooling refuses them anyway.

## Why this is hard

`main` is a shared multi-session trunk: at any moment hundreds of dirty files belong to live peers, not you. `git add <paths>` followed by a separate `git commit` is NOT atomic here — a background peer can sweep your staged file into their commit under their message, or their staged files can land inside yours. Furthermore, concurrent merges and upstream advances create divergence races. The remedy is to stage-and-commit by explicit pathspec in one locked step, wait out active peer merges (`MERGE_HEAD`), integrate upstream safely via `fak sync apply` / `fak sync reconcile` without raw merges or `--autostash`, and verify that only your paths and your message landed.

## The mantra (from [`CLAUDE.md`](../../../CLAUDE.md) / [`AGENTS.md`](../../../AGENTS.md))

- **Work directly on the trunk (`main`).** Never a feature branch — the trunk guard refuses `OFF_TRUNK`.
- **Safe merge & non-destructive convergence.** Never run unverified raw git merges, force-push, or use `git pull --rebase --autostash` (which churns peers' dirty files). Check for and wait out any active peer `MERGE_HEAD` (`MERGE_IN_PROGRESS`). Integrate upstream advances non-destructively via `fak sync apply` (`--ff-only`) or `fak sync reconcile` (`ROUTE_DISJOINT_INTEGRATE`, `ROUTE_SUPERSET_MERGE`, or `ROUTE_HOLD_DIRTY_COLLISION` with `fak wip park`).
- **Dual-repo awareness (`fak` & `fak-private`).** When committing shared interfaces (`pkg/*`) or companion modules affecting `fak-private`, synchronize both repositories: verify multi-module alignment with `go work sync`, verify zero private leak needles (`python tools/scrub_public_copy.py --audit-staged`), and coordinate commits across both repos.
- **Commit by explicit path.** Name every path you own; never `git add -A`.
- **Sign off with `-s` (DCO).** No `Co-Authored-By` trailer.
- **Conventional-Commits subject ending in a `(fak <leaf>)` stamp** so the `dos verify` referee can bind the commit to its lane — e.g. `fix(gateway): treat same-tick ready as positive (fak gateway)`. A bare un-stamped subject stays NOT_SHIPPED.
- **Default is to ship.** Once the tree is green (`make ci`), commit AND push unprompted via `fak sync push`.
- **Autonomous landing upon task completion.** Following the safe commit-and-land process to completion is MANDATORY and active BY DEFAULT when any task finishes (using a new subagent if needed). Never leave touched paths uncommitted or require an operator prompt to trigger landing.
- **Strict Ban on Staging PowerShell & Loose Scripts.** Never stage or commit new `.ps1` (PowerShell), `.sh`, `.bat`, or loose scripts. All new automation and tooling MUST be native Go programs (`cmd/*`, Go leaves registered as `fak` CLI verbs, or quarantined sub-modules) providing modular, integrated, long-term value. Unless there is an extraordinary, super heavily justified reason (~1 in entire repo), new scripts are rejected.

## The tools (dogfood these, not raw git)

### Check access in the actual landing executor

Before the first GitHub-dependent step or reporting an access blocker, check the
executor and shell that will perform that step. A desktop terminal, noninteractive
shell, and CI runner can share a host but have different `HOME`, `PATH`, and
authentication state. A browser login or a successful read in another surface does
not establish access for this `gh` invocation or for Git transport.

Use `fak-dev orient env --paths <p> --json` to choose the supported Git shell and
inspect host routing and leases. Its `CLEAR` verdict is not a GitHub authentication
or landing-readiness witness. In the intended executor, resolve `git`, `gh`, `fak`,
and `fak-dev` with that shell's command-discovery mechanism, then run these read-only
checks in the selected shell:

```text
git rev-parse --show-toplevel
git rev-parse HEAD
gh auth status --hostname github.com
gh api --hostname github.com user --jq .login
gh repo view https://github.com/anthony-chaudhary/fak --json nameWithOwner,viewerPermission
```

Check that the resolved checkout is the intended workspace. Keep the GitHub host
and target repository consistent when adapting these commands; an inherited
`GH_HOST` must not silently change which service a check observes. Record the
executor label, timestamp, checkout/base, resolved tool paths, account login,
repository, and each command's actual outcome in a **local** receipt. Keep private
paths, credential/config contents, and raw auth-status output out of public
summaries; never use `gh auth status --show-token` or move credentials between
execution contexts.

A failed read describes the checked context. Before declaring GitHub unavailable
or requesting reconnection, inspect the already-authorized intended executor;
otherwise report that precise context and failed command as the blocker. Continue
independent source verification where possible. If checkout/root resolution fails
for a verified source-only archive, checkout-dependent landing remains unavailable
there; do not invent Git history to turn source checks into committed-tip evidence.

Successful API reads and reported permissions do not prove a Git push succeeded,
that required native tools are ready, or that a write is authorized. All applicable
ownership, sync, provenance, validation, hook, and witness gates still apply. Repeat
the access check when the executor, account, host, or target changes; retain other
evidence only within its original source and execution-environment scope.

### Guarded validation and landing

**Validate in isolation first, always** — compiles, vets, and tests prospective tree:

```bash
fak validate --mine <p> [--mine <q>]
```

Isolates your owned delta against HEAD in a private checkout, runs gofmt, build, vet, and affected package tests under WSL while masking peer WIP. Do not commit if validation fails or times out.

**Lint first, always** — LINT-ONLY, touches no git:

```bash
fak commit --preview -m "<subject>" --path <p> [--path <q>]
```

Checks the subject is witness-gradeable, carries a bindable `(fak <leaf>)` stamp, and the leaf matches the paths' lane. Exit 0 clean / 1 issues / 2 usage. On a shared trunk you cannot amend, so lint the subject BEFORE the commit lands.

**Commit** — one locked stage-and-commit-and-verify step:

```bash
fak commit --path <p> [--path <q>] -m "<subject>" [--push]
```

Stages EXACTLY the named paths under an advisory lock, writes the message to a file (so an em-dash or multi-line subject can't misparse as a pathspec), commits, then VERIFIES the committed path-set == the requested path-set and the landed message == yours. If a peer raced, it refuses non-destructively — it never force-pushes. `-s` sign-off is the default; `--no-signoff` opts out. `--require-issue` makes a missing bindable `#N` blocking. `--core-lock-maintenance-witness <claim>` is the only way to clear a `CORE_SELF_MODIFY` refusal. `--json` emits the structured result (`committed`, `verified`, `committed_sha`, `reason`) instead of the default prose line — use it when a step needs to check those fields.

**Whole-lane sweep** — when the dirty tree spans a whole lane you own:

```bash
fak sweep [--json]                                        # group the dirty tree by lane
fak sweep --apply --lane <lane> -m "<subject>" [--push]   # commit one lane group by path
```

**Fallback** — ONLY when the `fak` binary is unavailable: raw `git commit -s -m "<subject> (fak <leaf>)" -- <paths>` — keep `-m`/`-F` BEFORE the `--` pathspec. A bare `git commit` with no message source opens the editor and hangs headless (the guard's `INTERACTIVE_HANG`); an `-m` placed AFTER `--` is parsed as a pathspec, not a message. Never `git add -A`. Say in the handoff that you fell back.

## Refusal vocabulary

`fak commit` refuses with a reason from a closed set. Remedies:

| reason | remedy |
|---|---|
| `OFF_TRUNK` | HEAD is off-trunk or detached — get back on `main` first. |
| `NOTHING_STAGED` | the pathspec has no change — re-check which paths you actually edited. |
| `MERGE_IN_PROGRESS` | a merge is mid-flight (`MERGE_HEAD` present) — a partial path-scoped commit can't run. Never force or run raw merges. If owned by a peer, unstage your paths and wait; do not abort or finish peer merges. If owned by your run, converge cleanly via `fak sync apply` (`--ff-only`) or `fak sync reconcile` (`ROUTE_DISJOINT_INTEGRATE` / `ROUTE_SUPERSET_MERGE`), parking conflicting WIP with `fak wip park` if needed (`ROUTE_HOLD_DIRTY_COLLISION`). |
| `PATHSPEC_RACE` | a peer's files landed in your commit (the headline guard) — the commit is left intact for review and NOT pushed; surface it, never force-push. |
| `MESSAGE_RACE` | the landed subject/body ≠ the one you requested — surface it for review. |
| `SYMLINK_ESCAPE` | a landed path resolves through a symlink to a target outside your lease (the CVE-2025-53109 class) — the commit is left intact for review and NOT pushed; surface it, never force-push. |
| `STALE_BASE_DELETION` | your working blob predates peer lines already on origin and would silently delete them — refresh your copy of the file first. |
| `STALE_UNTRACKED` | the path is `??` here but ALREADY exists on `origin/<trunk>`: your HEAD is behind, so this is not new work. Fetch + merge, then re-check — `git diff origin/main -- <p>` is misleading for an untracked path (it shows trunk's whole file as deleted); compare with `git show origin/main:<p>`. |
| `SPURIOUS_STAGED_DELETION` | a stale-index whole-path deletion with an untracked copy present — repair the index, keep the disk copy. |
| `CACHED_REMOVE_WORKTREE_PRESENT` | `git rm --cached` left the file on disk — reconcile intent before committing. |
| `PRESTAGED_PATH_OVERLAP` | a requested path already has staged hunks of unknown ownership — unstage it and keep the worktree bytes. |
| `CORE_SELF_MODIFY` | a hard-self core-lock path — needs an external maintenance witness (`--core-lock-maintenance-witness`). |
| `REVIEW_REFUTED` | the opt-in scout review refuted the diff — fix the finding before re-committing. |
| `COMMITTED_RED` | the prospective tree fails build, vet, gofmt, strix-validation, or affected tests (exit 4). Run the printed `fak validate --mine ...` and fix the named phase. A build red already present at HEAD is admitted automatically with an advisory. The one exception is a pre-existing red in the **test step only**: see **Pre-existing red** below, which may land with `--no-build-check` without asking. |
| `LOCK_BUSY` / `WINDOW_FULL` | another fak writer holds the lane — retryable, wait and retry. |
| `WRITER_LEASE_HELD` | a fak-managed sync-apply window holds the #4240 worktree writer lease — retryable, wait for the sync to finish and retry. |
| `HOOK_REFUSED` | a git/commit hook declined — read the hook output and fix the cause. |
| `PUSH_REJECTED` | remote rejected push (non-fast-forward conflict on origin). Never force-push or use `git pull --rebase --autostash`. Integrate upstream changes via `fak sync apply` (or `fak sync reconcile --apply`), re-validate your owned paths with `fak validate --mine`, and push with `fak sync push`. When committing cross-repo changes affecting `fak-private`, verify boundary leak hygiene (`python tools/scrub_public_copy.py --audit-staged`) and coordinate both pushes. |

`LOCK_BROKEN holder_dead …` is informational, not a refusal — a stale lock from a dead process was reclaimed and the commit proceeded.

## Pre-existing red (operator policy 2026-09-26)

A gate that is red **only** because tests already fail on the landing base does not block your change. Land it with `fak commit --no-build-check`, or with `fak worktree worker land --verify off` for a managed worker. **Do not ask** when every condition below holds and each has a witness. Never use `git commit --no-verify`, `ALLOW_COMMITTED_RED`, `FLEET_BUILDCHECK_GUARD=off`, `FAK_COMMIT_BUILD_CHECK=off|allow-timeout`, or `--allow-build-check-timeout` for this.

1. **Test step only.** A `COMMITTED_RED` is eligible only when its detail names the `test` step and nothing else. `build`, `vet`, `gofmt`, `test-audit-selection`, `wsl_preflight`, and `strix-validation` are never eligible. The gate already admits a build red that exists at HEAD, so a build-phase refusal is a red your change introduced. Hardware evidence is never skipped. A `--verify go-build` land refusal is eligible only when the same packages fail with the same first error on the landing base. `PATHSPEC_RACE`, `HOOK_REFUSED`, `CORE_SELF_MODIFY`, DCO, and the leak/secret audit are never eligible. A `BUILD_CHECK_TIMEOUT` becomes eligible only after the condition-3 `-skip` rerun shows the timeout is the base's witnessed hang.
2. **Base witnessed red twice, in the same environment.** The base is the exact commit the change lands on (the trunk tip after `fak sync apply`), not the fork point. In a detached worktree at that SHA, run the gate's affected packages with `go test -count=1 -json`, **twice**, in the gate's environment. If the gate's runner was WSL, run inside WSL from a Linux-filesystem copy, never from the Windows host or `/mnt/c`. Record `go version` and `go env GOOS GOARCH CGO_ENABLED GOFLAGS GOWORK`; both must match the change run. Every Failing-Set test must fail in both runs. A test that passes in either run is flaky, not pre-existing, and must pass with the change. If trunk moves or you edit after the witness, witness again.
3. **No new red, and the rest of the package is green.** Compare per package using the `-json`/`-v` output. An entry matches only when the test name **and** the first failure line match (with paths, line numbers, and addresses normalized). A test with the same name that fails differently is new red. `[build failed]`, `[setup failed]`, `panic:`, `test timed out`, and a package-level FAIL count as entries too. A base-red test that is missing from the change's failing set must show `PASS` in the change run. Then rerun every red package on the change with `go test -count=1 -skip '^(TestA|TestB|...)$'`, using the exact Failing-Set names. Each package must report `ok`, with no FAIL, panic, timeout, or `[build failed]`. If a hang or panic cannot be skipped by name, the change is not eligible.
4. **Own tests pass, not skip.** After the last edit, in the gate environment, run `go test -v -count=1 -run '<own tests>' <pkg>`. Every added or modified test must report `--- PASS`, not `--- SKIP` and not `no tests to run`.
5. **Disjoint from the red.** The diff edits no Failing-Set test file and no non-test file on a failing test's call path. Sharing a package is allowed only when the files are disjoint **and** that package is `ok` in the condition-3 rerun. If you cannot show the diff is disjoint, the change is not eligible.
6. **Every other check still passes.** `--no-build-check` also drops gofmt, importer build/vet, and test compilation. Run each of these yourself in the gate environment; each must exit 0:
   - `gofmt -l <changed .go files>` prints nothing.
   - `go build` and `go vet` on the changed packages, their in-module importers, and `./cmd/fak`. A failure is allowed only in a package that fails identically on the base and that the diff does not touch.
   - `go test -count=1 -run '^$' <affected pkgs>`.
   - The leak audit (`fak-dev audit-leak --staged`) is clean.

**Receipts (mandatory):**

- The commit body, written to a file and passed with `fak commit -F <file>`:

  ```text
  Gate-Skip: pre-existing-red (operator policy 2026-09-26)
  Parent: <sha>   (must equal the landed commit's first parent)
  Witness: <exact base command> x2 (env: <runner>; <go version>; <go env line>)
  Failing-Set: <pkg.TestA>, <pkg.TestB>, ... (name + first failure match; change set is a subset)
  Remainder: <go test -count=1 -skip ... pkgs> -> ok
  Own-Tests: <go test -v -count=1 -run ... pkg> -> PASS
  Disjoint: diff=<files>; red=<files>
  Logs: <issue comment holding the raw base and change outputs> sha256:<base>,<change>
  Tracking: #<issue>
  ```

- An issue tracks the trunk red and holds the raw logs. Cite an existing one, or open one. Then triage the red through `/ci-repair`.

**Never** use this exception for a red your change caused, for a new failure, hang, or failure signature, for own tests that fail or skip, or to skip a non-test gate. If any condition is missing, fix the red before landing.

## Exit codes

Nothing-landed is **two** outcomes, not one, and they need different responses (#5505 W4).
Exit 3 is the only code you may retry on a loop.

- **0** — success: committed, verified, (pushed if asked).
- **2** — usage error.
- **3** — CONTENTION: you never got as far as a verdict, because another writer held the lane. Nothing landed and the answer may differ next tick — **retry with backoff**. Reasons: `LOCK_BUSY`, `WINDOW_FULL`, `WRITER_LEASE_HELD`.
- **4** — REFUSED on the merits: nothing landed either, but re-running the identical command cannot change the answer — **fix the named cause or replan; never sit in a retry loop**. Reasons: `OFF_TRUNK`, `MERGE_IN_PROGRESS`, `NOTHING_STAGED`, `STALE_BASE_DELETION`, `STALE_UNTRACKED`, `SPURIOUS_STAGED_DELETION`, `CACHED_REMOVE_WORKTREE_PRESENT`, `PRESTAGED_PATH_OVERLAP`, `CORE_SELF_MODIFY`, `REVIEW_REFUTED`, `COMMITTED_RED` (see **Pre-existing red** above). Also `NOT_A_REPO`, a `--require-issue` or `--build-check` pre-lint refusal, and `fak commit preflight` / `fak sweep --apply` refusals.
- **1** — a POST-attempt failure: the commit ran but its result is bad — halt and have a human review. Reasons: `PATHSPEC_RACE`, `MESSAGE_RACE`, `SYMLINK_ESCAPE`, `HOOK_REFUSED`, `PUSH_REJECTED`.

Before the split both nothing-landed classes returned 3, so a lander that (correctly) read
3 as "retry me" spent its whole backoff budget on refusals that could never clear.

## Steps

0. **Bind access to the intended executor** — before a GitHub-dependent step or access-blocker claim, follow the read-only context check above; a result from another shell is not this executor's receipt.
1. **Validate your owned delta** — run `fak validate --mine <p>...` over the exact files you changed. Prove prospective build, vet, and affected tests pass in isolation. If only the test step fails, and it fails identically on the landing base, follow **Pre-existing red**. Tests your change adds or modifies must still pass. For non-Go/docs-only changes, verify links.
2. **List the exact paths YOU changed** — never a peer's. On a hot tree check mtimes/`git log -- <file>` if ownership is unclear.
3. **Lint:** `fak commit --preview -m "<subject>" --path <p> …` — fix any subject/stamp/lane issue it flags before anything lands.
4. **Commit:** `fak commit --path <p> [--path <q>] -m "<type>(<scope>): <what> (fak <leaf>)" [--push]`.
5. **Witness:** on success, `fak commit` auto-executes `dos commit-audit` (verifying diff-witnessed shape) and `dos verify` (confirming leaf registration), printing both inline. Check `git show --stat <committed_sha>` and run `dos review origin/main..HEAD` to confirm zero residual (`has_residual: false`).
6. **On a `reason` refusal,** act per the vocabulary table above — never convert a race or refusal into a force-push or an amend.

## Never

- `git add -A`, `git add .`, or `git commit -a` — they sweep peers' work into your commit.
- Force-push, amend, or `git reset --hard` on the shared trunk.
- `--no-build-check` / `--verify off` for a red your change caused, a new failure or hang, a non-test step (build, vet, gofmt, strix-validation), or any non-test gate. Never reach for `--no-verify` or the `ALLOW_COMMITTED_RED` / `FAK_COMMIT_BUILD_CHECK` env escapes instead.
- `git pull --rebase --autostash` — it churns peers' dirty files.
- Unverified raw git merges on the shared trunk (converge via `fak sync apply` / `fak sync reconcile`).
- Stage a peer's uncommitted file, even to "help".
- Put `-m` after the `--` pathspec in raw git — paths-before-message trips the guard's hang detector; keep `-m "…"` first, `--` paths last.

## Read next

- [`AGENTS.md`](../../../AGENTS.md) — the full mantra plus the `fak commit` / `fak sweep` rules.
- [`CLAUDE.md`](../../../CLAUDE.md) — the three headline rules.
- `internal/safecommit` — the executor that enforces this (locking, pathspec verify, refusal reasons).
- `cmd/fak/commit.go` — the CLI front door.
- Sibling ship skill: [`release`](../release/SKILL.md) — the versioned-release counterpart with the same git-authorization posture.
