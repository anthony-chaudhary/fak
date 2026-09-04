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

`main` is a shared multi-session trunk: at any moment hundreds of dirty files belong to live peers, not you. `git add <paths>` followed by a separate `git commit` is NOT atomic here — a background peer can sweep your staged file into their commit under their message, or their staged files can land inside yours. The remedy is to stage-and-commit by explicit pathspec in one locked step, then verify that only your paths and your message landed.

## The mantra (from [`CLAUDE.md`](../../../CLAUDE.md) / [`AGENTS.md`](../../../AGENTS.md))

- **Work directly on the trunk (`main`).** Never a feature branch — the trunk guard refuses `OFF_TRUNK`.
- **Commit by explicit path.** Name every path you own; never `git add -A`.
- **Sign off with `-s` (DCO).** No `Co-Authored-By` trailer.
- **Conventional-Commits subject ending in a `(fak <leaf>)` stamp** so the `dos verify` referee can bind the commit to its lane — e.g. `fix(gateway): treat same-tick ready as positive (fak gateway)`. A bare un-stamped subject stays NOT_SHIPPED.
- **Default is to ship.** Once the tree is green (`make ci`), commit AND push unprompted.

## The tools (dogfood these, not raw git)

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
| `MERGE_IN_PROGRESS` | a merge is mid-flight (`MERGE_HEAD` present) — a partial path-scoped commit can't run; finish or abort the merge, then commit by path. |
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
| `LOCK_BUSY` / `WINDOW_FULL` | another fak writer holds the lane — retryable, wait and retry. |
| `WRITER_LEASE_HELD` | a fak-managed sync-apply window holds the #4240 worktree writer lease — retryable, wait for the sync to finish and retry. |
| `HOOK_REFUSED` | a git/commit hook declined — read the hook output and fix the cause. |
| `PUSH_REJECTED` | non-fast-forward — integrate via `fak sync apply`, never force. |

`LOCK_BROKEN holder_dead …` is informational, not a refusal — a stale lock from a dead process was reclaimed and the commit proceeded.

## Exit codes

Nothing-landed is **two** outcomes, not one, and they need different responses (#5505 W4).
Exit 3 is the only code you may retry on a loop.

- **0** — success: committed, verified, (pushed if asked).
- **2** — usage error.
- **3** — CONTENTION: you never got as far as a verdict, because another writer held the lane. Nothing landed and the answer may differ next tick — **retry with backoff**. Reasons: `LOCK_BUSY`, `WINDOW_FULL`, `WRITER_LEASE_HELD`.
- **4** — REFUSED on the merits: nothing landed either, but re-running the identical command cannot change the answer — **fix the named cause or replan; never sit in a retry loop**. Reasons: `OFF_TRUNK`, `MERGE_IN_PROGRESS`, `NOTHING_STAGED`, `STALE_BASE_DELETION`, `STALE_UNTRACKED`, `SPURIOUS_STAGED_DELETION`, `CACHED_REMOVE_WORKTREE_PRESENT`, `PRESTAGED_PATH_OVERLAP`, `CORE_SELF_MODIFY`, `REVIEW_REFUTED`. Also `NOT_A_REPO`, a `--require-issue` or `--build-check` pre-lint refusal, and `fak commit preflight` / `fak sweep --apply` refusals.
- **1** — a POST-attempt failure: the commit ran but its result is bad — halt and have a human review. Reasons: `PATHSPEC_RACE`, `MESSAGE_RACE`, `SYMLINK_ESCAPE`, `HOOK_REFUSED`, `PUSH_REJECTED`.

Before the split both nothing-landed classes returned 3, so a lander that (correctly) read
3 as "retry me" spent its whole backoff budget on refusals that could never clear.

## Steps

1. **Validate your owned delta** — run `fak validate --mine <p>...` over the exact files you changed. Prove prospective build, vet, and affected tests pass in isolation. For non-Go/docs-only changes, verify links.
2. **List the exact paths YOU changed** — never a peer's. On a hot tree check mtimes/`git log -- <file>` if ownership is unclear.
3. **Lint:** `fak commit --preview -m "<subject>" --path <p> …` — fix any subject/stamp/lane issue it flags before anything lands.
4. **Commit:** `fak commit --path <p> [--path <q>] -m "<type>(<scope>): <what> (fak <leaf>)" [--push]`.
5. **Witness:** on success, `fak commit` auto-executes `dos commit-audit` (verifying diff-witnessed shape) and `dos verify` (confirming leaf registration), printing both inline. Check `git show --stat <committed_sha>` and run `dos review origin/main..HEAD` to confirm zero residual (`has_residual: false`).
6. **On a `reason` refusal,** act per the vocabulary table above — never convert a race or refusal into a force-push or an amend.

## Never

- `git add -A`, `git add .`, or `git commit -a` — they sweep peers' work into your commit.
- Force-push, amend, or `git reset --hard` on the shared trunk.
- `git pull --rebase --autostash` — it churns peers' dirty files.
- Stage a peer's uncommitted file, even to "help".
- Put `-m` after the `--` pathspec in raw git — paths-before-message trips the guard's hang detector; keep `-m "…"` first, `--` paths last.

## Read next

- [`AGENTS.md`](../../../AGENTS.md) — the full mantra plus the `fak commit` / `fak sweep` rules.
- [`CLAUDE.md`](../../../CLAUDE.md) — the three headline rules.
- `internal/safecommit` — the executor that enforces this (locking, pathspec verify, refusal reasons).
- `cmd/fak/commit.go` — the CLI front door.
- Sibling ship skill: [`release`](../release/SKILL.md) — the versioned-release counterpart with the same git-authorization posture.
