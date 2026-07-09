---
name: commit-clean
description: Commit finished work cleanly on the shared trunk — lint the subject with `fak commit --preview`, then stage-and-commit EXACTLY your paths in one locked step via `fak commit --path … -m "…"`, verify the landed path-set and message are yours, and push when asked. Mechanizes the repo's "commit clean by default" mantra (trunk-only, explicit pathspec, DCO sign-off, Conventional-Commits subject with a bindable `(fak <leaf>)` stamp). Use when the user says "commit this", "ship my work", "commit cleanly", "land my change", or when green work is finished and ready to land on the shared trunk.
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

**Lint first, always** — LINT-ONLY, touches no git:

```bash
fak commit --preview -m "<subject>" --path <p> [--path <q>]
```

Checks the subject is witness-gradeable, carries a bindable `(fak <leaf>)` stamp, and the leaf matches the paths' lane. Exit 0 clean / 1 issues / 2 usage. On a shared trunk you cannot amend, so lint the subject BEFORE the commit lands.

**Commit** — one locked stage-and-commit-and-verify step:

```bash
fak commit --path <p> [--path <q>] -m "<subject>" [--push]
```

Stages EXACTLY the named paths under an advisory lock, writes the message to a file (so an em-dash or multi-line subject can't misparse as a pathspec), commits, then VERIFIES the committed path-set == the requested path-set and the landed message == yours. If a peer raced, it refuses non-destructively — it never force-pushes. `-s` sign-off is the default; `--no-signoff` opts out. `--require-issue` makes a missing bindable `#N` blocking. `--core-lock-maintenance-witness <claim>` is the only way to clear a `CORE_SELF_MODIFY` refusal.

**Whole-lane sweep** — when the dirty tree spans a whole lane you own:

```bash
fak sweep [--json]                                        # group the dirty tree by lane
fak sweep --apply --lane <lane> -m "<subject>" [--push]   # commit one lane group by path
```

**Fallback** — ONLY when the `fak` binary is unavailable: raw `git commit -s -- <paths>` (never `git add -A`, never `-m` after `--`). Say in the handoff that you fell back.

## Refusal vocabulary

`fak commit` refuses with a reason from a closed set. Remedies:

| reason | remedy |
|---|---|
| `OFF_TRUNK` | HEAD is off-trunk or detached — get back on `main` first. |
| `NOTHING_STAGED` | the pathspec has no change — re-check which paths you actually edited. |
| `PATHSPEC_RACE` | a peer's files landed in your commit (the headline guard) — the commit is left intact for review and NOT pushed; surface it, never force-push. |
| `MESSAGE_RACE` | the landed subject/body ≠ the one you requested — surface it for review. |
| `STALE_BASE_DELETION` | your working blob predates peer lines already on origin and would silently delete them — refresh your copy of the file first. |
| `SPURIOUS_STAGED_DELETION` | a stale-index whole-path deletion with an untracked copy present — repair the index, keep the disk copy. |
| `CACHED_REMOVE_WORKTREE_PRESENT` | `git rm --cached` left the file on disk — reconcile intent before committing. |
| `PRESTAGED_PATH_OVERLAP` | a requested path already has staged hunks of unknown ownership — unstage it and keep the worktree bytes. |
| `CORE_SELF_MODIFY` | a hard-self core-lock path — needs an external maintenance witness (`--core-lock-maintenance-witness`). |
| `REVIEW_REFUTED` | the opt-in scout review refuted the diff — fix the finding before re-committing. |
| `LOCK_BUSY` / `WINDOW_FULL` | another fak writer holds the lane — retryable, wait and retry. |
| `HOOK_REFUSED` | a git/commit hook declined — read the hook output and fix the cause. |
| `PUSH_REJECTED` | non-fast-forward — integrate via `fak sync apply`, never force. |

## Exit codes

- **0** — success: committed, verified, (pushed if asked).
- **2** — usage error.
- **3** — a PRE-commit refusal: nothing landed, safe to retry or replan. Reasons: `OFF_TRUNK`, `NOTHING_STAGED`, `LOCK_BUSY`, `WINDOW_FULL`, `STALE_BASE_DELETION`, `SPURIOUS_STAGED_DELETION`, `CACHED_REMOVE_WORKTREE_PRESENT`, `PRESTAGED_PATH_OVERLAP`, `CORE_SELF_MODIFY`, `REVIEW_REFUTED`.
- **1** — a POST-attempt failure: the commit ran but its result is bad — halt and have a human review. Reasons: `PATHSPEC_RACE`, `MESSAGE_RACE`, `HOOK_REFUSED`, `PUSH_REJECTED`.

## Steps

1. **Confirm the tree is green for your lane** — `make ci`, or the scoped test for the packages you touched.
2. **List the exact paths YOU changed** — never a peer's. On a hot tree check mtimes/`git log -- <file>` if ownership is unclear.
3. **Lint:** `fak commit --preview -m "<subject>" --path <p> …` — fix any subject/stamp/lane issue it flags before anything lands.
4. **Commit:** `fak commit --path <p> [--path <q>] -m "<type>(<scope>): <what> (fak <leaf>)" [--push]`.
5. **Witness:** on success, `git show --stat HEAD` (or the JSON `verified: true` + `committed_sha`) shows EXACTLY your paths and your subject on HEAD. A clean result is `committed && verified && reason == ""`.
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
