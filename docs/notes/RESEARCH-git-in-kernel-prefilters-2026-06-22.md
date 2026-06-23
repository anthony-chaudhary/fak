---
title: "Git operations in the kernel prefilter — what's possible, what isn't"
description: "A research note + shipped first slice: which git hazards a stateless kernel prefilter can refuse at the call boundary (force-push, --amend, add -A, --no-verify, tag -f, rebase -i), which ones fundamentally cannot (OFF_TRUNK, sweep-a-peer, MERGE_HEAD — they need repo state), and the gitgate adjudicator rung that lifts the decidable set off the git-hook layer into the kernel."
---

# Git operations in the kernel prefilter — what's possible, what isn't

Date: 2026-06-22

Scope: the design behind `internal/gitgate` (the shipped first slice) and the
research that bounds it. The kernel already adjudicates every tool call before it
runs; the question this note answers is *which git operations belong in that
prefilter, and which cannot live there no matter how much we want them to.*

Companion code: [`internal/gitgate/gitgate.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/gitgate/gitgate.go).
Companion floor (the after-the-fact layer this complements):
[`tools/githooks/`](https://github.com/anthony-chaudhary/fak/tree/main/tools/githooks).

---

## 1. The question

A coding agent's most dangerous moves in this repo are git moves: force-pushing
the shared trunk, amending a peer's HEAD, `git add -A` sweeping another session's
files, skipping the commit guards, spinning up an off-trunk branch. The repo's
laws against these (AGENTS.md / CLAUDE.md) are enforced two ways today, and both
fire *after* the agent has already committed to the action:

- **git hooks** (`tools/githooks/*`) reject at the git transaction boundary — the
  `git` process starts, a hook says no, the agent re-plans.
- **the referee** (`dos verify`, `commit_stamp_doctor`) grades commits *after they
  land*.

`fak` is an agent kernel: it adjudicates every tool call *before it runs*. So the
question is whether these git laws can move one layer earlier — from "the doomed
`git push --force` runs and a hook complains" to "the tool call is denied at the
boundary with a machine-readable reason the agent loop consumes." The honest
answer splits cleanly in two.

## 2. The two layers today (inventory)

What was already in the prefilter before this work, and what was not:

| git operation | enforced where (before) | note |
|---|---|---|
| `git_push` / `git_merge` / `git_tag` (named **tools**) | prefilter | `DevAgentPolicy.Deny` blocks them *by tool name* — does **not** catch a `Bash` call with `command="git push"`. |
| `git apply/checkout/restore/stash`, `git -C <dir> <mutate>` | prefilter | substring-matched in `commandWrites`, but **only** to protect a guarded tree from SELF_MODIFY — never a general force-push/amend guard. |
| write into `.git/` | prefilter | `.git/` is a self-modify glob; protects plumbing, not branch/commit semantics. |
| ship / release (high-level) | prefilter → witness | `shipgate` returns RequireWitness; the witness resolver corroborates against git evidence off the fast path. |
| **OFF_TRUNK** (off-main branch) | git-hook | `reference-transaction` refuses branch creation. |
| **`git push --force` / `--amend` / `add -A` / `--no-verify` / `rebase -i` via Bash** | **none at the kernel** | the repo's hardest laws were *doc-only* at the call boundary. |
| commit-message shape, `(fak <leaf>)` stamp | git-hook / referee | `check_commit_msg`, `dos verify`. |
| PUBLIC_LEAK / FILE_ADMISSION / DOC_PLACEMENT | git-hook + CI | pre-commit gate sequence. |

The gap is the bolded row: the laws an agent is *most* likely to trip, invoked the
way an agent actually invokes them (a shell tool carrying `command="git ..."`),
reached the kernel and fell straight through.

## 3. What a prefilter CAN decide (shipped)

A hazard is liftable into a stateless prefilter exactly when it is **decidable
from the call's argv alone** — no repo state, no second source of truth. These all
qualify, and `internal/gitgate` refuses them:

| hazard | rule | law |
|---|---|---|
| force-push | `push` + `--force` / `-f` / `--force-with-lease` | never force-push the shared trunk |
| skip-hooks | `push`/`commit` + `--no-verify`; `git -c core.hooksPath=…` | never bypass the guards |
| skip-signing | `commit` + `--no-gpg-sign` | keep commit signing on |
| amend | `commit` + `--amend` | never amend in a shared tree (HEAD moves between peers) |
| bulk staging | `add` + `-A`/`--all`/`-u` or bare `.`; `commit` + `-a`/`--all` | commit by explicit path |
| tag rewrite | `tag` + `-f` / `-d` | shared-history tags are append-only |
| history rewrite | `rebase` + `-i` | no interactive rebase on the trunk |

Each maps 1:1 to a documented law and each is a pure function of the command
string. The win is not new *authority* — the hooks already refuse most of these
eventually — it is **earlier, structured** refusal: a `VerdictDeny` carrying the
specific law as a bounded-disclosure witness, so an integrated agent's loop
self-corrects without ever spawning the doomed `git` process.

## 4. What a prefilter CANNOT decide

These are the laws people *want* in the prefilter most, and they are precisely the
ones a stateless rung cannot honestly enforce. Naming the boundary is half the
research:

- **OFF_TRUNK (commit/push because the current branch isn't `main`).** The current
  branch is repo state (`git rev-parse --abbrev-ref HEAD`), not in the argv. A
  prefilter could read it, but only by coupling the fast decide path to disk + a
  per-call `git` spawn, and the answer is racy (the branch can change between check
  and op). It stays a git hook. (gitgate *can* catch the argv-visible half —
  `git checkout -b` / `switch -c` / `worktree add` — and may grow that later, but
  the "already off-trunk" case is irreducibly stateful.)
- **Sweeping a peer's staged files.** Needs the live index (`git diff --cached`)
  and which hunks the agent itself staged. Mutable state, not in the call, racy on
  a shared tree.
- **Foreign `MERGE_HEAD` in flight.** A transient `.git` file that appears and
  disappears between check and op (TOCTOU). Not an argv decision.
- **Arbitrary-shell intent.** A git op laundered through an alias (`alias g=git`),
  a wrapper script, `eval`, backticks, or command injection is not safely
  recoverable by substring or a simple tokenizer. The honest stance is the same as
  the existing self-modify floor: conservative tells, *not* a claim of full
  coverage. A determined agent can evade any argv-level rung.
- **Stamp binding / claim-honesty (`dos verify`).** Binding a shipped subject to a
  real lane and corroborating it against the diff is the referee's job, over git
  history, post-commit. Re-implementing it in a rung would be a second source of
  truth that drifts from `dos.toml`.
- **Anything, against a client that bypasses the kernel.** The prefilter only sees
  calls *routed through it* (gateway / MCP / in-process agent loop). A direct HTTP
  client or an out-of-band shell hits the git hooks, not this rung.

## 5. The design: a `gitgate` rung

A new mechanism-tier leaf, `internal/gitgate`, registering an `abi.Adjudicator` at
**rank 35** (after `plancfi`/`ifc`, before `shipgate`/the rank-100 monitor). It is
pure — a string read plus an argv walk — so it execs nothing, imports only the
frozen ABI, and passes architest's interpreter-free / cgo-free / layered-DAG
gates. It mirrors `shipgate`: stateless, returns a `Verdict`, leaves the heavy
git work to others.

Two deliberate choices, both toward fail-closed:

- **Unconditional, not `CallScope`-scoped.** The synthesis first proposed scoping
  the rung to `{Bash, exec, run_shell, …}`. But the registry's own contract warns
  *"a wrong scope is a fail-OPEN bug"* — a git hazard via an unrecognized shell
  tool name (Cursor's `run_terminal_cmd`, an MCP shell wrapper) would slip through.
  For a security floor that is the wrong direction. gitgate instead detects a shell
  call by the *presence of a `command`/`cmd` arg* and Defers on everything else, so
  an unknown shell tool name is covered, not exempted.
- **`ReasonPolicyBlock` + a rich witness Claim, not a new reason code.** Every
  hazard cites the existing closed-vocabulary `POLICY_BLOCK` and carries the
  specific law + corrective move in the witness Claim ("force-push refused… re-run
  without --force"). This keeps the model's label space stable (no new code to
  train on) while still giving the agent a precise, repairable reason.

The fold does the rest: `gitgate` returns `Deny` (foldRank 100) for a hazard and
`Defer` (foldRank 1) otherwise, so it wins over a downstream Allow when it has an
opinion and never weakens another rung when it doesn't.

An operator whose git policy differs from this repo's trunk discipline sets
`FAK_GITGATE=off` — the rung then leaves itself unregistered and Defers by absence,
mirroring the `FLEET_*_GUARD=off` hook escapes. No fork.

## 6. The honest boundary

gitgate is **enforcing in-path and advisory everywhere else.** It is the earlier,
machine-readable complement to the git hooks, never their replacement: the hooks
remain the floor-of-last-resort that binds every actor (including a human, and an
agent that bypasses the kernel) at the git transaction boundary. And it is a
conservative tokenizer, not a shell parser — over-broad where a refusal is cheap
(it will deny `git push --force --dry-run`, a harmless preview), under-precise
where evasion is determined (aliases, wrappers, `$(…)` beyond the simple cases it
happens to catch). Both limits are documented in the code's package doc, not hidden.

## 7. What shipped, what's next

Shipped: `internal/gitgate` (the rung + the hazard table above), table-driven tests
covering the hazards and the load-bearing negatives (a flag inside a quoted commit
message, `git` mentioned in `echo`/`grep`, a safe `git push -u`), wired into the
defconfig and proven live through `fak preflight` (`verdict=DENY reason=POLICY_BLOCK
by=gitgate`).

Next, in rough value order, all additive (no ABI change): the argv-visible OFF_TRUNK
half (`checkout -b` / `switch -c` / `worktree add`); a commit-message-shape check
when the message is an in-band `-m` arg (the prefilter dual of `check_commit_msg`);
a `git push --delete`/colon-refspec remote-delete guard; and — for the stateful
laws — a witness-resolver path (like `internal/witness`) that reads the current
branch / `MERGE_HEAD` *off the fast path* to gate OFF_TRUNK and the merge hazards
without putting a `git` spawn on every decide.
