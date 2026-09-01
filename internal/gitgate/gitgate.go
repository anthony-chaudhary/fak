// Package gitgate is a git-aware kernel PREFILTER: a registered Adjudicator rung
// that inspects a shell tool call (Bash / exec / run_shell / ...) carrying a
// `command` string, recognizes the `git` invocation inside it, and PROVABLY
// REFUSES the structurally-decidable git hazards BEFORE the command runs. It turns
// a doomed git command (force-push, commit --amend, add -A, --no-verify, tag -f,
// rebase -i, reset --hard, clean -f, checkout ., a branch/worktree open) from
// "the process runs, a git hook rejects it, the agent re-plans"
// into a deny-as-value AT THE CALL BOUNDARY carrying a repairable, law-citing
// reason the agent loop consumes.
//
// WHY A SEPARATE RUNG (not the monitor's commandWrites). The adjudicator's
// existing git logic — shellWriteVerbs ("git apply"/"git checkout"/...), and the
// `git -C <dir>` / `git --git-dir` mutating-subcommand parse — fires ONLY to
// protect a guarded tree from SELF_MODIFY: it is scoped to "a WRITE into
// internal/abi/, .git/, dos.toml, ...". These hazards are orthogonal to that. A
// `git push --force` to the shared trunk touches no guarded tree, so the
// self-modify floor never sees it. gitgate is the general git-SHAPE floor: the
// in-kernel dual of the repo's git HOOKS (tools/githooks/*). The hooks bind every
// actor (Claude Code, Codex, a human) at the git transaction boundary — defense in
// depth; this rung binds an agent that routes its tool calls THROUGH the kernel,
// one step earlier, with a machine-readable reason instead of a stderr message.
//
// WHAT IT DELIBERATELY DOES NOT DO (the honest boundary — see the RESEARCH note,
// docs/notes/RESEARCH-git-in-kernel-prefilters-*.md):
//
//   - TOKENIZER + A STATIC UNWRAP PASS, NOT A SHELL INTERPRETER. classify() runs the
//     hazard table over cmd AND over every command string the shell grammar wraps
//     around a git call that the flat tokenizer cannot see on its own — a `$(...)` /
//     backtick command substitution and the `-c` string of a `bash -c '...'` /
//     `sh -c '...'` sub-shell, recursively (unwrapShellSources). Pipes, `&&`/`||`/`;`,
//     and newline already segment inside the tokenizer. So a force-push laundered
//     through a pipe, an operator, a `$()`, or a `bash -c` string is now REFUSED, not
//     waved through. What stays OUT OF SCOPE is EXPANSION, which is provably undecidable
//     in a static pre-call pass: `$VAR` (`git $CMD --force`), an `alias`, and `eval`
//     all need runtime state (the variable value, the alias table, the eval result)
//     this pass does not have. Those — plus a wrapper script (`mygit push -f`) — DEGRADE
//     to defer/opaque (never to allow) and remain the git hooks' job. Like the
//     self-modify floor it mirrors, this rung is over-broad where a refusal is cheap and
//     under-precise where a determined agent can evade; it never CLAIMS full coverage.
//
//     WHY HAND-ROLLED, NOT mvdan.cc/sh (#823). #823 proposed wiring the unwrap pass to
//     mvdan.cc/sh/v3/syntax (a real shell AST). We implement it hand-rolled instead
//     because the module is ZERO-EXTERNAL-DEPS (go.mod, line 7) — there is no go.sum, and
//     code_quality_scorecard.py's `deps` KPI counts any added require / a present go.sum as
//     debt. Pulling mvdan.cc/sh + a go.sum onto the live decision path would break the
//     single-static-binary invariant DIRECTION.md rests on. And a real AST buys NO verdict
//     here: on every laundering case #823 names — a pipe, an `&&`/`||`/`;` operator, a
//     `$(...)` / backtick substitution, a `bash -c`/`sh -c` string — this pass already
//     REFUSES (proven load-bearing in gitgate_launder_test.go), and it hits the SAME
//     expansion wall a real parser does: `$VAR`, `eval`, an `alias`, and a command-
//     substitution RESULT (`git $(echo push) --force`) need runtime state no STATIC parse
//     has, so they stay opaque under mvdan/sh too. The AST would only change which esoteric
//     quoting the lexer disclaims — never a verdict on the hazards this rung exists to catch.
//
//   - ARGV-DECIDABLE HAZARDS ONLY. Laws that need REPO STATE — OFF_TRUNK (the
//     current branch), the shared-tree staging sweep (the live index), a peer's
//     in-flight MERGE_HEAD (a transient .git file) — are NOT decidable in a pure,
//     stateless prefilter. Reading them would couple the fast decide path to disk
//     plus a per-call git spawn and a TOCTOU race, so they stay with the witness
//     resolver (internal/witness, off the fast path) and the git hooks. This rung
//     DEFERS on them — the fold passes to the next link, fail-closed by default.
//
//   - ENFORCING ONLY IN-PATH. A client that bypasses the kernel hits the git
//     hooks, not this rung. gitgate is the earlier, in-path complement to the
//     hooks, never their replacement.
//
// COLLECTIVE-COMMIT BARRIER. The synthetic gitgate.collective_commit tool is a
// pure argv/lease check for a many-writer shared-trunk commit plan: held lease
// trees must be pairwise disjoint, every writer path must sit inside that
// writer's lease, and the final ordered commit pathspec may touch only the union
// of paths those writers declared. This borrows the MPI_File_write_all shape
// (many ranks, one shared file, a consistency view), but it is NOT distributed
// filesystem I/O and claims no cross-machine transaction or atomicity beyond git
// plus the lease partition. Truly stateful checks — live index sweep, current
// branch, a peer's MERGE_HEAD — stay deferred to the witness resolver and git
// hooks, not this in-path pure rung.
//
// The structural decision is PURE (a string read + an argv walk). When a recorder
// is explicitly wired, a denial also appends a best-effort side-ref note through
// internal/witness; the verdict never depends on that forensic write.
package gitgate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/shelltoken"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// hazard is one structurally-decidable git refusal: a (subcommand, flag) pair the
// trunk discipline forbids, plus the law text cited back to the agent so a deny is
// repairable rather than opaque. Flags are matched in two forms: the long flag
// (exact, or `--flag=value` for the optional-value forms like --force-with-lease)
// and, when short != 0, that short LETTER appearing in a single-dash cluster
// (so `git commit -am "x"` catches the bundled `-a`). Matching is per-subcommand,
// so the same letter means different things safely: `-n` is no-verify for commit
// but dry-run for push, `-d` is delete for push/tag — only the listed pairs fire.
type hazard struct {
	sub   string // git subcommand it applies to (e.g. "push", "commit")
	long  string // long flag that triggers it, e.g. "--force" ("" = none)
	short byte   // short flag LETTER triggering it in a -cluster (0 = none); case-sensitive
	law   string // the agent-facing reason cited in the deny witness
}

const neverAmendSharedReason = "NEVER_AMEND_SHARED"

const neverAmendSharedLaw = neverAmendSharedReason + ": shared-history rewrite refused: never amend, rebase, or force-push on the shared trunk. Make a new path-scoped commit (`git commit -- <paths>`, or `fak commit --path <path>`), or fetch and merge the trunk in place."

// defaultHazards is the repo's structurally-decidable trunk discipline, encoded
// once. Every entry maps 1:1 to a documented law (AGENTS.md / CLAUDE.md) that today
// only a doc sentence or an after-the-fact git hook enforces.
var defaultHazards = []hazard{
	// Never force-push the shared trunk (AGENTS.md). Closes the gap that the
	// named-tool `git_push` deny leaves for a Bash command="git push --force".
	{sub: "push", long: "--force", short: 'f', law: neverAmendSharedLaw + " force-push refused: re-run `git push` WITHOUT --force/-f."},
	{sub: "push", long: "--force-with-lease", law: neverAmendSharedLaw + " force-push refused: re-run `git push` WITHOUT --force-with-lease."},
	// Never skip the guards / signing.
	{sub: "push", long: "--no-verify", law: "skip-hooks refused: never bypass the pre-push guards (push --no-verify). Push with the hooks enabled."},
	// Do not delete a remote ref from an agent.
	{sub: "push", long: "--delete", short: 'd', law: "remote-ref delete refused: do not delete a remote branch from an agent (push --delete/-d)."},
	// Never amend in a shared tree — HEAD moves between peers (CLAUDE.md).
	{sub: "commit", long: "--amend", law: neverAmendSharedLaw + " amend refused: HEAD moves between peers. If the existing commit has only a message typo, leave that message intact; shared history has no compliant rewrite path. Make a NEW path-scoped commit only for new content, and validate future subjects first with `fak commit --preview`."},
	{sub: "commit", long: "--no-verify", short: 'n', law: "skip-hooks refused: never bypass the commit guards (commit --no-verify/-n). Commit with the hooks enabled."},
	{sub: "commit", long: "--no-gpg-sign", law: "skip-signing refused: do not disable commit signing (commit --no-gpg-sign)."},
	// Commit by explicit path — never sweep a peer's files in a shared tree (AGENTS.md).
	{sub: "commit", long: "--all", short: 'a', law: "commit-by-explicit-path: `git commit -a/--all` sweeps every tracked change in a shared tree (AGENTS.md). Stage explicit paths, then `git commit -- <paths>`."},
	{sub: "add", long: "--all", short: 'A', law: "commit-by-explicit-path: `git add -A/--all` stages everything, incl. a peer's files (AGENTS.md). Add explicit paths instead."},
	{sub: "add", long: "--update", short: 'u', law: "commit-by-explicit-path: `git add -u` stages every tracked change (AGENTS.md). Add explicit paths instead."},
	// Shared-history tags are append-only.
	{sub: "tag", long: "--force", short: 'f', law: "tag-force refused: never overwrite a tag (tag -f/--force); shared-history tags are append-only."},
	{sub: "tag", long: "--delete", short: 'd', law: "tag-delete refused: do not delete a tag from an agent (tag -d/--delete)."},
	// No history rewrite on the shared trunk.
	{sub: "rebase", long: "--interactive", short: 'i', law: neverAmendSharedLaw + " history-rewrite refused: no interactive rebase on the shared trunk (rebase -i/--interactive)."},
	{sub: "pull", long: "--rebase", law: neverAmendSharedLaw + " pull-rebase refused: fetch, then merge the trunk in place instead of rebasing shared-trunk commits."},
	// Never --autostash in the shared tree: an aborted/conflicted rebase pops the
	// stash back as a working-tree blob, dumping a peer's in-flight WIP into your
	// tree and leaving a dangling `autostash` stash (CLAUDE.md / [[fak-shared-tree-high-churn-commit]]).
	// The remedy MUST name `git merge`, not `git rebase`: rebase is refused
	// categorically below (sub=="rebase") for the shared trunk, so a remedy that
	// says "then rebase" points the agent straight at another refusal — the
	// self-refuting-remedy loop (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
	{sub: "rebase", long: "--autostash", law: "autostash refused: never `rebase --autostash` in the shared tree — an abort pops the stash into your working tree, sweeping a peer's WIP (CLAUDE.md). Reach a clean tree first (stash explicit paths or commit your work), then `git fetch` + `git merge origin/main` in place — never rebase the shared trunk."},
	{sub: "pull", long: "--autostash", law: "autostash refused: never `pull --rebase --autostash` in the shared tree — a conflict abort pops the stash into your working tree, sweeping a peer's WIP (CLAUDE.md). Reach a clean tree first (stash explicit paths or commit your work), then `git fetch` + `git merge origin/main` in place — never rebase the shared trunk."},
	// Never destroy the shared working tree: `reset --hard` discards tracked-file
	// changes and `clean -f` deletes untracked files — both sweep a peer's WIP
	// (AGENTS.md destructive-op list). A `--soft`/`--mixed` reset and a `clean -n`
	// dry-run are non-destructive and do not match.
	{sub: "reset", long: "--hard", law: "reset-hard refused: `git reset --hard` discards every working-tree change to tracked files — incl. a peer's unstaged WIP in the shared tree (AGENTS.md destructive-op list). Reconcile in place, or scope your undo: `git restore -- <your-paths>`."},
	{sub: "clean", long: "--force", short: 'f', law: "clean-force refused: `git clean -f` deletes untracked files — incl. a peer's new files and your own uncommitted work in the shared tree (AGENTS.md). Remove specific files explicitly; never whole-tree clean a shared checkout."},
	// Never open a feature branch — the argv-decidable half of OFF_TRUNK (AGENTS.md).
	{sub: "checkout", short: 'b', law: offTrunkBranchLaw},
	{sub: "checkout", short: 'B', law: offTrunkBranchLaw},
	{sub: "switch", long: "--create", short: 'c', law: offTrunkBranchLaw},
	{sub: "switch", long: "--force-create", short: 'C', law: offTrunkBranchLaw},
	// `git push --mirror` overwrites EVERY remote ref (and deletes remote refs
	// absent locally) — catastrophic on a shared remote (a superset of force-push).
	{sub: "push", long: "--mirror", law: "push-mirror refused: `git push --mirror` overwrites EVERY remote ref and deletes remote refs absent locally — catastrophic on a shared remote. Push specific refs without --mirror."},
	// `git push --prune` deletes every remote ref absent from THIS clone under the
	// pushed refspec (no --mirror needed) — the converge that emptied the fleet's
	// refs/fak/locks/* and refs/fak/wip/* from a stale clone (#5360). Same catastrophe
	// class as --mirror; scoped to push so a safe fetch/remote prune (local-tracking
	// cleanup) stays deferred.
	{sub: "push", long: "--prune", law: "push-prune refused: `git push --prune` deletes every remote ref absent from THIS clone under the pushed refspec — on a shared fleet remote where each clone holds only a subset of refs, that mass-deletes peers' branches, lock leases, and WIP checkpoint refs (a superset of push --delete). Push specific refs without --prune; a safe `git fetch --prune` prunes only local remote-tracking refs."},
}

const dotAddLaw = "commit-by-explicit-path: `git add .` stages the whole tree (AGENTS.md). Add explicit paths instead."

// pushForceRefspecLaw / pushDeleteRefspecLaw fire on the REFSPEC spellings of a
// force-push and a remote-ref delete, which the flag-matching hazard table cannot
// see: `git push origin +<refspec>` forces the remote update exactly like --force
// (just scoped to that ref), and `git push origin :<dst>` (an empty source refspec)
// deletes the remote ref exactly like push --delete. Scoped to `push` only — a
// fetch/pull `+refspec` merely force-updates a LOCAL remote-tracking ref (the
// standard fetch refspec shape), and the bare matching-branches `:` neither forces
// nor deletes, so both stay deferred.
const pushForceRefspecLaw = neverAmendSharedLaw + " force-push refused: a `+<refspec>` push forces the remote ref exactly like --force. Push WITHOUT the leading `+`."

const pushDeleteRefspecLaw = "remote-ref delete refused: `git push origin :<branch>` (empty source refspec) deletes the remote ref exactly like push --delete/-d. Do not delete a remote branch from an agent."

// massRemoteRefDeleteThreshold is the number of remote refs a single push may name
// for deletion before the act stops being "retire this one stale ref" and becomes a
// CONVERGE over a namespace — the class of act that wiped ~8400 lock/wip refs on
// 2026-07-22 (#5360). 8 is a CHOSEN number, not a measured one: nothing was sampled
// to derive it. It is picked to sit above any plausible hand-written cleanup (an
// agent retiring its own lease names one ref, a small batch names a handful) and far
// below a namespace sweep, so the two acts get different laws and different remedies.
// Both sides of it still DENY — the threshold selects which law is cited, never
// whether the push is admitted — so moving it can never open a hole.
const massRemoteRefDeleteThreshold = 8

// massRemoteRefDeleteLaw is cited when one push names >= massRemoteRefDeleteThreshold
// remote refs for deletion. Spelled-out-one-refspec-at-a-time is the same catastrophe
// as --prune/--mirror, just typed longhand, so it may not ride the singular
// "delete a remote branch" law: that law's remedy is silent about the legitimate bulk
// retirement, and a refusal that names no route is indistinguishable from having no
// route (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
func massRemoteRefDeleteLaw(n int) string {
	return fmt.Sprintf("mass-remote-ref-delete refused: this push names %d remote refs for deletion in one act (threshold %d) — a namespace converge spelled one refspec at a time, the same catastrophe as `push --prune`/`--mirror` (#5360: an unattributed converge wiped ~8400 lock/wip refs, incl. peers' live lock leases and every WIP checkpoint). A bulk ref retirement is not an agent shell act: run it as a compiled fak verb that pushes through internal exec (the `fak sync push` shape), scoped to refs that verb has PROVEN expired, with FLEET_ALLOW_REF_PRUNE=1 for the pre-push hook. To retire ONE ref you own, name that one ref.", n, massRemoteRefDeleteThreshold)
}

// pushMirrorConfigLaw is the CONFIG spelling of the already-refused `push --mirror`
// flag. `remote.<name>.mirror=true` makes a PLAIN `git push <remote>` behave exactly as
// if --mirror were on the command line (git-config(1)), so the flag refusal alone is a
// half-closed door: the same mass delete rides in with no hazardous flag on the argv.
// Same durable-sibling shape as configHooksLaw / configSignLaw, and it must catch both
// the per-invocation `git -c` override and the persistent `git config` write.
const pushMirrorConfigLaw = "push-mirror refused: `remote.<name>.mirror=true` makes a PLAIN `git push <remote>` behave exactly as `git push --mirror` — it overwrites EVERY remote ref and deletes every remote ref absent from THIS clone, mass-deleting peers' branches, lock leases and WIP refs on a shared fleet remote (#5360). It is the unflagged spelling of a flag this gate already refuses. Leave it off and push specific refs by refspec; for local-tracking cleanup use `git fetch --prune`, which never touches the remote."

// xargsPushLaw fires on a `git push` whose arguments are fed by `xargs`. This is the
// FAIL-CLOSED rung of #5360: the refspecs arrive on a PIPE, so the set of remote refs
// the push updates or DELETES is not in the argv at all — it cannot be bounded,
// attributed, or shown to the operator afterwards. That is not a hypothetical shape:
// ~8400 refspecs do not fit on one command line, so a converge of that size can only
// reach git through a generator pipe, and `xargs git push origin --delete` launders a
// spelling this gate refuses outright when written directly. Uncertainty in a
// destructive path resolves toward REFUSING, never toward admitting.
const xargsPushLaw = "unattributable-push refused: a `git push` fed by `xargs` takes its refspecs from a PIPE, so the set of remote refs it updates — and every ref it DELETES — is absent from the argv and cannot be bounded or attributed (#5360: an unattributed converge wiped ~8400 lock/wip refs; that many refspecs reach git only through a generator pipe, and `xargs git push --delete` launders a spelling this gate refuses when written directly). A destructive push whose ref set cannot be named fails CLOSED. Push refs you can name: one explicit refspec per `git push`. For a bulk retirement, run a compiled fak verb that pushes through internal exec (the `fak sync push` shape), scoped to refs it has PROVEN expired."

// unscopedStashLaw fires on a whole-tree stash CREATE in the shared trunk. A bare
// `git stash` (or `git stash push`/`save` with no pathspec) snapshots EVERY dirty
// file — including a peer's in-flight WIP — then leaves it parked in a stash that
// the workflow never pops, stranding that peer's work (the `peer-wip-before-*` /
// `WIP on main` stash pile this law exists to prevent). The clean-tree move on a
// shared trunk is to commit your own files by explicit path or, if you must park
// them, scope the stash to YOUR paths: `git stash push -- <your-paths>`
// (CLAUDE.md / [[fak-shared-tree-high-churn-commit]]).
const unscopedStashLaw = "unscoped-stash refused: a bare `git stash`/`git stash push`/`save` snapshots the WHOLE shared tree, sweeping a peer's in-flight WIP into a stash that never gets popped (CLAUDE.md). To reach a clean tree, commit your files by explicit path, or scope the stash to your own paths: `git stash push -- <your-paths>`."

// wholeTreeDiscardLaw fires on `git checkout .` / `git restore .` — a whole-tree
// working-tree discard. With a `.` operand (with or without a leading `--`) the op
// reverts EVERY working-tree change in the shared checkout (or, with --staged,
// unstages the shared index), sweeping a peer's in-flight WIP. A SPECIFIC-path
// revert (`git checkout -- <file>`) is left alone — only the whole-tree `.` form is
// refused, the same shape as the `git add .` law.
const wholeTreeDiscardLaw = "whole-tree-discard refused: `git checkout .` / `git restore .` operates on the WHOLE shared tree — discarding every working-tree change (or unstaging the shared index), sweeping a peer's in-flight WIP (AGENTS.md). Scope your undo to your own paths: `git restore -- <your-paths>`."

// offTrunkBranchLaw fires on a branch/worktree CREATE — the argv-decidable shape of
// the OFF_TRUNK escape (`git checkout -b`, `git switch -c`, `git worktree add`). The
// trunk guard refuses off-trunk commits after the fact; this catches the branch open
// at the call boundary. Switching to an EXISTING branch needs repo state (is the
// target development branch?) and stays deferred — only the unconditional CREATE forms fire here.
const offTrunkBranchLaw = "off-trunk refused: `git checkout -b` / `git switch -c` / raw `git worktree add` opens an unmanaged branch or worktree. Work directly on the configured development branch. For an explicitly requested detached worker, use the collision-safe sanctioned route instead: `fak worktree worker prepare --id <worker-id> --scope <path>`, then `fak worktree worker land --id <worker-id>` and `fak worktree worker reap --id <worker-id>` (AGENTS.md OFF_TRUNK)."

// historyRewriteLaw fires on a whole-history rewrite subcommand (`git filter-branch`,
// `git filter-repo`). These rewrite every commit on the shared trunk — the same class
// of forbidden act as a force-push or an interactive rebase, just applied wholesale.
const historyRewriteLaw = "history-rewrite refused: `git filter-branch` / `git filter-repo` rewrites shared history — forbidden on the trunk (AGENTS.md: never rewrite or force-push shared history). Make a new commit instead."

// configHooksLaw fires on a PERSISTENT `git config core.hooksPath ...` write — the
// durable sibling of the `git -c core.hooksPath=` per-invocation override the global
// scan already catches. Relocating the hooks directory disables the commit/push
// guards for every subsequent git op, not just one. A read (--get/--list) or an
// --unset (which RESTORES the default hooks) is safe and stays deferred.
const configHooksLaw = "skip-hooks refused: `git config core.hooksPath ...` persistently redirects the hooks directory, disabling the commit/push guards for every later git op (the durable sibling of `git -c core.hooksPath=`). Do not relocate hooks; reach the goal with the guards enabled."

// configSignLaw fires on a PERSISTENT `git config commit.gpgsign false` — the
// durable sibling of the per-commit `--no-gpg-sign` flag the hazard table already
// catches: it turns signing off for every later commit, not just one. Only the
// SET-to-false form fires; setting it true, a read, or an --unset is safe.
const configSignLaw = "skip-signing refused: `git config commit.gpgsign false` persistently disables commit signing for every later commit (the durable sibling of `commit --no-gpg-sign`). Leave signing enabled."

// ToolCollectiveCommit is the synthetic tool name for the collective-commit
// barrier. It never shells out; its args are a CollectiveCommitPlan JSON object.
const ToolCollectiveCommit = "gitgate.collective_commit"

// GitGate is the registered rung. Construct with New; the package Default instance
// registers itself in init() unless FAK_GITGATE=off. The rule table is read-only
// after construction, and the optional recorder is atomic, so one instance is safe
// for the whole process.
type GitGate struct {
	rules []hazard
	rec   atomic.Pointer[witness.Recorder]
}

// New builds a gate carrying the default trunk-discipline hazard table.
func New() *GitGate { return &GitGate{rules: defaultHazards} }

// SetRecorder wires an optional durable witness sink. When set, a refusal verdict
// also appends a witness.Decision to refs/notes/fak/decisions, best-effort: a note
// write failure never changes the deny/defer/allow verdict.
func (g *GitGate) SetRecorder(r *witness.Recorder) { g.rec.Store(r) }

func (g *GitGate) Caps() []abi.Capability { return nil }

// Adjudicate refuses a structurally-decidable git hazard in a shell tool call.
// A non-shell call (no command/cmd arg), a shell call whose command names no git
// op, and every git op whose hazard needs repo state all DEFER — the rung has no
// opinion, the fold passes to the next link (fail-closed: a Defer never grants an
// allow). A recognized hazard returns a PROVABLE Deny citing ReasonPolicyBlock,
// with the offending law carried as a bounded-disclosure witness Claim (the agent
// sees the specific rule + the corrective move, never the whole policy).
func (g *GitGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || len(g.rules) == 0 {
		return deferVerdict()
	}
	if c.Tool == ToolCollectiveCommit {
		return g.adjudicateCollective(ctx, c)
	}
	if c.Tool == ToolSweepGuard {
		return g.adjudicateSweepGuard(ctx, c)
	}
	cmd := shellCommand(ctx, c)
	// Cheap reject: no command arg, or no "git" anywhere in it — nothing to prove.
	if cmd == "" || !strings.Contains(strings.ToLower(cmd), "git") {
		return deferVerdict()
	}
	if law, denied := g.classify(cmd); denied {
		v := abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonPolicyBlock,
			By:      "gitgate",
			Payload: abi.WitnessPayload{Claim: law},
			// Surface the corrective move to the AGENT, not just the forensic channel.
			// The law names the sanctioned route, but it rides Payload.Claim
			// (Detail["claim"]), which the in-band refusal note DROPS — only
			// Detail["remedy"] (fed from Meta["fix"]) is rendered by remedyNote. Without
			// this, a gitgate Deny reaches the agent as a bare POLICY_BLOCK/TERMINAL with
			// no recovery path (#3524). Feed the same law through the one render seam the
			// arg-predicate rung already uses (Meta["fix"]) so the agent sees the route.
			Meta: map[string]string{"fix": law},
		}
		// Every law here is AUTHORED as "<law-id>[ refused]: <prose>", so its leading
		// atom is the law's own id — and it is the ONLY part of the law that is a
		// stable key rather than agent-facing prose. Promote it to the verdict's
		// closed-vocabulary rule id so a fleet operator can separate skip-hooks from
		// off-trunk from reset-hard: all seven trunk laws in the measured corpus land
		// on the same ("gitgate", "POLICY_BLOCK") pair, and telling them apart today
		// means prefix-matching a claim up to 447 characters long (#5863).
		// abi.DenyRuleID admits only declared ids, so a law whose id is not yet in the
		// vocabulary stamps nothing rather than leaking its prose.
		if rule, ok := abi.DenyRuleID(law); ok {
			v.Meta[abi.MetaDenyRule] = rule
		}
		g.recordRefusal(ctx, "gitgate", gitgateReasonClass(law, abi.ReasonName(v.Reason)), []string{"shell", "-c", cmd}, nil)
		return v
	}
	return deferVerdict()
}

func deferVerdict() abi.Verdict { return abi.Verdict{Kind: abi.VerdictDefer, By: "gitgate"} }

func gitgateReasonClass(law, fallback string) string {
	if strings.Contains(law, neverAmendSharedReason) {
		return neverAmendSharedReason
	}
	return fallback
}

// CollectiveCommitPlan is the argv/lease-decidable shape verified by the
// collective-commit barrier. Writers are independent workers holding lease trees;
// Paths are the repo-relative paths that writer contributes; CommitPaths is the
// ordered `git commit -- <paths>` pathspec the coordinator plans to run.
type CollectiveCommitPlan struct {
	Writers     []CollectiveWriter `json:"writers"`
	CommitPaths []string           `json:"commit_paths"`
}

// CollectiveWriter is one participant in a CollectiveCommitPlan.
type CollectiveWriter struct {
	ID     string   `json:"id"`
	Leases []string `json:"leases"`
	Paths  []string `json:"paths"`
}

// CollectiveFinding is the structured result of CheckCollectiveCommit.
type CollectiveFinding struct {
	OK     bool
	Reason abi.ReasonCode
	Claim  string
}

// CheckCollectiveCommit verifies the pure collective-commit invariants without
// reading repo state: lease trees are pairwise disjoint, writer paths stay inside
// their own leases, and the final commit pathspec is covered by the union of
// writer-declared paths.
func CheckCollectiveCommit(plan CollectiveCommitPlan) CollectiveFinding {
	if len(plan.Writers) == 0 {
		return malformedCollective("collective-commit plan has no writers")
	}
	if len(plan.CommitPaths) == 0 {
		return malformedCollective("collective-commit plan has no explicit commit paths")
	}

	var leases []leaseTree
	var declared []declaredPath
	for wi, w := range plan.Writers {
		id := strings.TrimSpace(w.ID)
		if id == "" {
			id = fmt.Sprintf("writer[%d]", wi)
		}
		if len(w.Leases) == 0 {
			return malformedCollective(fmt.Sprintf("collective-commit writer %s has no leases", id))
		}
		writerLeases := make([]string, 0, len(w.Leases))
		for _, raw := range w.Leases {
			tree, ok := cleanLeaseTree(raw)
			if !ok {
				return malformedCollective(fmt.Sprintf("collective-commit writer %s has invalid lease %q", id, raw))
			}
			for _, prev := range leases {
				if treesOverlap(prev.tree, tree) {
					return leaseFinding(fmt.Sprintf("collective-commit lease conflict: writer %s lease %q overlaps writer %s lease %q; held leases must be pairwise disjoint", id, tree, prev.owner, prev.tree))
				}
			}
			leases = append(leases, leaseTree{owner: id, tree: tree})
			writerLeases = append(writerLeases, tree)
		}
		if len(w.Paths) == 0 {
			return malformedCollective(fmt.Sprintf("collective-commit writer %s has no committed paths", id))
		}
		for _, raw := range w.Paths {
			p, ok := cleanRepoPath(raw)
			if !ok {
				return malformedCollective(fmt.Sprintf("collective-commit writer %s has invalid path %q", id, raw))
			}
			if !coveredByAnyTree(p, writerLeases) {
				return leaseFinding(fmt.Sprintf("collective-commit path outside leased tree: writer %s path %q is outside leases [%s]", id, p, strings.Join(writerLeases, ", ")))
			}
			declared = append(declared, declaredPath{owner: id, path: p})
		}
	}

	for _, raw := range plan.CommitPaths {
		p, ok := cleanRepoPath(raw)
		if !ok {
			return malformedCollective(fmt.Sprintf("collective-commit has invalid commit path %q", raw))
		}
		if !coveredByDeclaredPath(p, declared) {
			return leaseFinding(fmt.Sprintf("collective-commit union violation: commit path %q is not covered by any writer-declared path", p))
		}
	}
	return CollectiveFinding{OK: true}
}

func (g *GitGate) adjudicateCollective(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	var plan CollectiveCommitPlan
	b := refBytes(ctx, c.Args)
	if len(b) == 0 {
		return collectiveDeny(malformedCollective("collective-commit missing JSON args"))
	}
	if err := json.Unmarshal(b, &plan); err != nil {
		return collectiveDeny(malformedCollective("collective-commit malformed JSON args: " + err.Error()))
	}
	finding := CheckCollectiveCommit(plan)
	if finding.OK {
		return abi.Verdict{Kind: abi.VerdictAllow, By: ToolCollectiveCommit}
	}
	v := collectiveDeny(finding)
	g.recordRefusal(ctx, ToolCollectiveCommit, abi.ReasonName(v.Reason), []string{ToolCollectiveCommit}, plan.CommitPaths)
	return v
}

func collectiveDeny(f CollectiveFinding) abi.Verdict {
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  f.Reason,
		By:      ToolCollectiveCommit,
		Payload: abi.WitnessPayload{Claim: f.Claim},
	}
}

func malformedCollective(claim string) CollectiveFinding {
	return CollectiveFinding{Reason: abi.ReasonMalformed, Claim: claim}
}

func leaseFinding(claim string) CollectiveFinding {
	return CollectiveFinding{Reason: abi.ReasonLeaseHeld, Claim: claim}
}

func (g *GitGate) recordRefusal(ctx context.Context, op, reason string, argv, tree []string) {
	rec := g.rec.Load()
	if rec == nil {
		return
	}
	d := witness.Decision{
		Op:          op,
		Verdict:     witness.VerdictRefuse,
		ReasonClass: reason,
		RefusedArgv: append([]string(nil), argv...),
		Tree:        append([]string(nil), tree...),
	}
	_ = rec.AppendDecision(ctx, "", d)
}

type leaseTree struct {
	owner string
	tree  string
}

type declaredPath struct {
	owner string
	path  string
}

func cleanLeaseTree(raw string) (string, bool) {
	s := strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	for strings.HasSuffix(s, "/**") {
		s = strings.TrimSuffix(s, "/**")
	}
	for strings.HasSuffix(s, "/*") {
		s = strings.TrimSuffix(s, "/*")
	}
	s = strings.TrimSuffix(s, "/")
	if strings.Contains(s, "*") {
		return "", false
	}
	return cleanRepoPath(s)
}

func cleanRepoPath(raw string) (string, bool) {
	s := strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	if s == "" || strings.ContainsRune(s, 0) || strings.HasPrefix(s, "/") {
		return "", false
	}
	p := path.Clean(s)
	if p == "." || p == ".." || strings.HasPrefix(p, "../") {
		return "", false
	}
	return p, true
}

func treesOverlap(a, b string) bool {
	return treeContains(a, b) || treeContains(b, a)
}

func treeContains(tree, p string) bool {
	return p == tree || strings.HasPrefix(p, tree+"/")
}

func coveredByAnyTree(p string, trees []string) bool {
	for _, tree := range trees {
		if treeContains(tree, p) {
			return true
		}
	}
	return false
}

func coveredByDeclaredPath(p string, declared []declaredPath) bool {
	for _, d := range declared {
		if treeContains(d.path, p) {
			return true
		}
	}
	return false
}

// CleanRepoPath is the exported form of the repo-path normalizer the collective-commit
// invariants use: it lower-noises a raw pathspec (backslash->forward, path.Clean) and
// reports false for anything that cannot be a committed repo-relative path — empty, a
// NUL, an absolute path, or an escape above the tree (".", "..", "../x"). The executor
// half (internal/safecommit) normalizes its requested and committed path sets through
// THIS one function so the policy and the executor share a single path rule.
func CleanRepoPath(raw string) (string, bool) { return cleanRepoPath(raw) }

// TreeContains reports whether repo-relative path p is tree or sits beneath it (tree ==
// p, or p has the prefix "tree/"). Exported so the executor's "did exactly the requested
// paths land" assertion uses the same containment the policy uses — a requested directory
// legitimately covers the files committed under it.
func TreeContains(tree, p string) bool { return treeContains(tree, p) }

// CoveredByAnyTree reports whether p is contained by at least one tree in trees. Exported
// as the set-membership primitive the executor folds over to find a committed file that NO
// requested path covers — the empirical signature of a peer-swept (raced) commit.
func CoveredByAnyTree(p string, trees []string) bool { return coveredByAnyTree(p, trees) }

// Classify is the pure, testable core: it reports the cited law and true if cmd
// contains a refused git hazard, else ("", false). Exported (via the method on
// the rule set) so tests exercise the tokenizer + table directly over command
// strings without building a ToolCall.
func (g *GitGate) Classify(cmd string) (string, bool) { return g.classify(cmd) }

func (g *GitGate) classify(cmd string) (string, bool) {
	cmd = stripQuotedHeredocBodies(cmd)
	// The unwrap pass yields cmd itself PLUS every command string the shell grammar
	// wraps around a git call the flat tokenizer cannot see: a `$(...)` / backtick
	// command substitution and the `-c` string of a `bash -c '...'` / `sh -c '...'`
	// sub-shell, recursively. Each recovered string is then tokenized + inspected by
	// the EXACT same defaultHazards rules — the pass only widens what the existing
	// rules can SEE, it adds no new hazard logic and changes no verdict. (Pipes,
	// `&&`/`||`/`;`, newline already segment correctly in tokenizeSegments.)
	for _, src := range unwrapShellSources(cmd) {
		for _, seg := range tokenizeSegments(src) {
			if argv := gitArgv(seg); argv != nil {
				if law, ok := g.inspectGit(argv); ok {
					return law, true
				}
				continue
			}
			// Not git in command position — but `xargs git ...` still RUNS git, with
			// its operands supplied by a pipe. Same rules, plus the provenance bit that
			// makes an unreadable push argv fail closed.
			if argv := gitArgvViaXargs(seg); argv != nil {
				if law, ok := g.inspectGitArgs(argv, true); ok {
					return law, true
				}
			}
		}
	}
	return "", false
}

// inspectGit walks the args of a git invocation (the tokens AFTER the `git`
// program word): it skips the value-bearing global options to locate the
// subcommand, catches a `-c core.hooksPath=...` skip-hooks override along the way,
// then matches the subcommand's flags against the hazard table.
func (g *GitGate) inspectGit(args []string) (string, bool) { return g.inspectGitArgs(args, false) }

// inspectGitArgs is inspectGit with the provenance of the argv: viaXargs is true when
// the git call was found behind an `xargs` (see gitArgvViaXargs), meaning its OPERANDS
// arrive on stdin and are not in the argv the gate can read.
func (g *GitGate) inspectGitArgs(args []string, viaXargs bool) (string, bool) {
	i, law, refused := inspectGitGlobalArgs(args)
	if refused {
		return law, true
	}
	if i >= len(args) {
		return "", false // no subcommand (e.g. `git --version`, `git -C x`)
	}
	sub := args[i]
	rest := args[i+1:]

	// A rebase that ADVANCES history stays categorically refused on the shared
	// trunk. The rebase STATE-CONTROL forms are not that act and no longer ride the
	// same refusal: `--abort` restores the pre-rebase HEAD and working tree (it is
	// the UNDO), `--quit` drops the rebase state without moving history further, and
	// `--show-current-patch` only reads. Refusing those made the law self-refuting —
	// a checkout that is ALREADY mid-rebase (started by a human, by another tool, or
	// by a `pull.rebase=true` config, none of which this argv-only rung can see) then
	// had no sanctioned exit, so the tree stayed parked in a conflicted detached-HEAD
	// state that fails every peer's commit in the shared checkout. That is the very
	// outcome the law exists to prevent, reached by refusing the repair — the
	// self-refuting-remedy loop this file already warns about for --autostash
	// (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md). `--continue` and `--skip`
	// keep applying commits, so they keep advancing the rewrite and stay refused;
	// because `--abort` is always available as the safe exit, nothing is trapped.
	if sub == "rebase" && !rebaseStateControlOnly(rest) {
		return neverAmendSharedLaw + " rebase refused: merge the trunk in place instead. If a rebase is ALREADY in progress, `git rebase --abort` (restore the pre-rebase HEAD and working tree) is allowed and is the sanctioned exit.", true
	}

	// The refspec spellings of force-push and remote-delete (see the law consts):
	// a push OPERAND starting with `+` forces the ref; one starting with `:`
	// (empty src, non-empty dst) deletes it. Flags are skipped — the hazard table
	// below owns them — and only `push` operands are hazardous refspecs.
	if sub == "push" {
		// FAIL CLOSED. Every other rule on this path decides from the argv; an
		// xargs-fed push has no argv to decide from — its refspecs, including the
		// deletions, are on a pipe. The gate therefore cannot show that the act is
		// bounded, and "cannot show it is bounded" on a mass-delete-capable path must
		// resolve to REFUSE, not to admit-with-a-warning: the 2026-07-22 converge was
		// exactly an unattributed deletion set, and admitting it a second time because
		// the evidence is missing would make the gate's silence mean consent.
		if viaXargs {
			return xargsPushLaw, true
		}
		// A converge spelled longhand — many `:ref` operands, or many refs after
		// --delete — is the SAME act as --prune, so it is judged by its scale before
		// the singular per-ref laws get to speak (both outcomes deny; this only picks
		// the law and the remedy).
		if n := countRemoteRefDeletes(rest); n >= massRemoteRefDeleteThreshold {
			return massRemoteRefDeleteLaw(n), true
		}
		for _, t := range rest {
			if strings.HasPrefix(t, "-") {
				continue
			}
			if strings.HasPrefix(t, "+") && len(t) > 1 {
				return pushForceRefspecLaw, true
			}
			if strings.HasPrefix(t, ":") && len(t) > 1 {
				return pushDeleteRefspecLaw, true
			}
		}
	}

	// `git add .` / `git add -- .` stages the whole tree regardless of flag order.
	if sub == "add" {
		for _, t := range rest {
			if t == "." {
				return dotAddLaw, true
			}
		}
	}

	// A whole-tree stash CREATE (bare `git stash`, or `git stash push`/`save`
	// with no pathspec) sweeps every dirty file, incl. a peer's WIP. Only the
	// stash CREATE forms are hazardous — list/show/pop/apply/drop/branch/clear
	// inspect or unwind an existing stash and never snapshot the tree. A `--`
	// (or a trailing pathspec) scopes the snapshot to the agent's own files, so
	// that form is allowed.
	if sub == "stash" && isUnscopedStashCreate(rest) {
		return unscopedStashLaw, true
	}

	// `git checkout .` / `git restore .` discards the WHOLE working tree (the `.`
	// operand may follow a `--`), the same shape as `git add .`. A specific-path
	// revert (`git checkout -- <file>`) is left alone — only `.` fires.
	if sub == "checkout" || sub == "restore" {
		for _, t := range rest {
			if t == "." {
				return wholeTreeDiscardLaw, true
			}
		}
	}

	// `git worktree add ...` opens a new worktree — the OFF_TRUNK escape (AGENTS.md).
	// Other worktree subcommands (list/remove/prune/move/lock) do not open one.
	if sub == "worktree" && len(rest) > 0 && rest[0] == "add" {
		return offTrunkBranchLaw, true
	}

	// `git filter-branch` / `git filter-repo` rewrite the whole shared history.
	if sub == "filter-branch" || sub == "filter-repo" {
		return historyRewriteLaw, true
	}

	// Persistent `git config` writes that disable a guard: `core.hooksPath` (relocates
	// the hooks dir) and `commit.gpgsign false` (turns signing off) — the durable
	// siblings of `git -c core.hooksPath=` and `commit --no-gpg-sign`. Refuse the SET
	// form only; a read (--get*/--list/-l), an --unset (restores the default), and
	// setting gpgsign back ON all fall through to defer.
	if sub == "config" {
		if law, refused := inspectGitConfig(rest); refused {
			return law, true
		}
	}

	return g.inspectGitRuleFlags(sub, rest)
}

func inspectGitGlobalArgs(args []string) (int, string, bool) {
	i := 0
	for i < len(args) {
		a := args[i]
		if !strings.HasPrefix(a, "-") {
			break
		}
		switch {
		case a == "-c" || a == "-C" || a == "--git-dir" || a == "--work-tree" || a == "--namespace" || a == "--exec-path":
			val := ""
			if i+1 < len(args) {
				val = args[i+1]
			}
			// Key-scope invocation config overrides so values that merely mention a
			// protected key do not trigger a refusal.
			if a == "-c" {
				key, cval, joined := splitConfigKey(strings.ToLower(val))
				if key == "core.hookspath" {
					return i, "skip-hooks refused: `git -c core.hooksPath=...` disables hooks for this invocation.", true
				}
				if isMirrorEnable(key, cval, joined) {
					return i, pushMirrorConfigLaw, true
				}
			}
			i += 2
		case strings.HasPrefix(a, "--") && strings.Contains(a, "="):
			if key, _, _ := splitConfigKey(strings.ToLower(strings.TrimPrefix(a, "--"))); key == "core.hookspath" {
				return i, "skip-hooks refused: a core.hooksPath override disables hooks for this invocation.", true
			}
			i++
		default:
			i++
		}
	}
	return i, "", false
}

func inspectGitConfig(rest []string) (string, bool) {
	hasHooksPath, gpgSignOff, isReadOrUnset := false, false, false
	mirrorOn := false
	for i, t := range rest {
		lt := strings.ToLower(t)
		switch t {
		case "--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l", "--unset", "--unset-all":
			isReadOrUnset = true
		}
		key, val, joined := splitConfigKey(lt)
		if key == "core.hookspath" {
			hasHooksPath = true
		}
		if key == "commit.gpgsign" {
			v := val
			if !joined && i+1 < len(rest) {
				v = strings.ToLower(rest[i+1])
			}
			if isGitFalse(v) {
				gpgSignOff = true
			}
		}
		if isMirrorKey(key) {
			v, has := val, joined
			if !joined && i+1 < len(rest) {
				v, has = strings.ToLower(rest[i+1]), true
			}
			if has && isMirrorEnable(key, v, true) {
				mirrorOn = true
			}
		}
	}
	if isReadOrUnset {
		return "", false
	}
	if hasHooksPath {
		return configHooksLaw, true
	}
	if gpgSignOff {
		return configSignLaw, true
	}
	if mirrorOn {
		return pushMirrorConfigLaw, true
	}
	return "", false
}

func (g *GitGate) inspectGitRuleFlags(sub string, rest []string) (string, bool) {
	for _, t := range rest {
		if t == "--" {
			break // end of options; the remainder are pathspecs/operands, not flags
		}
		for k := range g.rules {
			h := &g.rules[k]
			if h.sub != sub {
				continue
			}
			if h.long != "" && (t == h.long || strings.HasPrefix(t, h.long+"=")) {
				return h.law, true
			}
			if h.short != 0 && isShortCluster(t) && clusterHas(t, h.short) {
				return h.law, true
			}
		}
	}
	return "", false
}

// isUnscopedStashCreate reports whether the args AFTER `git stash` describe a
// whole-tree stash CREATE with no pathspec scoping it to the agent's own files.
// The create forms are: bare `git stash` (no subcommand → defaults to push),
// `git stash push ...`, and `git stash save ...`. Any other first word
// (list/show/pop/apply/drop/branch/clear/create/store) is a non-create stash op
// and is allowed. A create is "scoped" — and so allowed — when a `--` separator
// appears (everything after it is a pathspec) or, for `push`, when a bare
// (non-flag) operand follows, which git treats as a pathspec. `save` takes only
// a message, never a pathspec, so `git stash save ...` is always whole-tree.
// rebaseStateControlOnly reports whether a `git rebase` argv carries ONLY
// non-advancing rebase state control: `--abort` (restore the pre-rebase HEAD and
// working tree), `--quit` (drop the rebase state in place), or
// `--show-current-patch` (read the stopped-at commit). It is deliberately
// whole-argv and allow-listed, not a "contains --abort" scan: it requires at least
// one such flag AND admits nothing else on the line, so a revision operand, a
// `--continue`/`--skip`, or any other flag drops straight back to the categorical
// refusal. That is what keeps the exemption unlaunderable — there is no argv of the
// form `git rebase --abort <anything>` that rides it into a real rebase.
func rebaseStateControlOnly(rest []string) bool {
	found := false
	for _, t := range rest {
		switch {
		case t == "--abort", t == "--quit":
			found = true
		case t == "--show-current-patch", strings.HasPrefix(t, "--show-current-patch="):
			found = true
		default:
			return false
		}
	}
	return found
}

func isUnscopedStashCreate(rest []string) bool {
	// Skip leading valueless flags to find the stash subcommand word (e.g.
	// `git stash -k` / `git stash --keep-index` is still a bare create).
	i := 0
	for i < len(rest) && strings.HasPrefix(rest[i], "-") && rest[i] != "--" {
		i++
	}
	op := "push" // bare `git stash` defaults to push
	if i < len(rest) && rest[i] != "--" {
		op = rest[i]
		i++
	}
	switch op {
	case "push", "save":
		// fall through to the scoping check
	default:
		return false // list/show/pop/apply/drop/branch/clear/create/store/...
	}
	if op == "save" {
		return true // save never takes a pathspec — always whole-tree
	}
	// push: scoped iff a `--` appears OR a bare non-flag operand (a pathspec) follows.
	for ; i < len(rest); i++ {
		t := rest[i]
		if t == "--" {
			return false // explicit pathspec separator → scoped
		}
		if strings.HasPrefix(t, "-") {
			// a flag; `-m <msg>` consumes its value so the message is not a pathspec
			if t == "-m" || t == "--message" {
				i++
			}
			continue
		}
		return false // a bare operand on push is a pathspec → scoped
	}
	return true // push with no pathspec and no `--` → whole-tree create
}

// gitArgv returns the argument tokens of a git invocation in this segment (the
// tokens AFTER the `git` program word), or nil if the segment does not invoke git
// in command position. Leading `VAR=val` assignments and a leading `env` are
// skipped so `env FOO=bar git push -f` and `GIT_TRACE=1 git push -f` are still
// recognized. A wrapper script (`mygit`, `hub`) or alias is intentionally NOT
// recognized (documented non-goal — those remain the git hooks' floor).
func gitArgv(seg []string) []string {
	i := skipEnvPrefix(seg)
	if i >= len(seg) || !isGitProgram(seg[i]) {
		return nil
	}
	return seg[i+1:]
}

// gitArgvViaXargs returns the argument tokens of a git invocation this segment runs
// UNDER `xargs` (`... | xargs -n 200 git push origin`), or nil if the segment is not
// an xargs-wrapped git call. gitArgv deliberately recognizes git only in command
// position, which leaves `xargs git ...` invisible — and xargs is not a wrapper script
// (the documented non-goal) but an argument-stream multiplier: it runs git itself,
// with operands the argv does not contain. Recognizing it only WIDENS what the
// existing rules can see, exactly like the unwrap pass; a mis-skipped xargs flag
// yields nil (a miss), never a wrong deny.
func gitArgvViaXargs(seg []string) []string {
	i := skipEnvPrefix(seg)
	if i >= len(seg) || programBasename(seg[i]) != "xargs" {
		return nil
	}
	for i++; i < len(seg); {
		t := seg[i]
		if !strings.HasPrefix(t, "-") {
			break
		}
		switch t {
		case "-n", "-L", "-P", "-I", "-i", "-s", "-E", "-d", "-a":
			i += 2 // xargs option AND its separate value
		default:
			i++ // a valueless flag, or a joined --max-args=200
		}
	}
	if i >= len(seg) || !isGitProgram(seg[i]) {
		return nil
	}
	return seg[i+1:]
}

// isMirrorKey reports whether a lowercased config key is `remote.<name>.mirror` — the
// key that makes a plain push behave as `git push --mirror` (git-config(1)). The name
// may itself contain dots (`remote.my.fleet.mirror`), so the match is on the prefix and
// suffix with a non-empty name between, not on a fixed token count.
func isMirrorKey(key string) bool {
	const pre, suf = "remote.", ".mirror"
	return strings.HasPrefix(key, pre) && strings.HasSuffix(key, suf) && len(key) > len(pre)+len(suf)
}

// isMirrorEnable reports whether a config token ENABLES remote.<name>.mirror. Only an
// explicit git-false value (`=false`/`=off`/`=0`) is treated as harmless: git reads a
// key given with no value at all as TRUE, and any other value on this key is a shape
// the gate cannot read, so it fails closed rather than guessing in favor of a push that
// deletes every remote ref absent locally.
func isMirrorEnable(key, val string, joined bool) bool {
	if !isMirrorKey(key) {
		return false
	}
	return !(joined && isGitFalse(val))
}

// countRemoteRefDeletes counts how many remote refs one `git push` argv names for
// DELETION: every `:<dst>` refspec (empty source = delete), plus, when --delete/-d is
// present, every ref operand after the remote name. It is a count of the argv only —
// what a push deletes IMPLICITLY (--prune/--mirror/remote.<n>.mirror) is unbounded and
// is refused by its own rule rather than counted here.
func countRemoteRefDeletes(rest []string) int {
	deleteMode, operands, colonRefs := false, 0, 0
	for i := 0; i < len(rest); i++ {
		t := rest[i]
		if t == "--" {
			continue
		}
		if strings.HasPrefix(t, "-") {
			if t == "--delete" || (isShortCluster(t) && clusterHas(t, 'd')) {
				deleteMode = true
				continue
			}
			// Value-taking push options whose value is a separate token and is NOT a
			// ref — skip it so it is not miscounted as a deletion target.
			switch t {
			case "-o", "--push-option", "--repo", "--exec", "--receive-pack":
				i++
			}
			continue
		}
		if strings.HasPrefix(t, ":") && len(t) > 1 {
			colonRefs++
			continue
		}
		operands++
	}
	if deleteMode && operands > 1 {
		// The first bare operand is the remote; the rest are refs to delete.
		return colonRefs + operands - 1
	}
	return colonRefs
}

// skipEnvPrefix returns the index of the first token in seg that is not a leading
// shell env assignment (NAME=val) or a leading `env` word — i.e. where the actual
// command-position program token begins. Shared by gitArgv and dashCStrings, which
// both need to look past `VAR=val ... env` before testing the program word.
func skipEnvPrefix(seg []string) int {
	i := 0
	for i < len(seg) && (isAssign(seg[i]) || seg[i] == "env") {
		i++
	}
	return i
}

// isGitProgram reports whether a token names the git program in command position:
// its basename (after the last / or \), lowercased and with a trailing .exe
// stripped, is exactly "git". So `git`, `/usr/bin/git`, `C:\Program Files\Git\git.exe`,
// and `GIT` all match; `mygit`, `git-secret`, `legitimate` do not.
func isGitProgram(tok string) bool {
	return programBasename(tok) == "git"
}

// programBasename normalizes a command-position token to its lowercased basename
// with a trailing .exe stripped (after the last / or \). Shared by isGitProgram
// and isShellProgram.
func programBasename(tok string) string {
	return shelltoken.ProgramBasename(tok)
}

// isAssign reports whether a token is a leading shell env assignment (NAME=...,
// NAME a valid shell identifier). These precede the command word and must be
// skipped to find it.
func isAssign(t string) bool {
	return shelltoken.IsAssign(t)
}

// isShortCluster reports whether a token is a single-dash short-flag cluster
// (`-f`, `-am`, `-fq`), distinct from a `--long` flag or a bare `-`/`--`.
func isShortCluster(t string) bool { return shelltoken.IsShortCluster(t) }

// clusterHas reports whether a short-flag cluster contains the letter ch
// (case-sensitive), scanning the cluster up to an attached `=value`.
func clusterHas(token string, ch byte) bool {
	return shelltoken.ClusterHas(token, ch)
}

// splitConfigKey splits a `git config` token into its key and (joined) value:
// `commit.gpgsign=false` -> ("commit.gpgsign", "false", true); a bare
// `commit.gpgsign` -> ("commit.gpgsign", "", false), where the value is the next
// operand. Used to recognize a persistent guard-disabling config write.
func splitConfigKey(tok string) (key, val string, joined bool) {
	if eq := strings.IndexByte(tok, '='); eq >= 0 {
		return tok[:eq], tok[eq+1:], true
	}
	return tok, "", false
}

// isGitFalse reports whether v is one of git's boolean-false spellings
// (false/no/off/0), used to detect a `commit.gpgsign false` signing-disable.
func isGitFalse(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "false", "no", "off", "0":
		return true
	}
	return false
}

// Default is the registered instance.
var Default = New()

func init() {
	// Operator opt-out: FAK_GITGATE=off leaves the rung unregistered, so it Defers
	// by absence — the escape hatch for an adopter whose git policy differs from
	// this repo's trunk discipline (mirrors the FLEET_*_GUARD=off hook escapes).
	if strings.EqualFold(os.Getenv("FAK_GITGATE"), "off") {
		return
	}
	// Rank 35: after plancfi (25) / ifc-sink (30), before shipgate (40) and the
	// rank-100 authoritative monitor. Rank only orders WORK — the kernel folds the
	// chain by abi.FoldRank, so a Deny here (foldRank 100) wins over any downstream
	// Allow regardless, and a Defer (foldRank 1) never weakens another rung.
	abi.RegisterAdjudicator(35, Default)
	abi.RegisterCapability("gitgate.v1")
}
