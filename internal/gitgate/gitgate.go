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

// defaultHazards is the repo's structurally-decidable trunk discipline, encoded
// once. Every entry maps 1:1 to a documented law (AGENTS.md / CLAUDE.md) that today
// only a doc sentence or an after-the-fact git hook enforces.
var defaultHazards = []hazard{
	// Never force-push the shared trunk (AGENTS.md). Closes the gap that the
	// named-tool `git_push` deny leaves for a Bash command="git push --force".
	{sub: "push", long: "--force", short: 'f', law: "force-push refused: never force-push the shared trunk (AGENTS.md). Re-run `git push` WITHOUT --force/-f."},
	{sub: "push", long: "--force-with-lease", law: "force-push refused: never force-push the shared trunk (AGENTS.md). Re-run `git push` WITHOUT --force-with-lease."},
	// Never skip the guards / signing.
	{sub: "push", long: "--no-verify", law: "skip-hooks refused: never bypass the pre-push guards (push --no-verify). Push with the hooks enabled."},
	// Do not delete a remote ref from an agent.
	{sub: "push", long: "--delete", short: 'd', law: "remote-ref delete refused: do not delete a remote branch from an agent (push --delete/-d)."},
	// Never amend in a shared tree — HEAD moves between peers (CLAUDE.md).
	{sub: "commit", long: "--amend", law: "amend refused: never amend in the shared tree — HEAD moves between peers (CLAUDE.md). Make a NEW commit instead."},
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
	{sub: "rebase", long: "--interactive", short: 'i', law: "history-rewrite refused: no interactive rebase on the shared trunk (rebase -i/--interactive)."},
	// Never --autostash in the shared tree: an aborted/conflicted rebase pops the
	// stash back as a working-tree blob, dumping a peer's in-flight WIP into your
	// tree and leaving a dangling `autostash` stash (CLAUDE.md / [[fak-shared-tree-high-churn-commit]]).
	{sub: "rebase", long: "--autostash", law: "autostash refused: never `rebase --autostash` in the shared tree — an abort pops the stash into your working tree, sweeping a peer's WIP (CLAUDE.md). Reach a clean tree first (stash explicit paths or commit your work), THEN `git rebase` with no autostash."},
	{sub: "pull", long: "--autostash", law: "autostash refused: never `pull --rebase --autostash` in the shared tree — a conflict abort pops the stash into your working tree, sweeping a peer's WIP (CLAUDE.md). Reach a clean tree first, then `git fetch` + `git rebase origin/main` with no autostash."},
	// Never destroy the shared working tree: `reset --hard` discards tracked-file
	// changes and `clean -f` deletes untracked files — both sweep a peer's WIP
	// (AGENTS.md destructive-op list). A `--soft`/`--mixed` reset and a `clean -n`
	// dry-run are non-destructive and do not match.
	{sub: "reset", long: "--hard", law: "reset-hard refused: `git reset --hard` discards every working-tree change to tracked files — incl. a peer's unstaged WIP in the shared tree (AGENTS.md destructive-op list). Reconcile in place, or scope your undo: `git restore -- <your-paths>`."},
	{sub: "clean", long: "--force", short: 'f', law: "clean-force refused: `git clean -f` deletes untracked files — incl. a peer's new files and your own uncommitted work in the shared tree (AGENTS.md). Remove specific files explicitly; never whole-tree clean a shared checkout."},
	// Never open a feature branch — the argv-decidable half of OFF_TRUNK (AGENTS.md).
	{sub: "checkout", short: 'b', law: offTrunkBranchLaw},
	{sub: "checkout", short: 'B', law: offTrunkBranchLaw},
	{sub: "switch", short: 'c', law: offTrunkBranchLaw},
	{sub: "switch", short: 'C', law: offTrunkBranchLaw},
	// `git push --mirror` overwrites EVERY remote ref (and deletes remote refs
	// absent locally) — catastrophic on a shared remote (a superset of force-push).
	{sub: "push", long: "--mirror", law: "push-mirror refused: `git push --mirror` overwrites EVERY remote ref and deletes remote refs absent locally — catastrophic on a shared remote. Push specific refs without --mirror."},
}

const dotAddLaw = "commit-by-explicit-path: `git add .` stages the whole tree (AGENTS.md). Add explicit paths instead."

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
// target main?) and stays deferred — only the unconditional CREATE forms fire here.
const offTrunkBranchLaw = "off-trunk refused: `git checkout -b` / `git switch -c` / `git worktree add` opens a feature branch or worktree — work directly on the trunk (`main`); never branch or spin a worktree in this repo (AGENTS.md OFF_TRUNK)."

// historyRewriteLaw fires on a whole-history rewrite subcommand (`git filter-branch`,
// `git filter-repo`). These rewrite every commit on the shared trunk — the same class
// of forbidden act as a force-push or an interactive rebase, just applied wholesale.
const historyRewriteLaw = "history-rewrite refused: `git filter-branch` / `git filter-repo` rewrites shared history — forbidden on the trunk (AGENTS.md: never rewrite or force-push shared history). Make a new commit instead."

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
		}
		g.recordRefusal(ctx, "gitgate", abi.ReasonName(v.Reason), []string{"shell", "-c", cmd}, nil)
		return v
	}
	return deferVerdict()
}

func deferVerdict() abi.Verdict { return abi.Verdict{Kind: abi.VerdictDefer, By: "gitgate"} }

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
	// The unwrap pass yields cmd itself PLUS every command string the shell grammar
	// wraps around a git call the flat tokenizer cannot see: a `$(...)` / backtick
	// command substitution and the `-c` string of a `bash -c '...'` / `sh -c '...'`
	// sub-shell, recursively. Each recovered string is then tokenized + inspected by
	// the EXACT same defaultHazards rules — the pass only widens what the existing
	// rules can SEE, it adds no new hazard logic and changes no verdict. (Pipes,
	// `&&`/`||`/`;`, newline already segment correctly in tokenizeSegments.)
	for _, src := range unwrapShellSources(cmd) {
		for _, seg := range tokenizeSegments(src) {
			argv := gitArgv(seg)
			if argv == nil {
				continue // this segment's command word is not git
			}
			if law, ok := g.inspectGit(argv); ok {
				return law, true
			}
		}
	}
	return "", false
}

// inspectGit walks the args of a git invocation (the tokens AFTER the `git`
// program word): it skips the value-bearing global options to locate the
// subcommand, catches a `-c core.hooksPath=...` skip-hooks override along the way,
// then matches the subcommand's flags against the hazard table.
func (g *GitGate) inspectGit(args []string) (string, bool) {
	i := 0
	for i < len(args) {
		a := args[i]
		if !strings.HasPrefix(a, "-") {
			break // the subcommand
		}
		switch {
		case a == "-c" || a == "-C" || a == "--git-dir" || a == "--work-tree" || a == "--namespace" || a == "--exec-path":
			val := ""
			if i+1 < len(args) {
				val = args[i+1]
			}
			if a == "-c" && strings.Contains(strings.ToLower(val), "core.hookspath") {
				return "skip-hooks refused: `git -c core.hooksPath=...` disables hooks for this invocation.", true
			}
			i += 2 // consume the option AND its value
		case strings.HasPrefix(a, "--") && strings.Contains(a, "="):
			if strings.Contains(strings.ToLower(a), "core.hookspath") {
				return "skip-hooks refused: a core.hooksPath override disables hooks for this invocation.", true
			}
			i++ // a joined long global, e.g. --git-dir=...
		default:
			i++ // a valueless global flag (--no-pager, -p, --bare, ...)
		}
	}
	if i >= len(args) {
		return "", false // no subcommand (e.g. `git --version`, `git -C x`)
	}
	sub := args[i]
	rest := args[i+1:]

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
	i := 0
	for i < len(seg) && (isAssign(seg[i]) || seg[i] == "env") {
		i++
	}
	if i >= len(seg) || !isGitProgram(seg[i]) {
		return nil
	}
	return seg[i+1:]
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
	b := tok
	if k := strings.LastIndexAny(b, `/\`); k >= 0 {
		b = b[k+1:]
	}
	b = strings.ToLower(b)
	return strings.TrimSuffix(b, ".exe")
}

// isAssign reports whether a token is a leading shell env assignment (NAME=...,
// NAME a valid shell identifier). These precede the command word and must be
// skipped to find it.
func isAssign(t string) bool {
	eq := strings.IndexByte(t, '=')
	if eq <= 0 {
		return false
	}
	for i := 0; i < eq; i++ {
		ch := t[i]
		ok := ch == '_' ||
			(ch >= 'A' && ch <= 'Z') ||
			(ch >= 'a' && ch <= 'z') ||
			(i > 0 && ch >= '0' && ch <= '9')
		if !ok {
			return false
		}
	}
	return true
}

// isShortCluster reports whether a token is a single-dash short-flag cluster
// (`-f`, `-am`, `-fq`), distinct from a `--long` flag or a bare `-`/`--`.
func isShortCluster(t string) bool { return len(t) >= 2 && t[0] == '-' && t[1] != '-' }

// clusterHas reports whether a short-flag cluster contains the letter ch
// (case-sensitive), scanning the cluster up to an attached `=value`.
func clusterHas(token string, ch byte) bool {
	for i := 1; i < len(token); i++ {
		if token[i] == '=' {
			break
		}
		if token[i] == ch {
			return true
		}
	}
	return false
}

// shellCommand extracts the shell command string from a tool call's args,
// resolving the args Ref the same way the monitor does and reading the `command`
// then `cmd` scalar key (the two conventions across shell tools). Returns "" when
// there is no command arg — the not-a-shell-call case the rung Defers on.
func shellCommand(ctx context.Context, c *abi.ToolCall) string {
	b := refBytes(ctx, c.Args)
	if len(b) == 0 {
		return ""
	}
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return ""
	}
	if s, ok := m["command"].(string); ok {
		return s
	}
	if s, ok := m["cmd"].(string); ok {
		return s
	}
	return ""
}

// refBytes materializes a Ref's bytes (inline directly, otherwise via the active
// resolver), mirroring internal/adjudicator's decodeArgs read path.
func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

// tokenizeSegments splits a shell command into segments at unquoted command
// separators (`;` `|` `&` newline and the subshell-grouping parens `(` `)` — the
// `&&` / `||` chains end in `&` / `|`, so a doubled separator just yields an empty
// segment that is dropped), and tokenizes each segment into words at unquoted
// whitespace and redirection operators (`<` `>`), with surrounding single/double
// quotes stripped. It is a deliberately small shell-ish lexer, NOT a shell parser:
// it does not interpret backslash escapes, `$(...)`/backtick substitution, or
// variable expansion. Those launder a git op past this floor and remain the git
// hooks' job (documented non-goal). Quote stripping is what keeps a flag mentioned
// INSIDE a quoted operand — `git commit -m "always use git push --force"` — from
// being read as a flag: the message is one de-quoted operand token, not `--force`.
func tokenizeSegments(cmd string) [][]string {
	var segs [][]string
	var cur []string
	var tok strings.Builder
	var quote byte // 0, '\'' or '"'

	flushTok := func() {
		if tok.Len() > 0 {
			cur = append(cur, tok.String())
			tok.Reset()
		}
	}
	flushSeg := func() {
		flushTok()
		if len(cur) > 0 {
			segs = append(segs, cur)
			cur = nil
		}
	}
	for i := 0; i < len(cmd); i++ {
		ch := cmd[i]
		if quote != 0 {
			if ch == quote {
				quote = 0
			} else {
				tok.WriteByte(ch)
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case ' ', '\t', '\r', '<', '>':
			flushTok()
		case ';', '|', '&', '\n', '(', ')':
			flushSeg()
		default:
			tok.WriteByte(ch)
		}
	}
	flushSeg()
	return segs
}

// maxUnwrapDepth bounds the recursion of unwrapShellSources so a pathological input
// (deeply nested `$( $( $( ... )))` or a `bash -c` of a `bash -c` of a ...) cannot make
// the pure decide path blow the stack. 8 levels is far past any real laundering chain.
const maxUnwrapDepth = 8

// maxUnwrapSources bounds the TOTAL number of command strings the unwrap pass yields, so
// a string packed with many substitutions cannot turn one classify() into unbounded work.
const maxUnwrapSources = 256

// unwrapShellSources returns cmd PLUS every command string the shell grammar wraps around a
// git call that the flat tokenizer (tokenizeSegments) cannot see on its own: the body of a
// `$(...)` / backtick command substitution, and the `-c` string of a recognized `bash -c`
// / `sh -c` sub-shell — recursively, so a git call nested inside a `$()` inside a `bash -c`
// is recovered. Pipes / `&&` / `||` / `;` / newline already segment correctly inside
// tokenizeSegments, so they need no extra source here; the recursion only adds the
// substitution bodies and sub-shell strings.
//
// It is the recovery half of the documented honest boundary: it makes pipes, operators,
// command substitution, and `-c` strings VISIBLE to the existing rules. It deliberately
// does NOT resolve EXPANSION — `$VAR`, `alias`, and `eval` require runtime state (the
// variable's value, the alias table, the eval result) a static pre-call pass does not have,
// so `git $CMD --force` is unrecoverable here and DEGRADES to defer/opaque (never to allow):
// we simply cannot see a git call we cannot reconstruct, and the git-hooks floor +
// internal/witness remain the backstop. A malformed / unbalanced / over-deep input yields
// only the sources we could safely extract — it never silently drops cmd itself, so the
// flat-tokenizer floor is preserved as a strict subset of this pass.
func unwrapShellSources(cmd string) []string {
	out := make([]string, 0, 4)
	seen := map[string]bool{}
	var walk func(s string, depth int)
	walk = func(s string, depth int) {
		if len(out) >= maxUnwrapSources {
			return
		}
		if s = strings.TrimSpace(s); s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
		if depth >= maxUnwrapDepth {
			return
		}
		for _, sub := range commandSubstitutions(s) {
			walk(sub, depth+1)
		}
		for _, inner := range dashCStrings(s) {
			walk(inner, depth+1)
		}
	}
	walk(cmd, 0)
	return out
}

// commandSubstitutions extracts the bodies of every UNQUOTED `$()` and backtick
// command substitution in s. A `$()` is paren-depth-tracked so a nested `$(... $(...) ...)`
// yields the OUTER body (the recursion in unwrapShellSources re-extracts the inner one). A
// substitution inside SINGLE quotes is inert in the shell (no expansion happens there), so
// it is skipped; one inside DOUBLE quotes is active, so it is extracted — matching real shell
// semantics, which keeps a `$(...)` mentioned inside a single-quoted commit message from being
// read as a live call. Backticks do not nest, so the first matching backtick closes.
func commandSubstitutions(s string) []string {
	var subs []string
	var quote byte // 0, '\'' or '"' — the surrounding quote context
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if quote == '\'' {
			// Single quotes are literal: nothing expands, just find the close.
			if ch == '\'' {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'':
			if quote == 0 {
				quote = '\''
			}
		case '"':
			if quote == '"' {
				quote = 0
			} else if quote == 0 {
				quote = '"'
			}
		case '$':
			if i+1 < len(s) && s[i+1] == '(' {
				body, end, ok := extractParenBody(s, i+1)
				if ok {
					subs = append(subs, body)
					i = end // skip past the closing ')'
				}
			}
		case '`':
			if j := strings.IndexByte(s[i+1:], '`'); j >= 0 {
				subs = append(subs, s[i+1:i+1+j])
				i = i + 1 + j // skip past the closing backtick
			}
		}
	}
	return subs
}

// extractParenBody, given s[open]=='(', returns the substring between the balanced parens,
// the index of the matching ')', and whether the parens balanced. Quote-aware so a ')' inside
// a quoted operand does not close the group prematurely. Unbalanced input returns ok=false
// (the laundering degrades to opaque, not to a mis-parsed allow).
func extractParenBody(s string, open int) (body string, end int, ok bool) {
	depth := 0
	var quote byte
	for i := open; i < len(s); i++ {
		ch := s[i]
		if quote != 0 {
			if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return s[open+1 : i], i, true
			}
		}
	}
	return "", 0, false
}

// dashCStrings returns the `-c` operand string of every recognized `bash -c <str>` /
// `sh -c <str>` (also `/bin/bash`, `zsh`, `dash`) sub-shell invocation across the command
// SEGMENTS of s. It reuses tokenizeSegments (which de-quotes), so the recovered string is
// the already-unquoted program text the sub-shell would run — fed back through the unwrap
// recursion as its own program. Only the FIRST non-flag operand after `-c` is taken (the
// shell's command string); trailing operands are $0/positional args, not code.
func dashCStrings(s string) []string {
	var inner []string
	for _, seg := range tokenizeSegments(s) {
		i := 0
		for i < len(seg) && (isAssign(seg[i]) || seg[i] == "env") {
			i++
		}
		if i >= len(seg) || !isShellProgram(seg[i]) {
			continue
		}
		for j := i + 1; j < len(seg); j++ {
			t := seg[j]
			if t == "-c" {
				if j+1 < len(seg) {
					inner = append(inner, seg[j+1])
				}
				break
			}
			// A `-c` bundled in a cluster (`-lc`, `-ic`) still introduces the command
			// string as the next operand.
			if isShortCluster(t) && clusterHas(t, 'c') {
				if j+1 < len(seg) {
					inner = append(inner, seg[j+1])
				}
				break
			}
			if !strings.HasPrefix(t, "-") {
				break // a non-flag operand before -c: not a `-c` sub-shell we recognize
			}
		}
	}
	return inner
}

// isShellProgram reports whether a token names a POSIX shell in command position — the
// program whose `-c` operand is a nested program to unwrap. Mirrors isGitProgram's basename
// normalization (path + .exe stripped, lowercased).
func isShellProgram(tok string) bool {
	switch programBasename(tok) {
	case "sh", "bash", "dash", "zsh", "ksh":
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
