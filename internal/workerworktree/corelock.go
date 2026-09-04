package workerworktree

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/corelockgate"
	"github.com/anthony-chaudhary/fak/internal/corelocks"
)

// THE HOLE THIS CLOSES (#5392)
//
// The hard-self core lock (CORE_SELF_MODIFY: internal/adjudicator/**,
// internal/abi/**, internal/corelocks/** — see internal/corelocks' declarative
// taxonomy) used to be enforced in exactly ONE place: internal/safecommit, i.e.
// the `fak commit` path. But `fak commit` REFUSES a detached HEAD (OFF_TRUNK), so
// the sanctioned per-worker worktree — the path CLAUDE.md blesses for build
// isolation (#1334 / epic #3165) — could never take it, and Land had no core-lock
// question of its own. A kernel edit therefore reached the trunk through Land with
// no witness ever demanded, and the ABSENCE of a refusal read as clearance.
//
// Registering the gate in the internal/hooks pre-commit chain would NOT have closed
// it: Land's default race-free path (landIsolated, #3547/#3619) builds the commit
// with `git commit-tree` + `update-ref`, plumbing that runs no git hook at all.
// The question has to be asked HERE, in the lander, before anything is applied.
//
// The check itself is NOT re-implemented here. corelockgate.CheckCoreLockHardSelf
// is the single owner of the classification and of the confirm/refute/abstain
// witness semantics, so the two paths cannot drift into two policies; this file only
// decides WHEN Land asks and WHERE the witness claim comes from.
//
// That shared check used to live in internal/safecommit, and this file imported it
// from there. safecommit is tier-2 mechanism and this lander is a tier-1 foundation
// leaf, so the import was an upward edge the layered-DAG gate refuses
// (internal/architest, TestNoUpwardImports). The cure was to push the shared check
// DOWN into internal/corelockgate(1) — never to duplicate it here, and never to
// relax the tier table.

// ReasonCoreSelfModify is the structured refusal token Land stamps on Result.Reason
// when an unwitnessed hard-self core-lock pathset tries to land. It is read straight
// off the corelocks taxonomy that DECLARES the class, which is the same token
// safecommit.ReasonCoreSelfModify emits on the `fak commit` path, so an operator
// (and a log grep) sees ONE reason class for one policy across both paths.
const ReasonCoreSelfModify = corelocks.ReasonCoreSelfModify

// CoreLockWitnessTrailer is the commit-message trailer that carries a maintenance
// witness claim into a land, e.g.
//
//	Core-lock-maintenance-witness: commit:0f1e2d3c4b5a
//
// A trailer is the affordance the DISPATCHED land needs: `fak dispatch tick` calls
// Land in-process with no message file of its own (the message is derived from the
// worker's own worktree tip), so a worker that legitimately maintains a core-locked
// surface has no flag to pass — it writes the claim into the commit message it was
// already authoring. The claim is then resolved against independent evidence
// exactly as the `fak commit` flag is: writing the trailer asserts nothing on its
// own, and an unresolvable claim still refuses.
const CoreLockWitnessTrailer = "Core-lock-maintenance-witness"

// coreLockLandRemedy is the land-path half of the shared refusal detail: the two
// ways to supply the witness HERE. The `fak commit` remedy names a flag this
// command does not have, which is why the remedy is a parameter of the shared
// check rather than baked into it.
const coreLockLandRemedy = "Re-land with `fak worktree worker land --core-lock-maintenance-witness <claim>`, or carry a `" +
	CoreLockWitnessTrailer + ": <claim>` trailer in the worktree commit message, after independent read-back confirms the edit. " +
	"A file this land ADDS is not tracked on the trunk yet, so committed:<path> is refuted for it — name it with changed:<path>, which confirms only for a path this land actually carries."

// LandOption is an additive Land setting. Land takes them variadically so every
// existing caller (the dispatch land seam, the CLI, the soak tests) compiles and
// behaves unchanged, and only a caller that has something extra to say passes one.
type LandOption func(*landConfig)

type landConfig struct {
	// coreLockWitness is the claim supplied out-of-band (the CLI flag). It wins
	// over the commit-message trailer when both are present.
	coreLockWitness string
	recoveryRemote  string
	requireRemote   bool
	progress        func(LandProgressEvent)
	now             func() time.Time
	resources       func() landResourceSample
	tracker         *landProgressTracker
	queue           *LandingQueue
}

// WithCoreLockWitness supplies the hard-self core-lock maintenance witness claim
// for this land — the land-path counterpart of `fak commit
// --core-lock-maintenance-witness`, with identical semantics: the claim is
// RESOLVED against independent evidence, and only a CONFIRMED resolution clears
// the lock. An empty claim is the same as none.
// WithRecoveryRemote publishes and independently reads back each isolated-land
// candidate on remote before trunk CAS. Required mode refuses a missing witness;
// best-effort mode preserves local landing and reports LOCAL_ONLY.
func WithRecoveryRemote(remote string, require bool) LandOption {
	return func(c *landConfig) {
		c.recoveryRemote = strings.TrimSpace(remote)
		c.requireRemote = require
	}
}

func WithCoreLockWitness(claim string) LandOption {
	return func(c *landConfig) { c.coreLockWitness = strings.TrimSpace(claim) }
}

// WithLandingQueue configures a custom LandingQueue coordinator for Land operations.
func WithLandingQueue(q *LandingQueue) LandOption {
	return func(c *landConfig) { c.queue = q }
}

func newLandConfig(opts []LandOption) landConfig {
	var cfg landConfig
	for _, o := range opts {
		if o != nil {
			o(&cfg)
		}
	}
	return cfg
}

// coreLockLandGate is the hard-self core-lock question Land asks BEFORE it applies
// or commits anything. It returns (refusal, true) when the land must be refused;
// (zero, false) when the pathset raises no hard-self lock or a supplied witness
// resolved CONFIRMED.
//
// changed is the pathset that would actually LAND — the worktree's whole
// diff-since-base, not the lease-scoped subset, because the default isolated path
// stages the WHOLE captured patch into its throwaway index and `paths` only scopes
// the readback/sync. Classifying the lease subset would let a kernel edit ride in
// on an out-of-lane hunk.
func coreLockLandGate(root, nameOnly, diff, msgFile string, cfg landConfig, git GitRunner) (Result, bool) {
	changed := landChangedPaths(nameOnly, diff)
	claim := cfg.coreLockWitness
	if claim == "" {
		claim = coreLockWitnessFromMsgFile(msgFile)
	}
	detail, fired := corelockgate.CheckCoreLockHardSelf(context.Background(), corelockgate.CoreLockCheck{
		Dir:     root,
		Run:     witnessGitRunner(git),
		Changed: changed,
		Witness: claim,
		Remedy:  coreLockLandRemedy,
	})
	if !fired {
		return Result{}, false
	}
	return Result{
		OK: false, Applied: false, Committed: false,
		Reason: ReasonCoreSelfModify + ": " + detail,
	}, true
}

// witnessGitRunner adapts this package's GitRunner to the corelockgate/witness
// runner contract, so the witness resolves its evidence through whatever git seam
// the caller injected (the real binary in production, the fake in a test) instead
// of opening a second, unmockable one. A non-zero git exit is a code, never an
// err — the same contract both sides already use.
func witnessGitRunner(git GitRunner) corelockgate.Runner {
	return func(_ context.Context, dir string, args ...string) (string, int, error) {
		rc, out := run(git, dir, args)
		return out, rc, nil
	}
}

// landChangedPaths is the repo-relative pathset the land would carry. Land reads
// `git diff --name-only <base>` once and hands the result here; it is authoritative
// (git quotes/escapes correctly). When that call failed or returned nothing, the
// captured diff's own `diff --git a/X b/Y` headers are parsed instead, so a git
// hiccup can never hand the core-lock gate an EMPTY set and silently turn the
// refusal off.
func landChangedPaths(nameOnly, diff string) []string {
	if out := nonEmptyLines(nameOnly); len(out) > 0 {
		return out
	}
	return diffHeaderPaths(diff)
}

// diffHeaderPaths extracts both sides of every `diff --git a/OLD b/NEW` header in a
// unified diff. Both sides are kept because a rename touches both, and a rename OUT
// of a locked surface is as much a hard-self edit as a rename in.
func diffHeaderPaths(diff string) []string {
	seen := map[string]bool{}
	var out []string
	for _, line := range strings.Split(strings.ReplaceAll(diff, "\r\n", "\n"), "\n") {
		if !strings.HasPrefix(line, "diff --git ") {
			continue
		}
		for _, tok := range strings.Fields(strings.TrimPrefix(line, "diff --git ")) {
			p := strings.Trim(tok, `"`)
			if strings.HasPrefix(p, "a/") || strings.HasPrefix(p, "b/") {
				p = p[2:]
			}
			if p == "" || p == "/dev/null" || seen[p] {
				continue
			}
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

// coreLockWitnessFromMsgFile reads the CoreLockWitnessTrailer claim out of the
// resolved commit message, or "" when the file is absent/unreadable or carries no
// trailer. Best-effort by design: an unreadable message yields NO claim, which
// keeps the gate closed rather than opening it.
func coreLockWitnessFromMsgFile(msgFile string) string {
	if strings.TrimSpace(msgFile) == "" {
		return ""
	}
	raw, err := os.ReadFile(msgFile)
	if err != nil {
		return ""
	}
	return coreLockWitnessFromMessage(string(raw))
}

// coreLockWitnessFromMessage extracts the trailer claim from a commit message
// body. The LAST occurrence wins (a trailer block is read bottom-up), the key match
// is case-insensitive, and a key that is not at the start of its own line is
// ignored so prose quoting the trailer name cannot manufacture a witness.
func coreLockWitnessFromMessage(body string) string {
	claim := ""
	prefix := strings.ToLower(CoreLockWitnessTrailer) + ":"
	for _, line := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(strings.ToLower(trimmed), prefix) {
			continue
		}
		if v := strings.TrimSpace(trimmed[len(prefix):]); v != "" {
			claim = v
		}
	}
	return claim
}

func nonEmptyLines(s string) []string {
	var out []string
	for _, line := range strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n") {
		if p := strings.TrimSpace(line); p != "" {
			out = append(out, p)
		}
	}
	return out
}
