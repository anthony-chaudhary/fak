package safecommit

import (
	"context"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/corelockaudit"
	"github.com/anthony-chaudhary/fak/internal/corelockgate"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// THE CHECK ITSELF LIVES ONE LAYER DOWN, in internal/corelockgate (#5392).
//
// It was born here because `fak commit` was the only path that could ask it. The
// sanctioned detached worker-worktree land (internal/workerworktree) must ask the
// SAME question — `fak commit` refuses a detached HEAD, so that lander had no
// core-lock question of its own — but it is a foundation (tier-1) leaf and may not
// import this mechanism (tier-2) package: internal/architest's TestNoUpwardImports
// refuses the upward edge, and duplicating the classifier would give the fleet two
// policies wearing one name. So the shared question was pushed DOWN to a foundation
// leaf that both callers import, and what remains here is the re-export the existing
// `fak commit` callers bind to. Behaviour is unchanged; ownership moved.

// CoreLockCheck is one hard-self core-lock decision request. It is a true Go type
// ALIAS of corelockgate.CoreLockCheck (not a distinct wrapper type), so every
// existing struct literal, field name, and call site keeps compiling and both
// callers really do pass the SAME type into the SAME check.
type CoreLockCheck = corelockgate.CoreLockCheck

// CheckCoreLockHardSelf reports the hard-self core-lock refusal for a changed
// pathset, or ("", false) when the set raises no hard-self lock or when the offered
// witness resolves CONFIRMED. It performs no staging and no writes, so a caller can
// run it before anything touches an index, a worktree, or HEAD.
//
// FAIL-CLOSED on the lock, fail-open on the taxonomy: a missing, refuted, merely
// abstaining, or unresolvable witness keeps the refusal (an abstain is not
// clearance), while an unreadable/malformed taxonomy classifies nothing and
// therefore refuses nothing.
//
// The only thing this wrapper adds to corelockgate's check is safecommit's own
// default git seam: a caller here that injects no Runner still gets realRunner —
// the same merged-stderr, GIT_OPTIONAL_LOCKS=0 seam the rest of the commit path
// runs through — rather than the resolver's plain one.
func CheckCoreLockHardSelf(ctx context.Context, c CoreLockCheck) (detail string, fired bool) {
	if c.Run == nil {
		c.Run = corelockgate.Runner(realRunner)
	}
	return corelockgate.CheckCoreLockHardSelf(ctx, c)
}

// CoreLockRemedyCommit is the `fak commit` way to supply the maintenance witness.
// It is the default remedy, so a caller that names none still prints a real cure.
const CoreLockRemedyCommit = corelockgate.CoreLockRemedyCommit

// checkCoreLockHardSelf refuses a hard-self pathset unless the caller supplies a
// maintenance witness claim that the resolver independently confirms. This gate
// runs before staging, so an unwitnessed hard-self edit leaves the index and HEAD
// untouched. It is the Options-shaped face of CheckCoreLockHardSelf.
func checkCoreLockHardSelf(ctx context.Context, run Runner, opts Options, changedPaths []string) (detail string, fired bool) {
	return CheckCoreLockHardSelf(ctx, CoreLockCheck{
		Dir:      opts.Dir,
		Run:      corelockgate.Runner(run),
		Resolver: opts.CoreLockWitnessResolver,
		Changed:  changedPaths,
		Witness:  opts.CoreLockMaintenanceWitness,
		Remedy:   CoreLockRemedyCommit,
	})
}

// coreLockHardSelfFinding is the commit path's binding to the shared classifier: it
// is what tells CommitWith WHICH paths the lock named, so the accepted commit can
// record them on the maintenance decision.
func coreLockHardSelfFinding(paths []string) (corelockaudit.Finding, bool) {
	return corelockgate.HardSelfFinding(paths)
}

// statusChangedPaths extracts repo paths from `git status --porcelain -- <paths>`
// output. It lets a broad requested pathspec such as "internal" still trip the
// hard-self guard when the actual changed file is under a locked surface.
func statusChangedPaths(status string) []string {
	seen := map[string]bool{}
	for _, line := range strings.Split(strings.ReplaceAll(status, "\r\n", "\n"), "\n") {
		p := statusLinePath(line)
		if p != "" {
			seen[p] = true
		}
	}
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func statusLinePath(line string) string {
	if len(line) < 4 {
		return ""
	}
	p := strings.TrimSpace(line[3:])
	if p == "" {
		return ""
	}
	if strings.Contains(p, " -> ") {
		parts := strings.Split(p, " -> ")
		p = strings.TrimSpace(parts[len(parts)-1])
	}
	return strings.Trim(p, `"`)
}

func recordCoreLockMaintenance(ctx context.Context, opts Options, res Result) {
	claim := strings.TrimSpace(res.CoreLockWitness)
	if claim == "" {
		claim = strings.TrimSpace(opts.CoreLockMaintenanceWitness)
	}
	if opts.Recorder == nil || !res.Verified || res.SHA == "" || claim == "" {
		return
	}
	paths := append([]string(nil), res.CoreLockPaths...)
	if len(paths) == 0 {
		f, ok := coreLockHardSelfFinding(res.Paths)
		if !ok {
			return
		}
		paths = append(paths, f.Paths...)
	}
	d := witness.Decision{
		Op:                "corelock-maintenance",
		Verdict:           witness.VerdictAllow,
		ReasonClass:       ReasonCoreSelfModify,
		Tree:              paths,
		PathspecAssertion: "hard-self-maintenance-witness-confirmed",
		Witness:           claim,
	}
	_ = opts.Recorder.AppendDecision(ctx, res.SHA, d)
}
