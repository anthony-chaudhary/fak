package safecommit

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/corelockaudit"
	"github.com/anthony-chaudhary/fak/internal/corelocks"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// CoreLockCheck is one hard-self core-lock decision request: the changed pathset
// to classify, the maintenance witness claim offered to clear it, the evidence
// seams the resolver reads, and the caller-specific remedy named in the refusal.
//
// It exists so EVERY commit path that must honour the lock asks the SAME question
// of the SAME classifier with the SAME witness semantics. `fak commit` asks
// through Options (checkCoreLockHardSelf); the sanctioned detached worker-worktree
// land asks directly (internal/workerworktree, #5392) — that path is the one
// CLAUDE.md blesses for build isolation and it used to reach the trunk with no
// core-lock question asked at all. Only Remedy differs between the two callers,
// because only the way to SUPPLY the witness differs between the two commands;
// the locked path set and the confirm/refute/abstain semantics are shared.
type CoreLockCheck struct {
	// Dir is the repo the witness resolver reads its evidence from.
	Dir string
	// Run is the git seam the default resolver runs through. Nil uses the real
	// git binary (realRunner), so a caller with no injected runner still gets a
	// real, independently-checked witness.
	Run Runner
	// Resolver overrides the default git-backed witness resolver (tests, or an
	// alternate evidence source). Nil builds the real one over Run/Dir.
	Resolver abi.WitnessResolver
	// Changed is the changed repo-relative pathset to classify.
	Changed []string
	// Witness is the maintenance witness claim offered to clear a hard-self
	// pathset. Empty means none was offered, which is a refusal.
	Witness string
	// Remedy is the caller's own "how to supply that witness" sentence, quoted in
	// the refusal detail. Empty falls back to the `fak commit` remedy.
	Remedy string
}

// CheckCoreLockHardSelf reports the hard-self core-lock refusal for a changed
// pathset, or ("", false) when the set raises no hard-self lock or when the
// offered witness resolves CONFIRMED. It performs no staging and no writes, so a
// caller can run it before anything touches an index, a worktree, or HEAD.
//
// FAIL-CLOSED on the lock, fail-open on the taxonomy: a missing, refuted, or
// merely abstaining witness keeps the refusal (an abstain is not clearance), while
// an unreadable/malformed taxonomy classifies nothing and therefore refuses
// nothing — the same posture the `fak commit` path has always had.
func CheckCoreLockHardSelf(ctx context.Context, c CoreLockCheck) (detail string, fired bool) {
	f, ok := coreLockHardSelfFinding(c.Changed)
	if !ok {
		return "", false
	}
	claim := strings.TrimSpace(c.Witness)
	if claim == "" {
		return coreLockHardSelfDetail(f, "missing maintenance witness", c.Remedy), true
	}
	resolver := c.Resolver
	if resolver == nil {
		run := c.Run
		if run == nil {
			run = realRunner
		}
		resolver = witness.NewWithRunner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			return run(ctx, dir, args...)
		}, c.Dir)
	}
	outcome := resolver.Resolve(ctx, nil, claim)
	if outcome == abi.WitnessConfirmed {
		return "", false
	}
	return coreLockHardSelfDetail(f, fmt.Sprintf("maintenance witness %q resolved %s", claim, coreLockWitnessOutcome(outcome)), c.Remedy), true
}

// CoreLockRemedyCommit is the `fak commit` way to supply the maintenance witness.
// It is the default remedy, so a caller that names none still prints a real cure.
const CoreLockRemedyCommit = "Use a privileged maintenance path, or rerun fak commit with --core-lock-maintenance-witness <claim> after independent read-back confirms the edit."

// checkCoreLockHardSelf refuses a hard-self pathset unless the caller supplies a
// maintenance witness claim that the resolver independently confirms. This gate
// runs before staging, so an unwitnessed hard-self edit leaves the index and HEAD
// untouched. It is the Options-shaped face of CheckCoreLockHardSelf.
func checkCoreLockHardSelf(ctx context.Context, run Runner, opts Options, changedPaths []string) (detail string, fired bool) {
	return CheckCoreLockHardSelf(ctx, CoreLockCheck{
		Dir:      opts.Dir,
		Run:      run,
		Resolver: opts.CoreLockWitnessResolver,
		Changed:  changedPaths,
		Witness:  opts.CoreLockMaintenanceWitness,
		Remedy:   CoreLockRemedyCommit,
	})
}

func coreLockHardSelfFinding(paths []string) (corelockaudit.Finding, bool) {
	tax, err := corelocks.LoadFixture()
	if err != nil {
		return corelockaudit.Finding{}, false
	}
	rep := corelockaudit.Audit(tax, paths)
	for _, f := range rep.Findings {
		if f.Class == corelocks.ClassHardSelf && f.ReasonToken == corelocks.ReasonCoreSelfModify {
			return f, true
		}
	}
	return corelockaudit.Finding{}, false
}

func coreLockHardSelfDetail(f corelockaudit.Finding, cause, remedy string) string {
	paths := append([]string(nil), f.Paths...)
	sort.Strings(paths)
	if strings.TrimSpace(remedy) == "" {
		remedy = CoreLockRemedyCommit
	}
	return fmt.Sprintf(
		"hard-self core-lock path(s) require an external maintenance witness before this change may land; %s. %s Paths: %s",
		cause, remedy, strings.Join(paths, ", "),
	)
}

func coreLockWitnessOutcome(outcome abi.WitnessOutcome) string {
	switch outcome {
	case abi.WitnessConfirmed:
		return "confirmed"
	case abi.WitnessRefuted:
		return "refuted"
	default:
		return "abstain"
	}
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
