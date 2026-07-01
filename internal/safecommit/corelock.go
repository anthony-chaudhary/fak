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

// checkCoreLockHardSelf refuses a hard-self pathset unless the caller supplies a
// maintenance witness claim that the resolver independently confirms. This gate
// runs before staging, so an unwitnessed hard-self edit leaves the index and HEAD
// untouched.
func checkCoreLockHardSelf(ctx context.Context, run Runner, opts Options, changedPaths []string) (detail string, fired bool) {
	f, ok := coreLockHardSelfFinding(changedPaths)
	if !ok {
		return "", false
	}
	claim := strings.TrimSpace(opts.CoreLockMaintenanceWitness)
	if claim == "" {
		return coreLockHardSelfDetail(f, "missing maintenance witness"), true
	}
	resolver := opts.CoreLockWitnessResolver
	if resolver == nil {
		resolver = witness.NewWithRunner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			return run(ctx, dir, args...)
		}, opts.Dir)
	}
	outcome := resolver.Resolve(ctx, nil, claim)
	if outcome == abi.WitnessConfirmed {
		return "", false
	}
	return coreLockHardSelfDetail(f, fmt.Sprintf("maintenance witness %q resolved %s", claim, coreLockWitnessOutcome(outcome))), true
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

func coreLockHardSelfDetail(f corelockaudit.Finding, cause string) string {
	paths := append([]string(nil), f.Paths...)
	sort.Strings(paths)
	return fmt.Sprintf(
		"hard-self core-lock path(s) require an external maintenance witness before fak commit may stage them; %s. "+
			"Use a privileged maintenance path, or rerun fak commit with --core-lock-maintenance-witness <claim> after independent read-back confirms the edit. Paths: %s",
		cause, strings.Join(paths, ", "),
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
