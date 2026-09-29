// Package worktreecheck is the IMPORTABLE vendor surface of fak's
// verify-checkout (topology-candidate) sweep.
//
// fak's owner-aware verify-checkout sweep lives in internal/workerworktree: every
// land-verify candidate (.fak-cand-validate-*) and exec-witness scratch checkout
// (fak-exec-witness-*) names itself with its creator's identity (pid + base36
// process-start millis), and a later sweep collects the ones whose creator is
// provably gone. Go's internal/ rule seals internal/workerworktree to this module,
// which is correct for the kernel's own consumers but blocks the audience the
// sweep exists to serve too: an OUT-OF-TREE (private platform/cadence) caller that
// wants to run the ONE sweep without importing internal/*.
//
// Every name below is a Go TYPE ALIAS or a thin wrapper over a symbol in
// internal/workerworktree. A value from pkg/worktreecheck is IDENTICAL (same
// underlying type) to its internal counterpart. Private code imports THIS package
// (github.com/anthony-chaudhary/fak/pkg/worktreecheck) and never
// internal/workerworktree, satisfying the core import invariant: private code
// imports ONLY fak/pkg/*. The wrappers add no behavior and hold no state; they are
// a stable, zero-cost re-export for Gate 3 of the architecture boundary.
package worktreecheck

import (
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type (
	// SweepOptions supplies the clock, the liveness probes, and the apply opt-in.
	SweepOptions = workerworktree.CandidateSweepOptions
	// SweepReport is the dry-run/apply result of a sweep. Candidates lists every
	// candidate found, kept ones included, so an operator can see why one stays.
	SweepReport = workerworktree.CandidateSweepReport
	// SweepItem is one candidate found in the sweep's parent directory.
	SweepItem = workerworktree.CandidateSweepItem
)

// TopologyCandidateParents returns the DISTINCT parents a full sweep must scan to
// cover every verify-checkout family for a repository at root: the sibling parent
// (or os.TempDir() when the workspace does not escape root) AND os.TempDir(). A
// repo with a sibling-escaping go.work materializes candidates beside root (and
// exec-witness checkouts in the sibling parent); every other repo uses the system
// temp directory for both. The system temp directory is always included so a
// mixed history is fully covered. Order is stable and dedup is case-insensitive
// on Windows.
func TopologyCandidateParents(root string) []string {
	var parents []string
	primary, err := workerworktree.TopologyCandidateParent(root)
	if err == nil && primary != "" {
		parents = append(parents, primary)
	}
	parents = append(parents, os.TempDir())
	seen := map[string]bool{}
	out := make([]string, 0, len(parents))
	for _, parent := range parents {
		key := parent
		if runtime.GOOS == "windows" {
			key = strings.ToLower(key)
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, parent)
	}
	return out
}

// Sweep runs the existing owner-aware topology-candidate sweep for one parent.
// git nil uses the package default (host git).
func Sweep(root, parent string, opts SweepOptions) SweepReport {
	return workerworktree.SweepTopologyCandidates(root, parent, nil, opts)
}

// Counts returns scanned, reaped, kept from a report. scanned == len(Candidates)
// (kept ones included), kept == scanned - reaped.
func Counts(r SweepReport) (scanned, reaped, kept int) {
	scanned = len(r.Candidates)
	reaped = r.Reaped
	kept = scanned - reaped
	return scanned, reaped, kept
}
