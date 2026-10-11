// Package worktreecheck re-exports fak's owner-aware verify-checkout sweep
// (internal/workerworktree) so out-of-tree callers can run it while importing
// only fak/pkg/*. Every name is a type alias or a stateless wrapper; values are
// identical to their internal counterparts.
package worktreecheck

import (
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type (
	// SweepOptions supplies the clock, liveness probes, and apply opt-in.
	SweepOptions = workerworktree.CandidateSweepOptions
	// SweepReport lists every candidate found, kept ones included.
	SweepReport = workerworktree.CandidateSweepReport
	// SweepItem is one candidate with its decision and evidence.
	SweepItem = workerworktree.CandidateSweepItem
)

// TopologyCandidateParents returns the distinct parents a full sweep must scan
// for root: the topology parent (beside root when go.work escapes it) and always
// os.TempDir(), so a mixed history is covered. Order is stable; dedup is
// case-insensitive on Windows.
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

// Sweep runs the owner-aware topology-candidate sweep for one parent using host git.
func Sweep(root, parent string, opts SweepOptions) SweepReport {
	return workerworktree.SweepTopologyCandidates(root, parent, nil, opts)
}

// Counts returns scanned, reaped, and kept from a report, where kept == scanned -
// reaped: in a dry run kept includes the WouldReap-eligible candidates.
func Counts(r SweepReport) (scanned, reaped, kept int) {
	scanned = len(r.Candidates)
	reaped = r.Reaped
	kept = scanned - reaped
	return scanned, reaped, kept
}
