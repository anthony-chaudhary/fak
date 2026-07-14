package milestonereport

// velocity.go is the CODE-MOVEMENT lens the milestone report renders beside the
// epic roadmap (#2494). The epic bars say what CLOSED; this says where the code
// actually MOVED, folded purely from the append-only module-versions ledger
// (schema fak-module-versions/1) via modver.Trend — the same independently-tested
// fold `fak version modules` reads. Reading both lenses on one report is what
// catches the "issues closing, code dormant" dissonance the parent spine
// (version-everything) exists to surface. The fold here is PURE — ledger bytes in,
// a Velocity out; no git, no clock, no I/O — so the report stays unit-testable and
// the CLI owns the one ledger read (WithVelocity, mirroring WithTrend).

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

// DefaultModuleLedgerRel is the committed, append-only module-versions ledger
// (schema fak-module-versions/1) that `fak version modules --stamp` writes — the
// source of the velocity lens. It lives under docs/ so it is durable trunk
// evidence, not a regenerable build artifact.
const DefaultModuleLedgerRel = "docs/nightrun/module-versions.jsonl"

// velocityTopCap bounds the rendered mover list so the card stays scannable, the
// same discipline Maturity.Worst uses.
const velocityTopCap = 6

// Velocity is the code-movement lens rendered beside the roadmap. It is a MIRROR,
// not a gate: it never touches the report's OK/verdict, it only reports where
// trunk code moved over the ledger window. Rows are the top rev movers (biggest
// delta first); Err is set when no ledger row could be measured, so an empty
// ledger renders an honest "unmeasured" line rather than a fabricated zero.
type Velocity struct {
	Rows          []VelocityRow `json:"rows"`
	Modules       int           `json:"modules"`         // distinct modules the ledger has seen
	LedgerRows    int           `json:"ledger_rows"`     // total parseable ledger rows folded
	Movers        int           `json:"movers"`          // modules whose rev advanced over the window
	TotalRevDelta int           `json:"total_rev_delta"` // summed rev movement across all modules
	Window        [2]string     `json:"window"`          // [earliest ts, latest ts] across the ledger
	Err           string        `json:"err,omitempty"`
	OK            bool          `json:"ok"`
}

// VelocityRow is one lane's rev movement over the ledger window: how many commits
// it added (RevDelta) and where its revision now stands (LastRev). It is the
// per-module projection of modver.ModuleTrend, carrying only what the milestone
// card renders.
type VelocityRow struct {
	Module     string `json:"module"`
	Kind       string `json:"kind"`
	RevDelta   int    `json:"rev_delta"`   // commits added over the window (LastRev - FirstRev)
	LastRev    int    `json:"last_rev"`    // the module's revision at the window close
	Stamps     int    `json:"stamps"`      // ledger rows seen for this module
	LastCommit string `json:"last_commit"` // the commit that last moved it
}

// InterpretVelocity folds the module-versions ledger into the code-movement lens
// via modver.Trend (which already sorts modules by rev-delta descending, ties by
// name — so the top movers fall out in order). It is PURE: ledger bytes in, a
// Velocity out. An empty or all-scar ledger is an ERRORED lens (nothing to
// measure), never a silent zero — matching the maturity/roadmap honesty rule. Only
// modules that actually moved (RevDelta > 0) enter the rendered top list; a dormant
// lane is not velocity.
func InterpretVelocity(ledger []byte) Velocity {
	tr := modver.Trend(ledger)
	v := Velocity{
		Modules:    len(tr.Modules),
		LedgerRows: tr.Rows,
		Window:     tr.Window,
	}
	if tr.Rows == 0 {
		v.Err = "empty module-versions ledger — nothing to measure"
		return v
	}
	for _, m := range tr.Modules {
		v.TotalRevDelta += m.RevDelta
		if m.RevDelta > 0 {
			v.Movers++
		}
	}
	for _, m := range tr.Modules {
		if m.RevDelta <= 0 {
			continue // a lane with no movement is not velocity; keep it out of the top list
		}
		if len(v.Rows) >= velocityTopCap {
			break
		}
		v.Rows = append(v.Rows, VelocityRow{
			Module:     m.Module,
			Kind:       m.Kind,
			RevDelta:   m.RevDelta,
			LastRev:    m.LastRev,
			Stamps:     m.Stamps,
			LastCommit: m.LastCommit,
		})
	}
	v.OK = true
	return v
}

// WithVelocity attaches the code-movement lens, returning the reconciled copy.
// Like WithTrend/WithProgramScorecard it keeps Fold pure of the ledger read: the
// CLI folds the ledger with InterpretVelocity and attaches the result here. The
// lens is a mirror beside the roadmap, so it never touches OK/verdict.
func (r Report) WithVelocity(v Velocity) Report {
	r.Velocity = &v
	return r
}

// renderVelocity renders the code-movement lens beside the epic roadmap: a header
// counting movers vs total modules, then the top lane rev movers. An unmeasured
// ledger renders one honest line rather than a fabricated zero. Returns nil when no
// lens is attached, so Render appends nothing (append(x, nil...) is a no-op).
func renderVelocity(v *Velocity) []string {
	if v == nil {
		return nil
	}
	if v.Err != "" {
		return []string{"      module velocity: unmeasured (" + v.Err + ")"}
	}
	lines := []string{fmt.Sprintf("      module velocity: %d mover(s) of %d module(s), %+d rev(s) over the ledger window",
		v.Movers, v.Modules, v.TotalRevDelta)}
	for _, row := range v.Rows {
		lines = append(lines, "        "+velocityRowLine(row))
	}
	return lines
}

// velocityRowLine renders one lane's rev movement: its signed delta over the window
// and the revision it now stands at, so an operator reads "which lanes moved, and
// how far" beside "which epics closed".
func velocityRowLine(row VelocityRow) string {
	kind := ""
	if row.Kind != "" {
		kind = " [" + row.Kind + "]"
	}
	return fmt.Sprintf("%s%s — %+d rev(s) -> r%d (%d stamp(s))", row.Module, kind, row.RevDelta, row.LastRev, row.Stamps)
}
